"""
pipeline.py — end-to-end hint() function for the DSA Hint Engine

Wires retriever → reranker → prompt_builder → generator.

Single public function:
    hint(problem, code_snippet=None) -> dict

Returned dict:
{
    "hint":          "...",
    "pattern":       "Sliding Window",
    "sources":       ["url1", "url2"],
    "chunks_used":   5,
    "input_tokens":  412,
    "output_tokens": 284,
    "model":         "llama-3.3-70b-versatile",
    "elapsed_sec":   2.4
}

Run directly:
    python -m src.pipeline
    python -m src.pipeline --problem "two sum" --code "..."
"""

import re
import time
from rich.console import Console
from rich.rule import Rule

from src.retriever      import retrieve
from src.reranker       import rerank
from src.prompt_builder import build_messages, count_prompt_tokens
from src.generator      import generate
from src.config         import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD, JINA_RERANKER_TOP_N

console = Console()

# Regex to extract the algorithm pattern from the LLM response.
# Matches several phrasings the LLM commonly uses:
#   "Pattern identified: X", "Pattern: X", "Approach: X",
#   "This is a sliding window problem", "use dynamic programming", etc.
_PATTERN_PHRASES = re.compile(
    r"(?:"
    r"pattern\s*(?:identified|used|here|to use)?\s*[:\-–]?\s*"
    r"|approach\s*[:\-–]\s*"
    r"|algorithm\s*[:\-–]\s*"
    r")"
    r"\**([\w][\w ,/\-]+?)\**"
    r"(?:\.|,|\n|$)",
    re.IGNORECASE,
)
_USE_PATTERN = re.compile(
    r"(?:use|using|apply|applies|with)\s+(?:a\s+|an\s+|the\s+)?"
    r"((?:sliding window|two pointer|hash map|dynamic programming|BFS|DFS"
    r"|binary search|backtracking|greedy|divide and conquer|trie|heap"
    r"|union.find|topological sort|monotonic stack|prefix sum)[\w ]*)",
    re.IGNORECASE,
)


def _extract_pattern(text: str) -> str:
    """Pull the algorithm pattern name from the LLM response."""
    m = _PATTERN_PHRASES.search(text)
    if m:
        return m.group(1).strip().strip("*_")
    m = _USE_PATTERN.search(text)
    if m:
        return m.group(1).strip().strip("*_")
    return "Unknown"


# ── Core pipeline ──────────────────────────────────────────────

def hint(
    problem:        str,
    code_snippet:   str | None = None,
    top_k:          int        = RETRIEVAL_TOP_K,
    score_threshold: float     = RETRIEVAL_SCORE_THRESHOLD,
    top_n:          int        = JINA_RERANKER_TOP_N,
    stream:         bool       = True,
    verbose:        bool       = False,
) -> dict:
    """
    Generate a DSA hint for a problem + optional stuck code.

    Args:
        problem:         DSA problem description
        code_snippet:    student's current code (optional)
        top_k:           chunks to retrieve from Pinecone
        score_threshold: min similarity score for retrieval
        top_n:           chunks to keep after reranking
        stream:          stream tokens to console (False in eval mode)
        verbose:         print retrieval + reranking details

    Returns:
        result dict with hint, pattern, sources, and usage stats
    """
    t_start = time.monotonic()

    # ── Step 1: Build retrieval query ──────────────────────────
    # Combine problem + code for better semantic matching
    if code_snippet and code_snippet.strip():
        retrieval_query = f"{problem}\n\n{code_snippet.strip()}"
    else:
        retrieval_query = problem

    # ── Step 2: Retrieve from Pinecone ─────────────────────────
    chunks = retrieve(
        retrieval_query,
        top_k=top_k,
        score_threshold=score_threshold,
        rerank=False,   # pipeline reranks explicitly below with clean `problem` query
    )

    if verbose:
        console.print(f"[dim]Retrieved {len(chunks)} chunks from Pinecone[/dim]")
        for c in chunks:
            console.print(
                f"  [dim]{c['score']:.3f}  [{c.get('source_type','?')}]  "
                f"{c['chunk_id']}[/dim]"
            )
        console.print()

    if not chunks:
        console.print(
            "[yellow]No relevant chunks found.[/yellow] "
            "Try lowering --threshold or rephrasing the problem."
        )

    # ── Step 3: Rerank ─────────────────────────────────────────
    # Use problem (short) as the rerank query — not retrieval_query
    # (problem + code). Jina 422s on queries longer than ~500 chars.
    reranked = rerank(problem, chunks, top_n=top_n)

    if verbose and reranked:
        console.print(f"[dim]After reranking — top {len(reranked)} chunks:[/dim]")
        for c in reranked:
            console.print(
                f"  [dim]{c.get('rerank_score', '?'):.4f}  "
                f"[{c.get('source_type','?')}]  {c['chunk_id']}[/dim]"
            )
        console.print()

    # ── Step 4: Build prompt ───────────────────────────────────
    messages = build_messages(
        chunks=reranked,
        question=problem,
        code_snippet=code_snippet,
    )

    if verbose:
        console.print(
            f"[dim]Prompt tokens: {count_prompt_tokens(messages)}[/dim]\n"
        )

    # ── Step 5: Generate ───────────────────────────────────────
    result  = generate(messages, stream=stream)
    elapsed = time.monotonic() - t_start
    pattern = _extract_pattern(result["explanation"])

    if verbose:
        console.print(
            f"\n[dim]Done in {elapsed:.1f}s — "
            f"in={result['input_tokens']} out={result['output_tokens']} "
            f"pattern={pattern}[/dim]"
        )

    return {
        "hint":          result["explanation"],
        "pattern":       pattern,
        "sources":       result["sources"],
        "chunks_used":   len(reranked),
        "input_tokens":  result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "model":         result["model"],
        "elapsed_sec":   round(elapsed, 2),
    }


# ── CLI ────────────────────────────────────────────────────────

def _interactive_mode() -> None:
    console.print("\n[bold]DSA Hint Engine — Interactive Mode[/bold]")
    console.print("[dim]Describe your problem, paste your code, get a hint. Ctrl+C to exit.[/dim]\n")

    while True:
        try:
            problem = input("Problem > ").strip()
            if not problem:
                continue

            console.print(
                "[dim]Paste your stuck code (Enter twice to skip, "
                "or Enter twice after pasting):[/dim]"
            )
            lines: list[str] = []
            while True:
                line = input()
                if line == "" and (not lines or lines[-1] == ""):
                    break
                lines.append(line)
            code = "\n".join(lines).strip() or None

            console.print()
            console.rule("[dim]Hint[/dim]")

            result = hint(
                problem=problem,
                code_snippet=code,
                stream=True,
                verbose=True,
            )

            console.print(f"\n[bold]Pattern:[/bold] {result['pattern']}")
            if result["sources"]:
                console.print("\n[bold]Sources:[/bold]")
                for url in result["sources"]:
                    console.print(f"  [blue]{url}[/blue]")
            console.rule()
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Bye.[/dim]\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DSA Hint Engine")
    ap.add_argument("--problem",   type=str,   default=None)
    ap.add_argument("--code",      type=str,   default=None)
    ap.add_argument("--top-k",     type=int,   default=RETRIEVAL_TOP_K)
    ap.add_argument("--threshold", type=float, default=RETRIEVAL_SCORE_THRESHOLD)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    if args.problem:
        console.print()
        console.rule("[dim]Hint[/dim]")

        result = hint(
            problem=args.problem,
            code_snippet=args.code,
            top_k=args.top_k,
            score_threshold=args.threshold,
            stream=not args.no_stream,
            verbose=args.verbose,
        )

        console.print(f"\n[bold]Pattern:[/bold] {result['pattern']}")
        if result["sources"]:
            console.print("\n[bold]Sources:[/bold]")
            for url in result["sources"]:
                console.print(f"  [blue]{url}[/blue]")
        console.print(
            f"\n[dim]chunks={result['chunks_used']} "
            f"in={result['input_tokens']} out={result['output_tokens']} "
            f"time={result['elapsed_sec']}s[/dim]\n"
        )
    else:
        _interactive_mode()