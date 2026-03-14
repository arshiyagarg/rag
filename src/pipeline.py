"""
pipeline.py — end-to-end explain() function

Wires retriever → prompt_builder → generator into a single call.

Single public function:
    explain(question, code_snippet=None) -> dict

Returned dict:
{
    "explanation":   "...",
    "sources":       ["url1", "url2"],
    "chunks_used":   3,
    "input_tokens":  412,
    "output_tokens": 284,
    "model":         "llama-3.3-70b-versatile",
    "question":      "...",
    "code_snippet":  "...",
}

Run directly for interactive use:
    python -m src.pipeline
    python -m src.pipeline --question "how does asyncio.gather work"
"""

import time
from rich.console import Console
from rich.rule import Rule

from src.retriever      import retrieve
from src.prompt_builder import build_messages, count_prompt_tokens
from src.generator      import generate
from src.config         import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD

console = Console()


# ── Core pipeline ──────────────────────────────────────────────

def explain(
    question: str,
    code_snippet: str | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    stream: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Explain a code snippet or answer an asyncio question using RAG.

    Args:
        question:        what the user wants to understand
        code_snippet:    optional code to explain (paste as string)
        top_k:           max chunks to retrieve from Pinecone
        score_threshold: min similarity score to include a chunk
        stream:          stream LLM tokens to console (set False in eval)
        verbose:         print retrieval details and token counts

    Returns:
        result dict with explanation, sources, and usage stats
    """
    t_start = time.monotonic()

    # ── Step 1: Retrieve ───────────────────────────────────────
    # Build a composite query — include code if provided so the
    # retriever finds chunks that match both the question and code context
    if code_snippet and code_snippet.strip():
        retrieval_query = f"{question}\n\n{code_snippet.strip()}"
    else:
        retrieval_query = question

    chunks = retrieve(
        retrieval_query,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    if verbose:
        console.print(
            f"[dim]Retrieved {len(chunks)} chunks "
            f"(top score: {chunks[0]['score'] if chunks else 'n/a'})[/dim]"
        )
        for c in chunks:
            console.print(
                f"  [dim]{c['score']:.3f}  {c['chunk_id']}  {c['title'][:40]}[/dim]"
            )
        console.print()

    if not chunks:
        console.print(
            "[yellow]No relevant chunks found.[/yellow] "
            "Try lowering --threshold or rephrasing the question."
        )

    # ── Step 2: Build prompt ───────────────────────────────────
    messages = build_messages(
        chunks=chunks,
        question=question,
        code_snippet=code_snippet,
    )

    if verbose:
        total_tokens = count_prompt_tokens(messages)
        console.print(f"[dim]Prompt tokens: {total_tokens}[/dim]\n")

    # ── Step 3: Generate ───────────────────────────────────────
    result = generate(messages, stream=stream)

    t_elapsed = time.monotonic() - t_start

    if verbose:
        console.print(
            f"\n[dim]Done in {t_elapsed:.1f}s — "
            f"in={result['input_tokens']} out={result['output_tokens']} tokens[/dim]"
        )

    return {
        "explanation":   result["explanation"],
        "sources":       result["sources"],
        "chunks_used":   len(chunks),
        "input_tokens":  result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "model":         result["model"],
        "question":      question,
        "code_snippet":  code_snippet or "",
        "elapsed_sec":   round(t_elapsed, 2),
    }


# ── CLI ────────────────────────────────────────────────────────

def _interactive_mode() -> None:
    """REPL loop — ask questions interactively."""
    console.print("\n[bold]Code Explainer v0 — Interactive Mode[/bold]")
    console.print("[dim]Paste code then press Enter twice. Type your question. Ctrl+C to exit.[/dim]\n")

    while True:
        try:
            # Collect optional code snippet
            console.print("[dim]Code snippet (optional — paste and press Enter twice, or just Enter to skip):[/dim]")
            lines: list[str] = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                if line == "" and not lines:
                    break
                lines.append(line)
            code = "\n".join(lines).strip() or None

            # Get question
            question = input("Question > ").strip()
            if not question:
                continue

            console.print()
            console.rule("[dim]Explanation[/dim]")

            result = explain(
                question=question,
                code_snippet=code,
                stream=True,
                verbose=True,
            )

            # Print sources footer
            if result["sources"]:
                console.print("\n[bold]Sources:[/bold]")
                for url in result["sources"]:
                    console.print(f"  [blue]{url}[/blue]")
            else:
                console.print("\n[dim]No sources cited.[/dim]")

            console.rule()
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Bye.[/dim]\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Explain asyncio code using RAG")
    ap.add_argument("--question",  type=str, default=None, help="Question to answer")
    ap.add_argument("--code",      type=str, default=None, help="Code snippet to explain")
    ap.add_argument("--top-k",     type=int, default=RETRIEVAL_TOP_K)
    ap.add_argument("--threshold", type=float, default=RETRIEVAL_SCORE_THRESHOLD)
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    ap.add_argument("--verbose",   action="store_true", help="Print retrieval + token details")
    args = ap.parse_args()

    if args.question:
        # Single-shot mode
        console.print()
        console.rule("[dim]Explanation[/dim]")

        result = explain(
            question=args.question,
            code_snippet=args.code,
            top_k=args.top_k,
            score_threshold=args.threshold,
            stream=not args.no_stream,
            verbose=args.verbose,
        )

        if result["sources"]:
            console.print("\n[bold]Sources:[/bold]")
            for url in result["sources"]:
                console.print(f"  [blue]{url}[/blue]")

        console.print(
            f"\n[dim]chunks={result['chunks_used']} "
            f"in={result['input_tokens']} "
            f"out={result['output_tokens']} "
            f"time={result['elapsed_sec']}s[/dim]\n"
        )

    else:
        _interactive_mode()