"""
retriever.py — query Pinecone to retrieve relevant chunks

Embeds the user query with Gemini (task_type=RETRIEVAL_QUERY),
queries Pinecone for top-K nearest vectors, filters by score threshold,
and returns a ranked list of chunk dicts ready for prompt_builder.py.

Single public function:
    retrieve(query: str) -> list[dict]

Each returned chunk dict:
{
    "chunk_id":  "asyncio-task_004",
    "text":      "...",
    "url":       "https://docs.python.org/3/library/asyncio-task.html",
    "title":     "Coroutines and Tasks",
    "has_code":  true,
    "score":     0.87
}

Run directly to test retrieval interactively:
    python -m src.retriever
"""

import google.generativeai as genai
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.table import Table

from src.config import (
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
)

console = Console()

# ── Clients (initialised once at import time) ──────────────────
genai.configure(api_key=GEMINI_API_KEY)
_pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
_index = None   # lazy-loaded on first retrieve() call


def _get_index():
    """Lazy-load the Pinecone index — avoids connection overhead at import."""
    global _index
    if _index is None:
        _index = _pinecone_client.Index(PINECONE_INDEX_NAME)
    return _index


# ── Query embedding ────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _embed_query(query: str) -> list[float]:
    """
    Embed a single query string using Gemini.

    Uses task_type=RETRIEVAL_QUERY — asymmetric from RETRIEVAL_DOCUMENT
    used at index time. This is intentional and improves retrieval quality.
    """
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    embedding = result["embedding"]
    # Single string returns a flat list — return as-is
    if embedding and isinstance(embedding[0], list):
        return embedding[0]
    return embedding


# ── Retrieval ──────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:           natural language question or code snippet
        top_k:           number of candidates to fetch from Pinecone
        score_threshold: minimum cosine similarity score to include (0–1)

    Returns:
        list of chunk dicts sorted by score descending, filtered by threshold.
        Empty list if no results meet the threshold.
    """
    if not query or not query.strip():
        return []

    # Step 1 — embed the query
    try:
        query_vector = _embed_query(query.strip())
    except Exception as e:
        console.print(f"[red]Failed to embed query:[/red] {e}")
        return []

    # Step 2 — query Pinecone
    try:
        index = _get_index()
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as e:
        console.print(f"[red]Pinecone query failed:[/red] {e}")
        return []

    # Step 3 — parse and filter results
    matches = response.get("matches", [])
    chunks: list[dict] = []

    for match in matches:
        score = match.get("score", 0.0)
        if score < score_threshold:
            continue  # drop low-confidence results

        meta = match.get("metadata", {})
        chunks.append({
            "chunk_id":  match.get("id", ""),
            "text":      meta.get("text", ""),
            "url":       meta.get("url", ""),
            "title":     meta.get("title", ""),
            "has_code":  meta.get("has_code", False),
            "score":     round(score, 4),
        })

    # Already sorted by Pinecone (highest score first) — preserve order
    return chunks


# ── Debug helper ───────────────────────────────────────────────

def print_results(chunks: list[dict], query: str) -> None:
    """Pretty-print retrieval results as a Rich table."""
    if not chunks:
        console.print(f"\n[yellow]No results above threshold for:[/yellow] {query}\n")
        return

    table = Table(
        title=f"Results for: \"{query[:60]}\"",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("Score", justify="right", width=6)
    table.add_column("Title", width=28)
    table.add_column("Chunk ID", width=22)
    table.add_column("Preview", width=55)

    for chunk in chunks:
        preview = chunk["text"][:120].replace("\n", " ")
        if len(chunk["text"]) > 120:
            preview += "..."
        table.add_row(
            f"{chunk['score']:.3f}",
            chunk["title"][:28],
            chunk["chunk_id"],
            preview,
        )

    console.print(table)
    console.print(
        f"[dim]{len(chunks)} chunks retrieved "
        f"(threshold={RETRIEVAL_SCORE_THRESHOLD})[/dim]\n"
    )


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test retrieval interactively")
    ap.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string to test. If omitted, enters interactive mode.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"Number of results to fetch (default: {RETRIEVAL_TOP_K})",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=RETRIEVAL_SCORE_THRESHOLD,
        help=f"Minimum score threshold (default: {RETRIEVAL_SCORE_THRESHOLD})",
    )
    ap.add_argument(
        "--show-text",
        action="store_true",
        help="Print full chunk text for each result",
    )
    args = ap.parse_args()

    def run_query(q: str) -> None:
        console.print(f"\n[dim]Querying: {q}[/dim]")
        results = retrieve(q, top_k=args.top_k, score_threshold=args.threshold)
        print_results(results, q)
        if args.show_text and results:
            for chunk in results:
                console.rule(f"[dim]{chunk['chunk_id']} (score={chunk['score']})[/dim]")
                console.print(chunk["text"])
                console.print()

    if args.query:
        run_query(args.query)
    else:
        # Interactive mode
        console.print("\n[bold]Retriever interactive mode[/bold]")
        console.print("[dim]Type a query and press Enter. Ctrl+C to exit.[/dim]\n")
        while True:
            try:
                query = input("Query > ").strip()
                if query:
                    run_query(query)
            except KeyboardInterrupt:
                console.print("\n[dim]Bye.[/dim]")
                break
            except EOFError:
                break