"""
reranker.py — rerank retrieved chunks using Jina AI reranker API

Takes the top-K chunks from retriever.py, sends them to
jina-reranker-v2-base-multilingual, and returns the top-N
reranked by true relevance rather than vector similarity alone.

Single public function:
    rerank(query, chunks, top_n) -> list[dict]

Each returned chunk is the original dict with one extra field added:
    "rerank_score": 0.8821   # Jina relevance score (0–1)

Run directly to test:
    python -m src.reranker
"""

import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console

from src.config import JINA_API_KEY, JINA_RERANKER_MODEL, JINA_RERANKER_TOP_N, JINA_URL

console = Console()

# ── API call ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_jina(query: str, documents: list[str], top_n: int) -> list[dict]:
    """
    Call Jina reranker API.
    Returns raw results list: [{index, relevance_score, document}, ...]
    """
    resp = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":     JINA_RERANKER_MODEL,
            "query":     query,
            "documents": documents,
            "top_n":     top_n,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


# ── Main rerank ────────────────────────────────────────────────

def rerank(
    query:   str,
    chunks:  list[dict],
    top_n:   int = JINA_RERANKER_TOP_N,
) -> list[dict]:
    """
    Rerank a list of retrieved chunks by true relevance to the query.

    Args:
        query:  the user's question (same query used for retrieval)
        chunks: list of chunk dicts from retriever.py
        top_n:  how many top chunks to return after reranking

    Returns:
        top_n chunks sorted by rerank_score descending.
        Each chunk gets a new "rerank_score" field added.
        Falls back to original order if API call fails.
    """
    if not chunks:
        return []

    # Nothing to rerank if we already have fewer chunks than top_n
    if len(chunks) <= top_n:
        return chunks

    documents = [c["text"] for c in chunks]

    try:
        results = _call_jina(query, documents, top_n)
    except Exception as e:
        console.print(
            f"[yellow]Reranker unavailable:[/yellow] {e} — "
            "falling back to original retrieval order."
        )
        return chunks[:top_n]

    # Map reranked results back to original chunk dicts
    reranked: list[dict] = []
    for r in results:
        idx   = r["index"]
        score = round(r["relevance_score"], 4)
        chunk = {**chunks[idx], "rerank_score": score}
        reranked.append(chunk)

    return reranked


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.retriever import retrieve
    from rich.table import Table
    import argparse

    ap = argparse.ArgumentParser(description="Test Jina reranker against live Pinecone results")
    ap.add_argument(
        "query",
        nargs="?",
        default="how does asyncio.gather handle exceptions",
        help="Query to retrieve and rerank",
    )
    ap.add_argument("--top-k",  type=int, default=10, help="Chunks to retrieve before reranking")
    ap.add_argument("--top-n",  type=int, default=5,  help="Chunks to keep after reranking")
    args = ap.parse_args()

    console.print(f"\n[bold]Reranker test[/bold]")
    console.print(f"[dim]Query  : {args.query}[/dim]")
    console.print(f"[dim]top_k  : {args.top_k} (retrieve) → top_n: {args.top_n} (rerank)[/dim]\n")

    # Step 1 — retrieve
    console.print("[dim]Step 1: retrieving from Pinecone...[/dim]")
    chunks = retrieve(args.query, top_k=args.top_k)
    console.print(f"[dim]Got {len(chunks)} chunks.[/dim]\n")

    # Step 2 — rerank
    console.print("[dim]Step 2: reranking with Jina...[/dim]")
    reranked = rerank(args.query, chunks, top_n=args.top_n)

    # Before/after comparison table
    table = Table(
        title="Before vs after reranking",
        show_header=True,
        header_style="bold",
        show_lines=False,
    )
    table.add_column("Rank",     justify="center", width=5)
    table.add_column("Vec score",  justify="right",  width=10)
    table.add_column("Rerank score", justify="right", width=13)
    table.add_column("Chunk ID",   width=22)
    table.add_column("Title",      width=28)

    for i, chunk in enumerate(reranked, 1):
        table.add_row(
            str(i),
            f"{chunk.get('score', 0):.3f}",
            f"[bold]{chunk.get('rerank_score', 0):.4f}[/bold]",
            chunk["chunk_id"],
            chunk["title"][:28],
        )

    console.print(table)
    console.print(
        f"\n[dim]Reranking complete — "
        f"{len(chunks)} → {len(reranked)} chunks.[/dim]\n"
    )