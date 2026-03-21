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
    JINA_RERANKER_TOP_N,
)

console = Console()

# ── Clients (initialised once at import time) ──────────────────
genai.configure(api_key=GEMINI_API_KEY)
_pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
_index = None   # lazy-loaded on first retrieve() call


def _get_index():
    """Lazy-load Pinecone index — avoids connection overhead at import."""
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
    Embed a query string using Gemini RETRIEVAL_QUERY task type.
    Asymmetric from RETRIEVAL_DOCUMENT used at index time —
    intentional, improves retrieval quality.
    """
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    embedding = result["embedding"]
    if embedding and isinstance(embedding[0], list):
        return embedding[0]
    return embedding


# ── Pinecone query ─────────────────────────────────────────────

def _query_pinecone(
    query_vector:    list[float],
    top_k:           int,
    score_threshold: float,
    source_filter:   str | None,
) -> list[dict]:
    """
    Query Pinecone and return filtered chunk dicts.

    Args:
        query_vector:   embedded query
        top_k:          candidates to fetch
        score_threshold: min cosine similarity to include
        source_filter:  "stackoverflow" | "cp_algorithms" | None (both)

    Returns:
        list of chunk dicts with full metadata, sorted by score desc
    """
    # Build optional metadata filter for Pinecone
    pinecone_filter = None
    if source_filter:
        pinecone_filter = {"source_type": {"$eq": source_filter}}

    try:
        index    = _get_index()
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
    except Exception as e:
        console.print(f"[red]Pinecone query failed:[/red] {e}")
        return []

    chunks: list[dict] = []
    for match in response.get("matches", []):
        score = match.get("score", 0.0)
        if score < score_threshold:
            continue

        meta = match.get("metadata", {})
        chunks.append({
            "chunk_id":    match.get("id", ""),
            "text":        meta.get("text", ""),
            "url":         meta.get("url", ""),
            "title":       meta.get("title", ""),
            "source_type": meta.get("source_type", "unknown"),
            "so_score":    meta.get("so_score"),
            "is_accepted": meta.get("is_accepted"),
            "topic":       meta.get("topic"),
            "has_code":    meta.get("has_code", False),
            "score":       round(score, 4),
        })

    return chunks


# ── Main retrieve ──────────────────────────────────────────────

def retrieve(
    query:           str,
    top_k:           int        = RETRIEVAL_TOP_K,
    score_threshold: float      = RETRIEVAL_SCORE_THRESHOLD,
    source_filter:   str | None = None,
    rerank:          bool       = True,
    top_n:           int        = JINA_RERANKER_TOP_N,
) -> list[dict]:
    """
    Retrieve and rerank the most relevant chunks for a DSA query.

    Args:
        query:           problem description or code snippet
        top_k:           candidates to fetch from Pinecone (default 20)
        score_threshold: min vector similarity score (default 0.40)
        source_filter:   restrict to "stackoverflow" or "cp_algorithms"
                         None = search both sources (recommended)
        rerank:          run Jina reranker on results (default True)
                         set False to see raw Pinecone order for debugging
        top_n:           chunks to keep after reranking (default 5)

    Returns:
        list of chunk dicts sorted by rerank_score (or vector score if
        rerank=False), best first. Empty list if nothing found.
    """
    if not query or not query.strip():
        return []

    # Step 1 — embed query
    try:
        query_vector = _embed_query(query.strip())
    except Exception as e:
        console.print(f"[red]Failed to embed query:[/red] {e}")
        return []

    # Step 2 — fetch from Pinecone
    chunks = _query_pinecone(
        query_vector, top_k, score_threshold, source_filter
    )

    if not chunks:
        return []

    # Step 3 — rerank with Jina
    if rerank and len(chunks) > 1:
        from src.reranker import rerank as jina_rerank
        chunks = jina_rerank(query.strip(), chunks, top_n=top_n)

    return chunks


# ── Debug helpers ──────────────────────────────────────────────

def print_results(
    chunks:  list[dict],
    query:   str,
    reranked: bool = True,
) -> None:
    """Pretty-print retrieval results as a Rich table."""
    if not chunks:
        console.print(
            f"\n[yellow]No results found for:[/yellow] {query}\n"
            f"[dim]Try lowering --threshold or removing --source filter.[/dim]\n"
        )
        return

    score_col = "Rerank" if reranked else "Vec score"
    table = Table(
        title=f'Results for: "{query[:55]}"',
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column(score_col,   justify="right", width=9)
    table.add_column("Source",    width=14)
    table.add_column("Title",     width=26)
    table.add_column("Chunk ID",  width=22)
    table.add_column("Preview",   width=40)

    for chunk in chunks:
        score   = chunk.get("rerank_score", chunk.get("score", 0))
        src     = chunk.get("source_type", "?")
        so_info = ""
        if src == "stackoverflow" and chunk.get("so_score") is not None:
            acc     = " ✓" if chunk.get("is_accepted") else ""
            so_info = f" [dim](↑{chunk['so_score']}{acc})[/dim]"
        preview = chunk["text"][:80].replace("\n", " ")
        if len(chunk["text"]) > 80:
            preview += "..."
        table.add_row(
            f"{score:.4f}",
            f"{src}{so_info}",
            chunk["title"][:26],
            chunk["chunk_id"],
            preview,
        )

    console.print(table)
    console.print(
        f"[dim]{len(chunks)} chunks  "
        f"threshold={RETRIEVAL_SCORE_THRESHOLD}  "
        f"reranked={reranked}[/dim]\n"
    )


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test DSA retrieval + reranking")
    ap.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query to test. Omit for interactive mode.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"Candidates to fetch from Pinecone (default: {RETRIEVAL_TOP_K})",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=JINA_RERANKER_TOP_N,
        help=f"Chunks to keep after reranking (default: {JINA_RERANKER_TOP_N})",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=RETRIEVAL_SCORE_THRESHOLD,
        help=f"Min vector similarity score (default: {RETRIEVAL_SCORE_THRESHOLD})",
    )
    ap.add_argument(
        "--source",
        choices=["stackoverflow", "cp_algorithms"],
        default=None,
        help="Filter to a single source type",
    )
    ap.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking — show raw Pinecone order",
    )
    ap.add_argument(
        "--show-text",
        action="store_true",
        help="Print full chunk text for each result",
    )
    args = ap.parse_args()

    def run_query(q: str) -> None:
        use_rerank = not args.no_rerank
        console.print(
            f"\n[dim]Query    : {q}[/dim]\n"
            f"[dim]Source   : {args.source or 'all'}[/dim]\n"
            f"[dim]Rerank   : {use_rerank}[/dim]\n"
        )

        results = retrieve(
            query=q,
            top_k=args.top_k,
            score_threshold=args.threshold,
            source_filter=args.source,
            rerank=use_rerank,
            top_n=args.top_n,
        )
        print_results(results, q, reranked=use_rerank)

        if args.show_text and results:
            for chunk in results:
                score = chunk.get("rerank_score", chunk.get("score", 0))
                console.rule(
                    f"[dim]{chunk['chunk_id']}  "
                    f"[{chunk.get('source_type','?')}]  "
                    f"score={score:.4f}[/dim]"
                )
                console.print(chunk["text"])
                console.print()

    if args.query:
        run_query(args.query)
    else:
        console.print("\n[bold]Retriever interactive mode[/bold]")
        console.print(
            f"[dim]source={args.source or 'all'}  "
            f"rerank={not args.no_rerank}  "
            f"top_k={args.top_k} → top_n={args.top_n}[/dim]\n"
            "[dim]Ctrl+C to exit.[/dim]\n"
        )
        while True:
            try:
                q = input("Query > ").strip()
                if q:
                    run_query(q)
            except KeyboardInterrupt:
                console.print("\n[dim]Bye.[/dim]")
                break
            except EOFError:
                break