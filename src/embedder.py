"""
embedder.py — embed chunks with Gemini and upsert to Pinecone

Gemini free tier limits:
  - 100 requests/minute  (we use a rolling window to stay under)
  - 15 RPM on some accounts — if you hit 429s still, set EMBED_RPM_LIMIT=15 in .env

Run directly:
    python -m src.embedder
    python -m src.embedder --dry-run
    python -m src.embedder --delete-index
"""

import time
from collections import deque
from pathlib import Path

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import (
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_DIMENSION,
)
from src.chunker import load_chunks

console = Console()

# ── Rate limit config ──────────────────────────────────────────
EMBED_BATCH_SIZE = 20    # texts per embed API call
RPM_LIMIT        = 95    # stay 5 under the 100 RPM free tier cap
UPSERT_BATCH_SIZE = 100  # vectors per Pinecone upsert call


# ── Rate limiter ───────────────────────────────────────────────

class RpmLimiter:
    """
    Sliding-window rate limiter.
    Tracks timestamps of the last N requests in a 60s window.
    Blocks (sleeps) when the window is full.
    """

    def __init__(self, rpm: int = RPM_LIMIT):
        self.rpm = rpm
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        """Call before every API request. Sleeps if needed."""
        now = time.monotonic()

        # Drop timestamps older than 60 seconds
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.rpm:
            # Window is full — sleep until the oldest request is > 60s ago
            sleep_for = 60 - (now - self._timestamps[0]) + 0.5  # +0.5s buffer
            console.print(
                f"\n[yellow]Rate limit:[/yellow] {self.rpm} RPM reached. "
                f"Sleeping {sleep_for:.1f}s...\n"
            )
            time.sleep(sleep_for)
            # Re-prune after sleep
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] >= 60:
                self._timestamps.popleft()

        self._timestamps.append(time.monotonic())


limiter = RpmLimiter(rpm=RPM_LIMIT)


# ── Gemini embedding ───────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_fixed(62),      # on 429, wait a full minute before retrying
    reraise=True,
)
def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Gemini.
    task_type=RETRIEVAL_DOCUMENT for indexing (vs RETRIEVAL_QUERY at query time).
    Retries up to 4x with 62s wait on rate limit errors.
    """
    limiter.wait()   # respect RPM before every call

    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    embeddings = result["embedding"]
    # Normalise: single text returns flat list, batch returns list of lists
    if embeddings and not isinstance(embeddings[0], list):
        embeddings = [embeddings]
    return embeddings


def embed_all_chunks(chunks: list[dict]) -> list[tuple[dict, list[float]]]:
    """
    Embed all chunks in batches, respecting the 100 RPM free tier limit.

    Returns:
        list of (chunk, vector) tuples
    """
    texts = [c["text"] for c in chunks]
    n_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    # Estimated time with rate limiting
    est_minutes = n_batches / RPM_LIMIT
    console.print(
        f"[dim]{len(chunks)} chunks → {n_batches} batches "
        f"(~{est_minutes:.1f} min at {RPM_LIMIT} RPM)[/dim]\n"
    )

    results: list[list[float] | None] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding...", total=n_batches)

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]
            batch_num   = i // EMBED_BATCH_SIZE + 1
            progress.update(
                task,
                description=f"[dim]Batch {batch_num}/{n_batches}[/dim]",
            )

            try:
                vectors = embed_batch(batch_texts)
                results.extend(vectors)
            except Exception as e:
                console.print(f"  [red]FAIL[/red] batch {batch_num} — {e}")
                results.extend([None] * len(batch_texts))

            progress.advance(task)

    # Pair and drop any failed embeddings
    paired = [
        (chunk, vec)
        for chunk, vec in zip(chunks, results)
        if vec is not None
    ]

    dropped = len(chunks) - len(paired)
    if dropped:
        console.print(
            f"[yellow]Warning:[/yellow] {dropped} chunks dropped due to embed failures"
        )

    return paired


# ── Pinecone ───────────────────────────────────────────────────

def get_or_create_index(pc: Pinecone):
    """Get existing Pinecone index or create a new serverless one."""
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        console.print(f"[dim]Creating index '{PINECONE_INDEX_NAME}'...[/dim]")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        console.print(f"[green]Index '{PINECONE_INDEX_NAME}' ready.[/green]")
    else:
        console.print(f"[dim]Using existing index '{PINECONE_INDEX_NAME}'.[/dim]")

    return pc.Index(PINECONE_INDEX_NAME)


def upsert_to_pinecone(index, paired: list[tuple[dict, list[float]]]) -> int:
    """Upsert all (chunk, vector) pairs into Pinecone in batches."""
    total_upserted = 0

    console.print(f"\n[dim]Upserting {len(paired)} vectors...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        console=console,
    ) as progress:
        n_batches = (len(paired) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
        task = progress.add_task("Upserting...", total=n_batches)

        for i in range(0, len(paired), UPSERT_BATCH_SIZE):
            batch     = paired[i : i + UPSERT_BATCH_SIZE]
            batch_num = i // UPSERT_BATCH_SIZE + 1
            progress.update(task, description=f"[dim]Batch {batch_num}/{n_batches}[/dim]")

            vectors = [
                {
                    "id":     chunk["chunk_id"],
                    "values": vec,
                    "metadata": {
                        "text":       chunk["text"],
                        "url":        chunk["url"],
                        "title":      chunk["title"],
                        "has_code":   chunk["has_code"],
                        "char_count": chunk["char_count"],
                    },
                }
                for chunk, vec in batch
            ]

            try:
                index.upsert(vectors=vectors)
                total_upserted += len(vectors)
            except Exception as e:
                console.print(f"  [red]FAIL[/red] upsert batch {batch_num} — {e}")

            progress.advance(task)

    return total_upserted


# ── Main pipeline ──────────────────────────────────────────────

def embed_and_store(chunks: list[dict] | None = None) -> int:
    """Load chunks → embed → upsert to Pinecone."""
    if chunks is None:
        chunks = load_chunks()
    if not chunks:
        console.print("[red]No chunks to embed.[/red]")
        return 0

    console.print(f"\n[bold]Embedding {len(chunks)} chunks[/bold]")
    console.print(f"[dim]Embed model : {GEMINI_EMBED_MODEL}[/dim]")
    console.print(f"[dim]Vector index: {PINECONE_INDEX_NAME} ({PINECONE_DIMENSION}d)[/dim]")
    console.print(f"[dim]Rate limit  : {RPM_LIMIT} RPM (Gemini free tier)[/dim]\n")

    paired = embed_all_chunks(chunks)
    if not paired:
        console.print("[red]No embeddings produced. Aborting.[/red]")
        return 0

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)
    total = upsert_to_pinecone(index, paired)

    stats      = index.describe_index_stats()
    live_count = stats.total_vector_count
    console.print(f"\n[green]Upserted  [/green] {total} vectors")
    console.print(f"[green]Live count[/green] {live_count} total in Pinecone\n")

    return total


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Embed chunks and upsert to Pinecone")
    ap.add_argument("--dry-run",      action="store_true", help="Embed 3 chunks only, skip Pinecone")
    ap.add_argument("--delete-index", action="store_true", help="Delete index before upserting")
    args = ap.parse_args()

    if args.dry_run:
        console.print("\n[bold yellow]Dry run — 3 test chunks only[/bold yellow]\n")
        chunks = load_chunks()
        if chunks:
            vecs = embed_batch([c["text"] for c in chunks[:3]])
            for chunk, vec in zip(chunks[:3], vecs):
                console.print(
                    f"  [green]OK[/green] {chunk['chunk_id']}"
                    f" — dim={len(vec)}, val[0]={vec[0]:.6f}"
                )
        console.print("\n[dim]Dry run done. Nothing written to Pinecone.[/dim]")

    else:
        if args.delete_index:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            if PINECONE_INDEX_NAME in [i.name for i in pc.list_indexes()]:
                console.print(f"[yellow]Deleting '{PINECONE_INDEX_NAME}'...[/yellow]")
                pc.delete_index(PINECONE_INDEX_NAME)
                time.sleep(5)

        total = embed_and_store()
        console.print(f"[bold green]Done — {total} vectors in Pinecone.[/bold green]")