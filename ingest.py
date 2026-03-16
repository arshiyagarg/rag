"""
ingest.py — populate Pinecone with DSA Hint Engine data

Sources:
    stackoverflow   — top-voted DSA Q&A via Stack Exchange API
    cp_algorithms   — DSA articles from CP-Algorithms via GitHub

Usage:
    python ingest.py                           # ingest all sources
    python ingest.py --source stackoverflow    # single source
    python ingest.py --source cp_algorithms    # single source
    python ingest.py --dry-run                 # crawl + chunk only, no embed
    python ingest.py --from-step chunk         # skip crawl, start from chunk
    python ingest.py --from-step embed         # skip crawl + chunk
    python ingest.py --force                   # re-crawl even if files exist
    python ingest.py --delete-index            # wipe Pinecone index first
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from src.so_crawler import crawl_so
from src.cp_crawler import crawl_cp
from src.chunker    import chunk_all, load_chunks
from src.embedder   import embed_and_store
from src.config     import PINECONE_API_KEY, PINECONE_INDEX_NAME

console = Console()

STEPS   = ["crawl", "chunk", "embed"]
SOURCES = ["stackoverflow", "cp_algorithms"]


def print_summary(
    timings:       dict[str, float],
    total_chunks:  int,
    total_vectors: int,
) -> None:
    console.print(Rule("[bold green]Ingestion complete[/bold green]"))
    console.print()
    for step, dur in timings.items():
        console.print(f"  {step:<14} {dur:.1f}s")
    console.print()
    console.print(f"  [bold]Chunks created :[/bold] {total_chunks}")
    console.print(f"  [bold]Vectors in index:[/bold] {total_vectors}")
    total = sum(timings.values())
    console.print(f"  [bold]Total time     :[/bold] {total:.1f}s ({total/60:.1f} min)")
    console.print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest DSA data into Pinecone")
    ap.add_argument(
        "--source",
        choices=SOURCES + ["all"],
        default="all",
        help="Which source to ingest (default: all)",
    )
    ap.add_argument(
        "--from-step",
        choices=STEPS,
        default="crawl",
        metavar="STEP",
        help=f"Start pipeline from this step. Choices: {STEPS}",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawl + chunk but skip embed and Pinecone upsert",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-crawl even if files already exist on disk",
    )
    ap.add_argument(
        "--delete-index",
        action="store_true",
        help="Delete and recreate the Pinecone index before upserting",
    )
    args = ap.parse_args()

    sources   = SOURCES if args.source == "all" else [args.source]
    start_idx = STEPS.index(args.from_step)
    timings:  dict[str, float] = {}
    chunks:   list[dict] | None = None

    console.print()
    console.print(Rule("[bold]DSA Hint Engine — Ingestion Pipeline[/bold]"))
    console.print(f"  Sources     : [cyan]{sources}[/cyan]")
    console.print(f"  From step   : [cyan]{args.from_step}[/cyan]")
    console.print(f"  Dry run     : [cyan]{args.dry_run}[/cyan]")
    console.print(f"  Force       : [cyan]{args.force}[/cyan]")
    console.print(f"  Started     : {datetime.now().strftime('%H:%M:%S')}")
    console.print()

    # ── Step 1: Crawl ──────────────────────────────────────────
    if start_idx <= STEPS.index("crawl"):
        console.print(Rule("[bold]Step 1/3 — Crawl[/bold]"))
        t0 = time.monotonic()

        if "cp_algorithms" in sources:
            try:
                saved = crawl_cp(skip_existing=not args.force)
                console.print(
                    f"[green]✓[/green] CP-Algorithms: {len(saved)} articles\n"
                )
            except Exception as e:
                console.print(f"[red]CP-Algorithms crawl failed:[/red] {e}")
                sys.exit(1)

        if "stackoverflow" in sources:
            try:
                saved = crawl_so(skip_existing=not args.force)
                console.print(
                    f"[green]✓[/green] Stack Overflow: {len(saved)} questions\n"
                )
            except Exception as e:
                console.print(f"[red]Stack Overflow crawl failed:[/red] {e}")
                sys.exit(1)

        timings["crawl"] = time.monotonic() - t0
    else:
        console.print("[dim]Skipping crawl.[/dim]\n")

    # ── Step 2: Chunk ──────────────────────────────────────────
    if start_idx <= STEPS.index("chunk"):
        console.print(Rule("[bold]Step 2/3 — Chunk[/bold]"))
        t0 = time.monotonic()
        try:
            chunks = chunk_all(sources=sources, save_to_disk=True)
            timings["chunk"] = time.monotonic() - t0
            console.print(f"[green]✓[/green] {len(chunks)} chunks created\n")
        except Exception as e:
            console.print(f"[red]Chunking failed:[/red] {e}")
            sys.exit(1)
    else:
        # Load from disk if skipping chunk step
        console.print("[dim]Skipping chunk — loading from disk.[/dim]\n")
        chunks = load_chunks()

    # ── Dry run stops here ─────────────────────────────────────
    if args.dry_run:
        console.print(Rule("[bold yellow]Dry run — stopping before embed[/bold yellow]"))
        console.print(
            "\n[dim]Inspect data/chunks/ to verify chunk quality:[/dim]\n"
            "  [bold]python -m src.chunker --inspect 5[/bold]\n"
        )
        console.print(
            "[dim]Re-run without --dry-run to embed and upsert:[/dim]\n"
            f"  [bold]python ingest.py --source {args.source} --from-step embed[/bold]\n"
        )
        if timings:
            console.print()
            for step, dur in timings.items():
                console.print(f"  {step:<14} {dur:.1f}s")
        console.print()
        sys.exit(0)

    # ── Step 3: Embed + upsert ─────────────────────────────────
    if start_idx <= STEPS.index("embed"):
        console.print(Rule("[bold]Step 3/3 — Embed + Upsert[/bold]"))

        if args.delete_index:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = [i.name for i in pc.list_indexes()]
            if PINECONE_INDEX_NAME in existing:
                console.print(
                    f"[yellow]Deleting index '{PINECONE_INDEX_NAME}'...[/yellow]"
                )
                pc.delete_index(PINECONE_INDEX_NAME)
                time.sleep(5)
                console.print("[green]Index deleted.[/green]\n")

        t0 = time.monotonic()
        try:
            total_vectors = embed_and_store(chunks)
            timings["embed"] = time.monotonic() - t0
            console.print(f"[green]✓[/green] {total_vectors} vectors upserted\n")
        except Exception as e:
            console.print(f"[red]Embed failed:[/red] {e}")
            sys.exit(1)
    else:
        total_vectors = 0

    print_summary(timings, len(chunks) if chunks else 0, total_vectors)


if __name__ == "__main__":
    main()