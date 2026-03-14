"""
ingest.py — one-off runner to populate the Pinecone index

Wires the full ingestion pipeline in sequence:
    crawl → parse → chunk → embed → upsert

This script is run once (or whenever you want to re-index).
It is NOT part of the query pipeline.

Usage:
    python ingest.py                  # full run
    python ingest.py --dry-run        # stop after chunking, no embed/upsert
    python ingest.py --from-step parse  # skip crawl, start from parse
    python ingest.py --force          # re-download and re-parse even if files exist
    python ingest.py --delete-index   # wipe Pinecone index before upserting
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from src.crawler  import crawl
from src.parser   import parse_all
from src.chunker  import chunk_all
from src.embedder import embed_and_store, get_or_create_index
from pinecone import Pinecone
from src.config   import PINECONE_API_KEY, PINECONE_INDEX_NAME

console = Console()

STEPS = ["crawl", "parse", "chunk", "embed"]


def print_step(n: int, name: str) -> None:
    console.print(Rule(f"[bold]Step {n}/4 — {name}[/bold]"))


def print_summary(timings: dict[str, float], total_chunks: int, total_vectors: int) -> None:
    console.print(Rule("[bold green]Ingestion complete[/bold green]"))
    console.print()
    for step, duration in timings.items():
        console.print(f"  {step:<10} {duration:.1f}s")
    console.print()
    console.print(f"  [bold]Chunks created :[/bold] {total_chunks}")
    console.print(f"  [bold]Vectors in index:[/bold] {total_vectors}")
    total = sum(timings.values())
    console.print(f"  [bold]Total time     :[/bold] {total:.1f}s ({total/60:.1f} min)")
    console.print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest asyncio docs into Pinecone")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawl → parse → chunk but skip embed and Pinecone upsert",
    )
    ap.add_argument(
        "--from-step",
        choices=STEPS,
        default="crawl",
        metavar="STEP",
        help=f"Start pipeline from this step. Choices: {STEPS}",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-parse even if files already exist on disk",
    )
    ap.add_argument(
        "--delete-index",
        action="store_true",
        help="Delete and recreate the Pinecone index before upserting",
    )
    args = ap.parse_args()

    start_idx = STEPS.index(args.from_step)
    timings: dict[str, float] = {}

    console.print()
    console.print(Rule("[bold]Code Explainer v0 — Ingestion Pipeline[/bold]"))
    console.print(f"  Starting from : [cyan]{args.from_step}[/cyan]")
    console.print(f"  Dry run       : [cyan]{args.dry_run}[/cyan]")
    console.print(f"  Force re-run  : [cyan]{args.force}[/cyan]")
    console.print()

    # ── Step 1: Crawl ──────────────────────────────────────────
    if start_idx <= STEPS.index("crawl"):
        print_step(1, "Crawl")
        t0 = time.monotonic()
        try:
            saved = crawl(skip_existing=not args.force)
            timings["crawl"] = time.monotonic() - t0
            console.print(f"[green]✓[/green] Crawled {len(saved)} new pages\n")
        except Exception as e:
            console.print(f"[red]Crawl failed:[/red] {e}")
            sys.exit(1)
    else:
        console.print("[dim]Skipping crawl.[/dim]\n")

    # ── Step 2: Parse ──────────────────────────────────────────
    if start_idx <= STEPS.index("parse"):
        print_step(2, "Parse")
        t0 = time.monotonic()
        try:
            saved = parse_all(skip_existing=not args.force)
            timings["parse"] = time.monotonic() - t0
            console.print(f"[green]✓[/green] Parsed {len(saved)} documents\n")
        except Exception as e:
            console.print(f"[red]Parse failed:[/red] {e}")
            sys.exit(1)
    else:
        console.print("[dim]Skipping parse.[/dim]\n")

    # ── Step 3: Chunk ──────────────────────────────────────────
    if start_idx <= STEPS.index("chunk"):
        print_step(3, "Chunk")
        t0 = time.monotonic()
        try:
            chunks = chunk_all(save_to_disk=True)
            timings["chunk"] = time.monotonic() - t0
            console.print(f"[green]✓[/green] Created {len(chunks)} chunks\n")
        except Exception as e:
            console.print(f"[red]Chunk failed:[/red] {e}")
            sys.exit(1)
    else:
        chunks = None  # embedder will load from disk

    # ── Dry run stops here ─────────────────────────────────────
    if args.dry_run:
        console.print(Rule("[bold yellow]Dry run — stopping before embed[/bold yellow]"))
        console.print(
            "\n[dim]Inspect data/chunks/chunks.json to verify chunk quality[/dim]"
            "\n[dim]before running the full pipeline.[/dim]\n"
        )
        console.print(
            "Re-run without --dry-run to embed and upsert to Pinecone:\n"
            "  [bold]python ingest.py --from-step embed[/bold]\n"
        )
        if timings:
            for step, duration in timings.items():
                console.print(f"  {step:<10} {duration:.1f}s")
        sys.exit(0)

    # ── Step 4: Embed + upsert ─────────────────────────────────
    if start_idx <= STEPS.index("embed"):
        print_step(4, "Embed + upsert to Pinecone")

        if args.delete_index:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = [i.name for i in pc.list_indexes()]
            if PINECONE_INDEX_NAME in existing:
                console.print(f"[yellow]Deleting index '{PINECONE_INDEX_NAME}'...[/yellow]")
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

    # ── Final summary ──────────────────────────────────────────
    total_chunks = len(chunks) if chunks else 0
    print_summary(timings, total_chunks, total_vectors)


if __name__ == "__main__":
    main()