"""
chunker.py — split parsed documents into embedding-ready chunks

Reads parsed JSON docs from data/parsed/ via parser.load_parsed_docs(),
splits each document body into semantic chunks using chonkie,
and returns a flat list of chunk dicts ready for embedder.py.

Each output chunk has this shape:
{
    "chunk_id":   "asyncio-task_004",
    "text":       "...",
    "url":        "https://docs.python.org/3/library/asyncio-task.html",
    "title":      "Coroutines and Tasks",
    "sections":   ["Creating Tasks", "Awaitables"],
    "has_code":   true,
    "char_count": 312,
    "chunk_index": 4,
    "total_chunks": 18
}

Run directly:
    python -m src.chunker
"""

import json
import re
from pathlib import Path

from chonkie import SemanticChunker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from src.config import CHUNK_TOKEN_LIMIT, CHUNK_OVERLAP
from src.parser import load_parsed_docs

console = Console()

# ── Constants ──────────────────────────────────────────────────
CHUNKS_DIR = Path("data/chunks")

# Code block pattern — we protect these before chunking
# so SemanticChunker never splits inside a code example
CODE_BLOCK_PATTERN = re.compile(r"(>>>.*(?:\n>>>.*|\n\.\.\..+|\n(?!>>>|\n).+)*)", re.MULTILINE)


# ── Code block protection ──────────────────────────────────────

def protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace code blocks with placeholder tokens before chunking.
    This prevents SemanticChunker from splitting mid-example.

    Returns:
        (protected_text, placeholder_map) where placeholder_map lets
        us restore original code after chunking.
    """
    placeholder_map: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        code = match.group(0)
        token = f"__CODE_BLOCK_{len(placeholder_map):04d}__"
        placeholder_map[token] = code
        return token

    protected = CODE_BLOCK_PATTERN.sub(replace, text)
    return protected, placeholder_map


def restore_code_blocks(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore original code blocks from placeholder tokens."""
    for token, code in placeholder_map.items():
        text = text.replace(token, code)
    return text


# ── Chunk single document ──────────────────────────────────────

def chunk_document(
    doc: dict,
    chunker: SemanticChunker,
) -> list[dict]:
    """
    Split a single parsed document into semantic chunks.

    Args:
        doc:     parsed document dict from parser.py
        chunker: initialised SemanticChunker instance (shared across docs)

    Returns:
        list of chunk dicts with full metadata attached
    """
    body = doc["body"]

    # Protect code blocks from being split
    protected_body, placeholder_map = protect_code_blocks(body)

    # Run semantic chunking
    raw_chunks = chunker.chunk(protected_body)

    if not raw_chunks:
        return []

    # Build output chunk dicts
    chunks: list[dict] = []
    source_file = doc.get("source_file", "unknown")
    base_id = source_file.replace(".html", "").replace(".json", "")

    for i, raw in enumerate(raw_chunks):
        # Restore code blocks in this chunk
        chunk_text = restore_code_blocks(raw.text, placeholder_map)
        chunk_text = chunk_text.strip()

        if not chunk_text:
            continue

        chunks.append({
            "chunk_id":     f"{base_id}_{i:03d}",
            "text":         chunk_text,
            "url":          doc["url"],
            "title":        doc["title"],
            "sections":     doc.get("sections", []),
            "has_code":     doc.get("has_code", False),
            "char_count":   len(chunk_text),
            "chunk_index":  i,
            "total_chunks": len(raw_chunks),
        })

    return chunks


# ── Main chunking pipeline ─────────────────────────────────────

def chunk_all(
    parsed_dir: Path | None = None,
    output_dir: Path = CHUNKS_DIR,
    save_to_disk: bool = True,
) -> list[dict]:
    """
    Load all parsed docs, chunk them, and optionally save to disk.

    Args:
        parsed_dir:   override default parsed docs directory
        output_dir:   where to save chunk JSON files
        save_to_disk: if True, write chunks JSON to output_dir

    Returns:
        flat list of all chunk dicts across all documents
    """
    # Load parsed docs
    docs = load_parsed_docs(parsed_dir) if parsed_dir else load_parsed_docs()

    if not docs:
        console.print("[red]No parsed documents found.[/red]")
        console.print("[dim]Run parser.py first.[/dim]")
        return []

    console.print(f"\n[bold]Chunking {len(docs)} documents[/bold]")
    console.print(
        f"[dim]token_limit={CHUNK_TOKEN_LIMIT}, "
        f"overlap={CHUNK_OVERLAP}[/dim]\n"
    )

    # Initialise chunker once — loading the embedding model is expensive
    console.print("[dim]Loading SemanticChunker model...[/dim]")
    chunker = SemanticChunker(
        tokenizer_or_token_counter="gpt2",   # tiktoken-compatible, no API needed
        chunk_size=CHUNK_TOKEN_LIMIT,
        threshold=0.5,                        # semantic similarity split threshold
    )
    console.print("[dim]Model ready.[/dim]\n")

    all_chunks: list[dict] = []
    per_doc_stats: list[tuple[str, int, int]] = []  # (title, chunk_count, char_count)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking...", total=len(docs))

        for doc in docs:
            progress.update(task, description=f"[dim]{doc['title'][:40]}[/dim]")

            try:
                chunks = chunk_document(doc, chunker)
                all_chunks.extend(chunks)
                per_doc_stats.append((
                    doc["title"],
                    len(chunks),
                    sum(c["char_count"] for c in chunks),
                ))
            except Exception as e:
                console.print(f"  [red]FAIL[/red] {doc['title']} — {e}")

            progress.advance(task)

    # ── Save to disk ───────────────────────────────────────────
    if save_to_disk and all_chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "chunks.json"
        out_path.write_text(
            json.dumps(all_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[dim]Chunks saved → {out_path.resolve()}[/dim]")

    # ── Stats table ────────────────────────────────────────────
    table = Table(title="Chunking summary", show_header=True, header_style="bold")
    table.add_column("Document", style="dim", max_width=40)
    table.add_column("Chunks", justify="right")
    table.add_column("Chars", justify="right")

    for title, n_chunks, n_chars in per_doc_stats:
        table.add_row(title[:40], str(n_chunks), f"{n_chars:,}")

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(all_chunks)}[/bold]",
        f"[bold]{sum(c['char_count'] for c in all_chunks):,}[/bold]",
    )
    console.print(table)

    return all_chunks


def load_chunks(chunks_dir: Path = CHUNKS_DIR) -> list[dict]:
    """
    Load saved chunks from disk.
    Used by embedder.py as its input.
    """
    chunks_path = chunks_dir / "chunks.json"
    if not chunks_path.exists():
        console.print(f"[red]chunks.json not found at {chunks_path}[/red]")
        console.print("[dim]Run chunker.py first.[/dim]")
        return []

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    console.print(f"[dim]Loaded {len(chunks)} chunks from {chunks_path}[/dim]")
    return chunks


# ── CLI entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk parsed asyncio docs")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run chunking but don't write chunks.json to disk",
    )
    parser.add_argument(
        "--inspect",
        type=int,
        default=None,
        metavar="N",
        help="Print N random chunks after chunking for manual inspection",
    )
    args = parser.parse_args()

    chunks = chunk_all(save_to_disk=not args.no_save)

    if args.inspect and chunks:
        import random
        console.print(f"\n[bold]Inspecting {args.inspect} random chunks[/bold]\n")
        sample = random.sample(chunks, min(args.inspect, len(chunks)))
        for chunk in sample:
            console.print(f"[bold cyan]{chunk['chunk_id']}[/bold cyan]")
            console.print(f"[dim]title:[/dim] {chunk['title']}")
            console.print(f"[dim]url:[/dim]   {chunk['url']}")
            console.print(f"[dim]chars:[/dim] {chunk['char_count']}")
            console.print(f"\n{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}\n")
            console.rule()

    console.print(f"\n[bold green]Done — {len(chunks)} total chunks.[/bold green]")