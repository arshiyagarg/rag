"""
chunker.py — split documents from all sources into embedding-ready chunks

Handles two source types:
    - stackoverflow (DSA Q&A via so_crawler.py)
    - cp_algorithms (DSA articles via cp_crawler.py)

Each output chunk has this shape:
{
    "chunk_id":    "so_12345_ans_0",
    "text":        "...",
    "url":         "https://stackoverflow.com/a/67890",
    "title":       "How to find longest substring without repeating characters",
    "source_type": "stackoverflow",
    "topic":       "graph",               # cp_algorithms only
    "so_score":    189,                   # stackoverflow only
    "is_accepted": true,                  # stackoverflow only
    "has_code":    true,
    "char_count":  312,
    "chunk_index": 0,
    "total_chunks": 3
}

Run directly:
    python -m src.chunker                        # chunk all sources
    python -m src.chunker --source stackoverflow # single source
    python -m src.chunker --source cp_algorithms # single source
    python -m src.chunker --inspect 5            # inspect random chunks
"""

import json
import re
from pathlib import Path

from chonkie import SemanticChunker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from src.config import CHUNK_TOKEN_LIMIT, CHUNK_OVERLAP

from src.so_crawler import load_so_docs
from src.cp_crawler import load_cp_docs

console = Console()

# ── Constants ──────────────────────────────────────────────────
CHUNKS_DIR = Path("data/chunks")

# Protect >>> style Python code blocks (docs source)
PYTHON_CODE_RE = re.compile(
    r"(>>>.*(?:\n>>>.*|\n\.\.\..+|\n(?!>>>|\n).+)*)",
    re.MULTILINE,
)
# Protect fenced markdown code blocks (cp_algorithms source)
FENCED_CODE_RE = re.compile(
    r"(```[\w]*\n.*?```)",
    re.DOTALL,
)
# Protect indented code blocks — 4-space indent (cp_algorithms source)
INDENTED_CODE_RE = re.compile(
    r"((?:^    .+\n?)+)",
    re.MULTILINE,
)


# ── Code block protection (shared) ────────────────────────────

def protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace all code block patterns with placeholder tokens.
    Handles Python >>>, fenced ``` blocks, and 4-space indented blocks.
    Prevents SemanticChunker from splitting mid-example.
    """
    placeholder_map: dict[str, str] = {}
    counter = [0]  # mutable for use inside nested replace()

    def replace(match: re.Match) -> str:
        code  = match.group(0)
        token = f"__CODE_{counter[0]:04d}__"
        placeholder_map[token] = code
        counter[0] += 1
        return token

    text = FENCED_CODE_RE.sub(replace, text)
    text = PYTHON_CODE_RE.sub(replace, text)
    text = INDENTED_CODE_RE.sub(replace, text)
    return text, placeholder_map


def restore_code_blocks(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore original code blocks from placeholder tokens."""
    for token, code in placeholder_map.items():
        text = text.replace(token, code)
    return text


def _make_chunker() -> SemanticChunker:
    """Initialise a shared SemanticChunker instance."""
    console.print("[dim]Loading SemanticChunker model...[/dim]")
    chunker = SemanticChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=CHUNK_TOKEN_LIMIT,
        threshold=0.5,
    )
    console.print("[dim]Model ready.[/dim]\n")
    return chunker


def _semantic_split(
    text:    str,
    chunker: SemanticChunker,
) -> list[str]:
    """
    Run SemanticChunker on text (with code protection) and return
    a list of clean chunk strings.
    """
    protected, placeholder_map = protect_code_blocks(text)
    raw_chunks = chunker.chunk(protected)
    result = []
    for raw in raw_chunks:
        restored = restore_code_blocks(raw.text, placeholder_map).strip()
        if restored:
            result.append(restored)
    return result


# ── Source-specific chunkers ───────────────────────────────────

def chunk_docs_document(doc: dict, chunker: SemanticChunker) -> list[dict]:
    """
    Chunk a single docs page.
    Same logic as v0 but now adds source_type field.
    """
    texts   = _semantic_split(doc["body"], chunker)
    base_id = doc.get("source_file", "doc").replace(".html", "").replace(".json", "")

    return [
        {
            "chunk_id":    f"{base_id}_{i:03d}",
            "text":        text,
            "url":         doc["url"],
            "title":       doc["title"],
            "source_type": "docs",
            "topic":       None,
            "so_score":    None,
            "is_accepted": None,
            "has_code":    doc.get("has_code", False),
            "char_count":  len(text),
            "chunk_index": i,
            "total_chunks": len(texts),
        }
        for i, text in enumerate(texts)
    ]


def chunk_so_document(doc: dict, chunker: SemanticChunker) -> list[dict]:
    """
    Chunk a Stack Overflow Q&A document (from so_crawler.py).

    Strategy:
    - Question body → one semantic chunk (usually short enough)
    - Each answer body → chunked independently
    - Accepted answer always gets chunk_id prefix "accepted"
    - SO vote score stored per chunk so reranker/prompt can use it
    """
    chunks: list[dict] = []
    q_id  = doc["question_id"]
    title = doc["title"]

    # ── Question body ──────────────────────────────────────────
    if doc.get("question_body", "").strip():
        q_texts = _semantic_split(doc["question_body"], chunker)
        for i, text in enumerate(q_texts):
            chunks.append({
                "chunk_id":    f"so_{q_id}_q_{i:02d}",
                "text":        text,
                "url":         doc["url"],
                "title":       title,
                "source_type": "stackoverflow",
                "topic":       None,
                "so_score":    doc.get("score", 0),
                "is_accepted": None,
                "has_code":    "```" in text or ">>>" in text or "    " in text,
                "char_count":  len(text),
                "chunk_index": i,
                "total_chunks": len(q_texts),
            })

    # ── Answer bodies ──────────────────────────────────────────
    for ans in doc.get("answers", []):
        a_id      = ans["answer_id"]
        is_acc    = ans.get("is_accepted", False)
        a_score   = ans.get("score", 0)
        a_url     = ans.get("url", doc["url"])
        prefix    = "accepted" if is_acc else f"ans{a_id}"

        a_texts = _semantic_split(ans["body"], chunker)
        for i, text in enumerate(a_texts):
            chunks.append({
                "chunk_id":    f"so_{q_id}_{prefix}_{i:02d}",
                "text":        text,
                "url":         a_url,
                "title":       title,
                "source_type": "stackoverflow",
                "topic":       None,
                "so_score":    a_score,
                "is_accepted": is_acc,
                "has_code":    "```" in text or ">>>" in text or "    " in text,
                "char_count":  len(text),
                "chunk_index": i,
                "total_chunks": len(a_texts),
            })

    return chunks


def chunk_cp_document(doc: dict, chunker: SemanticChunker) -> list[dict]:
    """
    Chunk a CP-Algorithms article (from cp_crawler.py).

    Strategy:
    - Split on ## headings first to keep algorithm sections intact
    - Then run SemanticChunker within each section
    - Preserves fenced code blocks and indented code examples
    """
    chunks: list[dict] = []
    topic  = doc.get("topic", "")
    title  = doc["title"]
    url    = doc["url"]
    slug   = doc.get("slug", "cp")
    body   = doc["body"]

    # Split into sections on ## headings
    sections = re.split(r"\n(?=## )", body)
    section_chunks: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        # If section is short enough, keep as one chunk
        if len(section) <= CHUNK_TOKEN_LIMIT * 4:
            section_chunks.append(section)
        else:
            # Split longer sections semantically
            section_chunks.extend(_semantic_split(section, chunker))

    for i, text in enumerate(section_chunks):
        if not text.strip():
            continue
        chunks.append({
            "chunk_id":    f"cp_{topic}_{slug}_{i:03d}",
            "text":        text,
            "url":         url,
            "title":       title,
            "source_type": "cp_algorithms",
            "topic":       topic,
            "so_score":    None,
            "is_accepted": None,
            "has_code":    "```" in text or "    " in text,
            "char_count":  len(text),
            "chunk_index": i,
            "total_chunks": len(section_chunks),
        })

    return chunks


# ── Unified chunk_all ──────────────────────────────────────────

def chunk_all(
    sources:      list[str] | None = None,
    output_dir:   Path             = CHUNKS_DIR,
    save_to_disk: bool             = True,
) -> list[dict]:
    """
    Load all sources, chunk them, and optionally save to disk.

    Args:
        sources:      which sources to chunk. None = all.
                      Options: ["docs", "stackoverflow", "cp_algorithms"]
        output_dir:   where to write chunks.json
        save_to_disk: write chunks.json if True

    Returns:
        flat list of all chunk dicts across all sources
    """
    target = sources or ["stackoverflow", "cp_algorithms"]
    chunker = _make_chunker()
    all_chunks:    list[dict]               = []
    source_stats:  list[tuple[str, int, int]] = []  # (source, n_chunks, n_chars)

    console.print(f"[bold]Chunking sources: {target}[/bold]\n")

    # ── Docs ───────────────────────────────────────────────────
    if "docs" in target:
        if docs:
            doc_chunks = _chunk_source(
                docs, chunk_docs_document, chunker,
                label="docs", id_key="title",
            )
            all_chunks.extend(doc_chunks)
            source_stats.append((
                "docs",
                len(doc_chunks),
                sum(c["char_count"] for c in doc_chunks),
            ))

    # ── Stack Overflow ─────────────────────────────────────────
    if "stackoverflow" in target:
        so_docs = load_so_docs()
        if so_docs:
            so_chunks = _chunk_source(
                so_docs, chunk_so_document, chunker,
                label="stackoverflow", id_key="title",
            )
            all_chunks.extend(so_chunks)
            source_stats.append((
                "stackoverflow",
                len(so_chunks),
                sum(c["char_count"] for c in so_chunks),
            ))

    # ── CP-Algorithms ──────────────────────────────────────────
    if "cp_algorithms" in target:
        cp_docs = load_cp_docs()
        if cp_docs:
            cp_chunks = _chunk_source(
                cp_docs, chunk_cp_document, chunker,
                label="cp_algorithms", id_key="title",
            )
            all_chunks.extend(cp_chunks)
            source_stats.append((
                "cp_algorithms",
                len(cp_chunks),
                sum(c["char_count"] for c in cp_chunks),
            ))

    # ── Save ───────────────────────────────────────────────────
    if save_to_disk and all_chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save full combined file
        out_path = output_dir / "chunks.json"
        out_path.write_text(
            json.dumps(all_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Also save per-source files for incremental ingest
        by_source: dict[str, list[dict]] = {}
        for chunk in all_chunks:
            by_source.setdefault(chunk["source_type"], []).append(chunk)
        for src, chunks in by_source.items():
            src_path = output_dir / f"chunks_{src}.json"
            src_path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        console.print(f"\n[dim]Chunks saved → {output_dir.resolve()}[/dim]")

    # ── Stats table ────────────────────────────────────────────
    table = Table(title="Chunking summary", show_header=True, header_style="bold")
    table.add_column("Source",  style="dim", width=18)
    table.add_column("Chunks",  justify="right")
    table.add_column("Chars",   justify="right")

    for src, n_chunks, n_chars in source_stats:
        table.add_row(src, str(n_chunks), f"{n_chars:,}")

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(all_chunks)}[/bold]",
        f"[bold]{sum(c['char_count'] for c in all_chunks):,}[/bold]",
    )
    console.print(table)
    return all_chunks


def _chunk_source(
    docs:     list[dict],
    fn,
    chunker:  SemanticChunker,
    label:    str,
    id_key:   str,
) -> list[dict]:
    """Helper — iterate docs through a chunking function with progress bar."""
    chunks: list[dict] = []
    console.print(f"[dim]Chunking {len(docs)} {label} documents...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"{label}...", total=len(docs))
        for doc in docs:
            progress.update(
                task,
                description=f"[dim]{str(doc.get(id_key, ''))[:40]}[/dim]",
            )
            try:
                chunks.extend(fn(doc, chunker))
            except Exception as e:
                console.print(
                    f"  [red]FAIL[/red] {doc.get(id_key, '?')[:40]} — {e}"
                )
            progress.advance(task)

    console.print(f"  [green]✓[/green] {label}: {len(chunks)} chunks\n")
    return chunks


def load_chunks(
    chunks_dir:  Path         = CHUNKS_DIR,
    source_type: str | None   = None,
) -> list[dict]:
    """
    Load saved chunks from disk.

    Args:
        source_type: if set, load only chunks for that source
                     (reads chunks_{source_type}.json)

    Used by embedder.py as its input.
    """
    if source_type:
        path = chunks_dir / f"chunks_{source_type}.json"
    else:
        path = chunks_dir / "chunks.json"

    if not path.exists():
        console.print(f"[red]{path.name} not found at {chunks_dir}[/red]")
        console.print("[dim]Run chunker.py first.[/dim]")
        return []

    chunks = json.loads(path.read_text(encoding="utf-8"))
    console.print(f"[dim]Loaded {len(chunks)} chunks from {path.name}[/dim]")
    return chunks


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, random

    ap = argparse.ArgumentParser(description="Chunk all DSA hint engine sources")
    ap.add_argument(
        "--source",
        choices=["stackoverflow", "cp_algorithms"],
        default=None,
        help="Chunk a single source only (default: all)",
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Run chunking but don't write to disk",
    )
    ap.add_argument(
        "--inspect",
        type=int,
        default=None,
        metavar="N",
        help="Print N random chunks after chunking",
    )
    args = ap.parse_args()

    sources = [args.source] if args.source else None
    chunks  = chunk_all(sources=sources, save_to_disk=not args.no_save)

    if args.inspect and chunks:
        console.print(f"\n[bold]Inspecting {args.inspect} random chunks[/bold]\n")
        for chunk in random.sample(chunks, min(args.inspect, len(chunks))):
            src   = chunk["source_type"]
            color = {"docs": "blue", "stackoverflow": "green", "cp_algorithms": "magenta"}.get(src, "white")
            console.print(f"[bold {color}]{chunk['chunk_id']}[/bold {color}]  [{src}]")
            console.print(f"[dim]title    :[/dim] {chunk['title']}")
            console.print(f"[dim]url      :[/dim] {chunk['url']}")
            if chunk.get("so_score") is not None:
                console.print(f"[dim]so_score :[/dim] {chunk['so_score']}  accepted={chunk['is_accepted']}")
            if chunk.get("topic"):
                console.print(f"[dim]topic    :[/dim] {chunk['topic']}")
            console.print(f"[dim]chars    :[/dim] {chunk['char_count']}")
            console.print(f"\n{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}\n")
            console.rule()

    console.print(f"\n[bold green]Done — {len(chunks)} total chunks.[/bold green]")