"""
parser.py — clean raw HTML → structured JSON documents

Reads HTML files from data/raw/, extracts clean body text using
trafilatura, and saves one JSON file per page to data/parsed/.

Each output JSON has this shape:
{
    "url":      "https://docs.python.org/3/library/asyncio-task.html",
    "title":    "Coroutines and Tasks",
    "body":     "clean plain text...",
    "sections": ["Creating Tasks", "Awaitables", ...],
    "has_code": true,
    "char_count": 4821
}

Run directly:
    python -m src.parser
"""

import json
import re
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ── Constants ──────────────────────────────────────────────────
RAW_DIR    = Path("data/raw")
PARSED_DIR = Path("data/parsed")

# Base URL to reconstruct full URLs from filenames
BASE_URL = "https://docs.python.org/3/library/"


# ── Metadata extraction ────────────────────────────────────────

def extract_title(html: str) -> str:
    """
    Extract the page title from <h1> first, fall back to <title> tag.
    asyncio doc pages have clean h1s like "Coroutines and Tasks".
    """
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1:
        # Strip the ¶ permalink symbol sphinx adds
        return h1.get_text(strip=True).replace("¶", "").strip()

    title_tag = soup.find("title")
    if title_tag:
        # "Coroutines and Tasks — Python 3.x documentation" → "Coroutines and Tasks"
        return title_tag.get_text().split("—")[0].strip()

    return "Untitled"


def extract_sections(html: str) -> list[str]:
    """
    Extract h2/h3 heading text as a list of section names.
    Useful as metadata to help the retriever understand page structure.
    """
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True).replace("¶", "").strip()
        if text:
            sections.append(text)
    return sections


def extract_body(html: str) -> str | None:
    """
    Use trafilatura to extract clean body text.
    include_tables=True preserves argument tables common in Python docs.
    include_formatting=False gives plain text — better for embedding.
    """
    text = trafilatura.extract(
        html,
        include_tables=True,
        include_formatting=False,
        include_comments=False,
        no_fallback=False,
    )
    return text


def clean_body(text: str) -> str:
    """
    Post-process trafilatura output:
    - Collapse 3+ blank lines into 2
    - Strip leading/trailing whitespace
    - Remove sphinx navigation artifacts that slip through
    """
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common sphinx nav artifacts
    nav_patterns = [
        r"^(previous|next|index|modules)\s*\|.*$",
        r"^navigation$",
        r"^Table of Contents$",
    ]
    for pattern in nav_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text.strip()


def has_code_blocks(html: str) -> bool:
    """Check if the page contains any <code> or <pre> blocks."""
    soup = BeautifulSoup(html, "html.parser")
    return bool(soup.find(["code", "pre"]))


def filename_to_url(filename: str) -> str:
    """Convert a saved HTML filename back to its source URL."""
    return BASE_URL + filename


# ── Parse single file ──────────────────────────────────────────

def parse_file(html_path: Path) -> dict | None:
    """
    Parse a single raw HTML file into a structured document dict.
    Returns None if trafilatura can't extract meaningful content.
    """
    html = html_path.read_text(encoding="utf-8")

    body = extract_body(html)
    if not body or len(body.strip()) < 100:
        # Skip pages with no extractable content (e.g. redirect stubs)
        return None

    body = clean_body(body)
    title = extract_title(html)
    sections = extract_sections(html)
    url = filename_to_url(html_path.name)

    return {
        "url":        url,
        "title":      title,
        "body":       body,
        "sections":   sections,
        "has_code":   has_code_blocks(html),
        "char_count": len(body),
        "source_file": html_path.name,
    }


# ── Main parse ─────────────────────────────────────────────────

def parse_all(
    input_dir: Path  = RAW_DIR,
    output_dir: Path = PARSED_DIR,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Parse all HTML files in input_dir and write JSON to output_dir.

    Args:
        input_dir:     directory containing raw .html files.
        output_dir:    directory to write parsed .json files.
        skip_existing: skip files already parsed.

    Returns:
        list of Path objects for every JSON file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html_files = sorted(input_dir.glob("*.html"))

    if not html_files:
        console.print(f"[red]No HTML files found in {input_dir}[/red]")
        console.print("[dim]Run crawler.py first.[/dim]")
        return []

    saved: list[Path] = []
    skipped = 0
    failed: list[str] = []
    total_chars = 0

    console.print(f"\n[bold]Parsing {len(html_files)} HTML files[/bold]")
    console.print(f"Output → [dim]{output_dir.resolve()}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing...", total=len(html_files))

        for html_path in html_files:
            out_path = output_dir / html_path.with_suffix(".json").name
            progress.update(task, description=f"[dim]{html_path.name}[/dim]")

            if skip_existing and out_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            try:
                doc = parse_file(html_path)

                if doc is None:
                    console.print(
                        f"  [yellow]SKIP[/yellow] {html_path.name}"
                        " — no extractable content"
                    )
                    progress.advance(task)
                    continue

                out_path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                saved.append(out_path)
                total_chars += doc["char_count"]

            except Exception as e:
                console.print(f"  [red]FAIL[/red] {html_path.name} — {e}")
                failed.append(html_path.name)

            progress.advance(task)

    # ── Summary ────────────────────────────────────────────────
    console.print(f"\n[green]Parsed[/green]   {len(saved)} documents")
    if skipped:
        console.print(f"[dim]Skipped[/dim]  {skipped} already parsed")
    if failed:
        console.print(f"[red]Failed[/red]   {len(failed)}: {failed}")
    console.print(f"[dim]Total body text: {total_chars:,} chars[/dim]\n")

    return saved


def load_parsed_docs(parsed_dir: Path = PARSED_DIR) -> list[dict]:
    """
    Load all parsed JSON documents from disk.
    Used by chunker.py as its input.
    """
    docs = []
    for json_path in sorted(parsed_dir.glob("*.json")):
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
            docs.append(doc)
        except Exception as e:
            console.print(f"[red]Failed to load {json_path.name}:[/red] {e}")
    return docs


# ── CLI entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse raw asyncio HTML docs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse files even if JSON already exists",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Print parsed output for a single file, e.g. --inspect asyncio-task.html",
    )
    args = parser.parse_args() # to add arguments from command line

    if args.inspect:
        # Quick inspection mode — print one file's parsed output
        path = RAW_DIR / args.inspect
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
        else:
            doc = parse_file(path)
            if doc:
                console.print(f"\n[bold]Title:[/bold] {doc['title']}")
                console.print(f"[bold]URL:[/bold] {doc['url']}")
                console.print(f"[bold]Sections:[/bold] {doc['sections']}")
                console.print(f"[bold]Has code:[/bold] {doc['has_code']}")
                console.print(f"[bold]Chars:[/bold] {doc['char_count']:,}")
                console.print(f"\n[bold]Body preview:[/bold]\n{doc['body'][:500]}...")
    else:
        saved = parse_all(skip_existing=not args.force)
        console.print(f"[bold green]Done — {len(saved)} documents saved.[/bold green]")