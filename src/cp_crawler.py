import json
import re
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import GITHUB_TOKEN

console = Console()

# ── Constants ──────────────────────────────────────────────────
GITHUB_API   = "https://api.github.com"
REPO         = "cp-algorithms/cp-algorithms"
BRANCH       = "main"
SRC_PREFIX   = "src"
RAW_BASE     = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{SRC_PREFIX}"
SITE_BASE    = "https://cp-algorithms.com"
CP_RAW_DIR   = Path("data/raw/cp")
REQUEST_DELAY = 0.5   # seconds between requests

# DSA-relevant topic folders — skip geometry/num_methods for v1
DSA_TOPICS = [
    "data_structures",
    "dynamic_programming",
    "graph",
    "string",
    "algebra",
    "combinatorics",
]

# Files to skip — navigation/meta files, not articles
SKIP_FILES = {"index.md", "navigation.md", "contrib.md", "tags.md", "README.md"}


# ── GitHub API helpers ─────────────────────────────────────────

def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=15),
    reraise=True,
)
def _gh_get(url: str) -> dict | list:
    """GET request to GitHub API with retry."""
    resp = requests.get(url, headers=_headers(), timeout=15)

    # Surface rate limit info
    remaining = resp.headers.get("X-RateLimit-Remaining", "?")
    if remaining != "?" and int(remaining) < 5:
        reset = resp.headers.get("X-RateLimit-Reset", "?")
        console.print(
            f"[yellow]GitHub rate limit:[/yellow] {remaining} requests remaining "
            f"(resets at {reset}). Set GITHUB_TOKEN in .env for 5000/hr."
        )

    resp.raise_for_status()
    return resp.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=15),
    reraise=True,
)
def _fetch_raw(raw_url: str) -> str:
    """Fetch raw file content from GitHub."""
    resp = requests.get(raw_url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.text


# ── Article discovery ──────────────────────────────────────────

def list_articles(topic: str) -> list[dict]:
    """
    List all .md article files in a topic folder via GitHub Trees API.
    Returns list of {name, path, raw_url, site_url} dicts.
    """
    url  = f"{GITHUB_API}/repos/{REPO}/contents/{SRC_PREFIX}/{topic}"
    try:
        items = _gh_get(url)
    except Exception as e:
        console.print(f"[red]Failed to list {topic}:[/red] {e}")
        return []

    articles = []
    for item in items:
        if item.get("type") != "file":
            continue
        name = item["name"]
        if not name.endswith(".md"):
            continue
        if name in SKIP_FILES:
            continue

        slug    = name.replace(".md", "")
        raw_url = f"{RAW_BASE}/{topic}/{name}"
        # CP-Algorithms site URL maps directly from path
        site_url = f"{SITE_BASE}/{topic}/{slug}.html"

        articles.append({
            "name":     name,
            "slug":     slug,
            "topic":    topic,
            "raw_url":  raw_url,
            "site_url": site_url,
        })

    return articles


# ── Markdown cleaning ──────────────────────────────────────────

def clean_markdown(text: str, title: str) -> str:
    """
    Clean raw CP-Algorithms markdown for embedding:
    - Strip YAML frontmatter
    - Remove MkDocs-specific directives (admonition blocks etc.)
    - Collapse excess blank lines
    - Keep code blocks intact
    """
    # Strip YAML frontmatter (--- ... ---)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # Remove MkDocs admonition markers (!!!  note, ??? tip etc.)
    text = re.sub(r"^[!?]{3}\s+\w+.*$", "", text, flags=re.MULTILINE)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove raw HTML tags (tables are ok as text)
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_title(markdown: str, fallback: str) -> str:
    """Extract article title from first # heading, or use filename as fallback."""
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Convert slug to title (e.g. "segment_tree" → "Segment Tree")
    return fallback.replace("_", " ").replace("-", " ").title()


# ── Build document ─────────────────────────────────────────────

def build_document(article: dict, raw_markdown: str) -> dict:
    """Convert raw markdown + metadata into our clean document format."""
    title = extract_title(raw_markdown, article["slug"])
    body  = clean_markdown(raw_markdown, title)

    return {
        "title":       title,
        "topic":       article["topic"],
        "slug":        article["slug"],
        "body":        body,
        "url":         article["site_url"],
        "raw_url":     article["raw_url"],
        "char_count":  len(body),
        "source_type": "cp_algorithms",
        "source_file": f"{article['topic']}/{article['name']}",
    }


# ── Main crawl ─────────────────────────────────────────────────

def crawl_cp(
    topics:        list[str] | None = None,
    output_dir:    Path             = CP_RAW_DIR,
    skip_existing: bool             = True,
) -> list[Path]:
    """
    Fetch all DSA articles from CP-Algorithms and save to disk.

    Args:
        topics:        list of topic folders to crawl. Defaults to DSA_TOPICS.
        output_dir:    where to save JSON files
        skip_existing: skip articles already saved to disk

    Returns:
        list of Paths for every JSON file saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_topics = topics or DSA_TOPICS
    saved:  list[Path] = []
    skipped = 0
    failed: list[str]  = []

    if not GITHUB_TOKEN:
        console.print(
            "[yellow]Tip:[/yellow] Set GITHUB_TOKEN in .env for 5,000 req/hr "
            "(vs 60/hr unauthenticated). "
            "Generate at github.com/settings/tokens — no scopes needed for public repos."
        )

    console.print(f"\n[bold]Crawling CP-Algorithms ({len(target_topics)} topics)[/bold]")

    # Discover all articles first
    all_articles: list[dict] = []
    for topic in target_topics:
        articles = list_articles(topic)
        console.print(f"  [dim]{topic:<22} {len(articles)} articles[/dim]")
        all_articles.extend(articles)
        time.sleep(REQUEST_DELAY)

    console.print(f"\n[dim]Total: {len(all_articles)} articles to fetch[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching...", total=len(all_articles))

        for article in all_articles:
            fname    = f"{article['topic']}__{article['slug']}.json"
            out_path = output_dir / fname
            progress.update(
                task,
                description=f"[dim]{article['topic']}/{article['slug'][:30]}[/dim]",
            )

            if skip_existing and out_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            try:
                raw_md = _fetch_raw(article["raw_url"])
                doc    = build_document(article, raw_md)

                # Skip articles with too little content (stubs)
                if doc["char_count"] < 200:
                    progress.advance(task)
                    continue

                out_path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                saved.append(out_path)

            except Exception as e:
                console.print(
                    f"  [red]FAIL[/red] {article['topic']}/{article['slug']} — {e}"
                )
                failed.append(article["slug"])

            time.sleep(REQUEST_DELAY)
            progress.advance(task)

    # ── Summary ────────────────────────────────────────────────
    total_chars = sum(
        json.loads(p.read_text(encoding="utf-8"))["char_count"]
        for p in saved
    )
    console.print(f"\n[green]Saved[/green]    {len(saved)} articles")
    if skipped:
        console.print(f"[dim]Skipped  {skipped} already on disk[/dim]")
    if failed:
        console.print(f"[red]Failed   {len(failed)}:[/red] {failed}")
    console.print(f"[dim]Total body text: {total_chars:,} chars[/dim]\n")

    return saved


def load_cp_docs(cp_dir: Path = CP_RAW_DIR) -> list[dict]:
    """
    Load all saved CP-Algorithms JSON files from disk.
    Used by chunker.py as an additional input source.
    """
    docs = []
    for p in sorted(cp_dir.glob("*.json")):
        try:
            docs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            console.print(f"[red]Failed to load {p.name}:[/red] {e}")
    return docs


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Crawl CP-Algorithms DSA articles")
    ap.add_argument(
        "--topics",
        nargs="+",
        default=None,
        metavar="TOPIC",
        help=f"Topics to crawl. Choices: {DSA_TOPICS}",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if JSON already exists",
    )
    ap.add_argument(
        "--inspect",
        type=int,
        default=None,
        metavar="N",
        help="Print N saved docs after crawling",
    )
    args = ap.parse_args()

    # Validate topics
    if args.topics:
        invalid = [t for t in args.topics if t not in DSA_TOPICS]
        if invalid:
            console.print(f"[red]Unknown topics:[/red] {invalid}")
            console.print(f"[dim]Valid topics: {DSA_TOPICS}[/dim]")
            raise SystemExit(1)

    saved = crawl_cp(
        topics=args.topics,
        skip_existing=not args.force,
    )

    if args.inspect and saved:
        import random
        console.print(f"\n[bold]Inspecting {args.inspect} random articles[/bold]\n")
        for path in random.sample(saved, min(args.inspect, len(saved))):
            doc = json.loads(path.read_text(encoding="utf-8"))
            console.print(f"[bold cyan]{doc['title']}[/bold cyan]")
            console.print(f"[dim]topic={doc['topic']} chars={doc['char_count']:,}[/dim]")
            console.print(f"[dim]url={doc['url']}[/dim]")
            preview = doc["body"][:300].replace("\n", " ")
            console.print(f"[dim]{preview}...[/dim]")
            console.rule()

    console.print(f"\n[bold green]Done — {len(saved)} articles saved.[/bold green]")