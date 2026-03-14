import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import CHUNK_TOKEN_LIMIT  # just confirms config loads

console = Console()

# ── Constants ──────────────────────────────────────────────────
BASE_URL = "https://docs.python.org/3/library/"
RAW_DIR = Path("data/raw")

# All asyncio pages to crawl — explicit list is more reliable than
# recursive spidering on a docs site with lots of cross-links
ASYNCIO_PAGES = [
    "asyncio.html",
    "asyncio-task.html",
    "asyncio-eventloop.html",
    "asyncio-future.html",
    "asyncio-stream.html",
    "asyncio-sync.html",
    "asyncio-queue.html",
    "asyncio-subprocess.html",
    "asyncio-protocol.html",
    "asyncio-exceptions.html",
    "asyncio-dev.html",
    "asyncio-api-index.html",
    "asyncio-extending.html",
    "asyncio-runner.html",
]

HEADERS = {
    "User-Agent": "code-explainer-v0/1.0 (educational project; respectful crawler)",
}

REQUEST_DELAY = 1.0  # seconds between requests — be polite to python.org


# ── Core fetch ─────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_page(url: str) -> str:
    """Fetch a single URL and return raw HTML. Retries up to 3 times."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.text


def url_to_filename(url: str) -> str:
    """
    Convert a URL to a safe local filename.
    e.g. https://docs.python.org/3/library/asyncio-task.html
         → asyncio-task.html
    """
    path = urlparse(url).path
    name = Path(path).name
    if not name.endswith(".html"):
        # fallback for unusual URLs — use a hash
        name = hashlib.md5(url.encode()).hexdigest()[:12] + ".html"
    return name


# ── Link discovery ─────────────────────────────────────────────

def discover_links(html: str, base_url: str) -> list[str]:
    """
    Parse a page and return any asyncio sub-page links not already
    in our explicit list. Catches pages we might have missed.
    """
    soup = BeautifulSoup(html, "html.parser")
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only follow relative links to asyncio pages
        if href.startswith("asyncio") and href.endswith(".html"):
            full_url = urljoin(base_url, href)
            # Strip anchors
            full_url = full_url.split("#")[0]
            if full_url not in found:
                found.append(full_url)
    return found


# ── Main crawl ─────────────────────────────────────────────────

def crawl(
    pages: list[str] | None = None,
    output_dir: Path = RAW_DIR,
    delay: float = REQUEST_DELAY,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Crawl asyncio doc pages and save raw HTML files.

    Args:
        pages:        list of page filenames to crawl. Defaults to ASYNCIO_PAGES.
        output_dir:   directory to save HTML files into.
        delay:        seconds to sleep between requests.
        skip_existing: if True, skip pages already saved to disk.

    Returns:
        list of Path objects for every saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_pages = pages or ASYNCIO_PAGES
    urls = [urljoin(BASE_URL, p) for p in target_pages]

    saved: list[Path] = []
    skipped = 0
    failed: list[str] = []

    console.print(f"\n[bold]Crawling {len(urls)} asyncio doc pages[/bold]")
    console.print(f"Output → [dim]{output_dir.resolve()}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching...", total=len(urls))

        for url in urls:
            filename = url_to_filename(url)
            out_path = output_dir / filename
            progress.update(task, description=f"[dim]{filename}[/dim]")

            # Skip if already on disk
            if skip_existing and out_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            try:
                html = fetch_page(url)

                # Discover any extra asyncio links on this page
                extra = discover_links(html, BASE_URL)
                for extra_url in extra:
                    if extra_url not in urls:
                        urls.append(extra_url)
                        progress.update(task, total=len(urls))

                out_path.write_text(html, encoding="utf-8")
                saved.append(out_path)

            except Exception as e:
                console.print(f"  [red]FAIL[/red] {filename} — {e}")
                failed.append(url)

            time.sleep(delay)
            progress.advance(task)

    # ── Summary ────────────────────────────────────────────────
    total_bytes = sum(p.stat().st_size for p in output_dir.glob("*.html"))
    console.print(f"\n[green]Saved[/green]    {len(saved)} pages")
    if skipped:
        console.print(f"[dim]Skipped[/dim]  {skipped} already on disk")
    if failed:
        console.print(f"[red]Failed[/red]   {len(failed)} pages: {failed}")
    console.print(f"[dim]Total size: {total_bytes / 1024:.1f} KB[/dim]\n")

    return saved


# ── CLI entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl asyncio docs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download pages even if already saved",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds between requests (default: {REQUEST_DELAY})",
    )
    args = parser.parse_args()

    saved = crawl(skip_existing=not args.force, delay=args.delay)
    console.print(f"[bold green]Done — {len(saved)} new files saved.[/bold green]")