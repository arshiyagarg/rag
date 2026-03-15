"""
so_crawler.py — fetch asyncio questions + answers from Stack Overflow

Uses the Stack Exchange API v2.3 (free, no auth needed for read).
Fetches top-voted asyncio questions and their accepted/top answers,
saves one JSON file per question to data/raw/so/.

Each saved JSON has this shape:
{
    "question_id":    12345,
    "title":          "How does asyncio.gather handle exceptions?",
    "question_body":  "clean text...",
    "url":            "https://stackoverflow.com/questions/12345",
    "score":          247,
    "view_count":     18400,
    "tags":           ["python", "asyncio", "python-asyncio"],
    "answers": [
        {
            "answer_id":   67890,
            "body":        "clean text...",
            "score":       189,
            "is_accepted": true,
            "url":         "https://stackoverflow.com/a/67890"
        },
        ...
    ]
}

Rate limits (Stack Exchange API):
    - Unauthenticated : 300 req/day
    - Authenticated   : 10,000 req/day (set STACKAPPS_KEY in .env)

Run directly:
    python -m src.so_crawler
    python -m src.so_crawler --max-questions 50
    python -m src.so_crawler --force
"""

import json
import time
import html
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import STACKAPPS_KEY

console = Console()

# ── Constants ──────────────────────────────────────────────────
SO_API_BASE  = "https://api.stackexchange.com/2.3"
SO_RAW_DIR   = Path("data/raw/so")

# Tags to search — covers the full asyncio surface area
SEARCH_TAGS  = ["dsa c++","data structures and algorithms c++"]
EXTRA_TAGS   = ["C++","java"]          # secondary pass for older questions

# Only fetch questions with meaningful engagement
MIN_SCORE        = 5
MIN_ANSWER_SCORE = 3
MAX_QUESTIONS    = 100              # default cap — override with --max-questions
PAGE_SIZE        = 100              # max allowed by SE API

REQUEST_DELAY    = 0.2              # seconds between requests (be polite)


# ── HTML → plain text ──────────────────────────────────────────

def html_to_text(raw_html: str) -> str:
    """
    Convert Stack Overflow HTML body to clean plain text.
    Preserves code blocks with >>> markers for chunker.py to detect.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # Replace <code> blocks with backtick-fenced versions
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")

    # Replace <pre> blocks (multi-line code) with indented text
    for pre in soup.find_all("pre"):
        lines = pre.get_text().splitlines()
        indented = "\n".join(f"    {line}" for line in lines)
        pre.replace_with(f"\n{indented}\n")

    # Get plain text
    text = soup.get_text(separator="\n")

    # Unescape HTML entities (&amp; → &, etc.)
    text = html.unescape(text)

    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── API calls ──────────────────────────────────────────────────

def _build_params(extra: dict | None = None) -> dict:
    """Base params for every SE API call."""
    params = {
        "site":     "stackoverflow",
        "filter":   "withbody",       # include body field
        "order":    "desc",
        "sort":     "votes",
    }
    if STACKAPPS_KEY:
        params["key"] = STACKAPPS_KEY
    if extra:
        params.update(extra)
    return params


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=30),
    reraise=True,
)
def _get(endpoint: str, params: dict) -> dict:
    """Make a GET request to the SE API with retry logic."""
    url = f"{SO_API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # SE API rate limit warning
    quota = data.get("quota_remaining", "?")
    if isinstance(quota, int) and quota < 20:
        console.print(
            f"[yellow]Warning:[/yellow] only {quota} API requests remaining today."
        )

    return data


def fetch_questions(
    tag:           str,
    max_questions: int,
    min_score:     int,
) -> list[dict]:
    """
    Fetch top-voted questions for a given tag.
    Handles pagination automatically.
    """
    questions: list[dict] = []
    page = 1

    while len(questions) < max_questions:
        params = _build_params({
            "tagged":   tag,
            "pagesize": min(PAGE_SIZE, max_questions - len(questions)),
            "page":     page,
            "min":      min_score,
        })

        try:
            data = _get("questions", params)
        except Exception as e:
            console.print(f"[red]Failed to fetch questions page {page}:[/red] {e}")
            break

        items = data.get("items", [])
        if not items:
            break

        questions.extend(items)
        time.sleep(REQUEST_DELAY)

        if not data.get("has_more", False):
            break
        page += 1

    return questions[:max_questions]


def fetch_answers(question_id: int, min_score: int) -> list[dict]:
    """Fetch all answers for a question, filtered by minimum score."""
    params = _build_params({
        "pagesize": 10,
        "min":      min_score,
    })

    try:
        data = _get(f"questions/{question_id}/answers", params)
        return data.get("items", [])
    except Exception as e:
        console.print(f"[red]Failed to fetch answers for {question_id}:[/red] {e}")
        return []


# ── Build document ─────────────────────────────────────────────

def build_document(question: dict, answers: list[dict]) -> dict:
    """
    Convert raw SE API response into our clean document format.
    Sorts answers: accepted first, then by score descending.
    """
    q_id    = question["question_id"]
    q_url   = question.get("link", f"https://stackoverflow.com/questions/{q_id}")

    # Clean question body
    q_body_raw = question.get("body", "")
    q_body     = html_to_text(q_body_raw) if q_body_raw else ""

    # Sort answers — accepted answer always first
    sorted_answers = sorted(
        answers,
        key=lambda a: (a.get("is_accepted", False), a.get("score", 0)),
        reverse=True,
    )

    clean_answers = []
    for ans in sorted_answers:
        a_id   = ans["answer_id"]
        a_body = html_to_text(ans.get("body", ""))
        if not a_body:
            continue
        clean_answers.append({
            "answer_id":   a_id,
            "body":        a_body,
            "score":       ans.get("score", 0),
            "is_accepted": ans.get("is_accepted", False),
            "url":         f"https://stackoverflow.com/a/{a_id}",
        })

    return {
        "question_id":   q_id,
        "title":         html.unescape(question.get("title", "")),
        "question_body": q_body,
        "url":           q_url,
        "score":         question.get("score", 0),
        "view_count":    question.get("view_count", 0),
        "tags":          question.get("tags", []),
        "answer_count":  len(clean_answers),
        "answers":       clean_answers,
        "source_type":   "stackoverflow",
    }


# ── Main crawl ─────────────────────────────────────────────────

def crawl_so(
    max_questions:   int   = MAX_QUESTIONS,
    min_score:       int   = MIN_SCORE,
    min_answer_score: int  = MIN_ANSWER_SCORE,
    output_dir:      Path  = SO_RAW_DIR,
    skip_existing:   bool  = True,
) -> list[Path]:
    """
    Crawl Stack Overflow asyncio questions and save to disk.

    Args:
        max_questions:    total questions to fetch across all tags
        min_score:        minimum question score to include
        min_answer_score: minimum answer score to include
        output_dir:       where to save JSON files
        skip_existing:    skip questions already saved to disk

    Returns:
        list of Paths for every JSON file saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved:   list[Path] = []
    skipped = 0
    all_questions: list[dict] = []

    # Fetch from both tag variants to maximise coverage
    for tag in [*SEARCH_TAGS, *EXTRA_TAGS]:
        per_tag = max_questions // len([*SEARCH_TAGS, *EXTRA_TAGS])
        console.print(f"\n[dim]Fetching questions tagged [{tag}]...[/dim]")
        questions = fetch_questions(tag, per_tag, min_score)
        # Deduplicate by question_id
        existing_ids = {q["question_id"] for q in all_questions}
        new = [q for q in questions if q["question_id"] not in existing_ids]
        all_questions.extend(new)
        console.print(f"[dim]Found {len(new)} new questions (tag={tag})[/dim]")

    console.print(f"\n[bold]Fetching answers for {len(all_questions)} questions[/bold]")
    if not STACKAPPS_KEY:
        console.print(
            "[yellow]Tip:[/yellow] Set STACKAPPS_KEY in .env to increase "
            "rate limit from 300 to 10,000 req/day."
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Crawling...", total=len(all_questions))

        for question in all_questions:
            q_id     = question["question_id"]
            out_path = output_dir / f"{q_id}.json"

            progress.update(
                task,
                description=f"[dim]{str(question.get('title',''))[:45]}[/dim]",
            )

            if skip_existing and out_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            # Skip questions with no answers
            if question.get("answer_count", 0) == 0:
                progress.advance(task)
                continue

            answers = fetch_answers(q_id, min_answer_score)
            time.sleep(REQUEST_DELAY)

            doc = build_document(question, answers)

            # Skip if no clean answers survived filtering
            if not doc["answers"]:
                progress.advance(task)
                continue

            out_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved.append(out_path)
            progress.advance(task)

    # ── Summary ────────────────────────────────────────────────
    total_answers = sum(
        json.loads(p.read_text(encoding="utf-8"))["answer_count"]
        for p in saved
    )
    console.print(f"\n[green]Saved[/green]    {len(saved)} questions")
    console.print(f"[dim]Answers  {total_answers} total answer bodies[/dim]")
    if skipped:
        console.print(f"[dim]Skipped  {skipped} already on disk[/dim]")

    return saved


def load_so_docs(so_dir: Path = SO_RAW_DIR) -> list[dict]:
    """
    Load all saved SO JSON files from disk.
    Used by chunker.py as an additional input source.
    """
    docs = []
    for p in sorted(so_dir.glob("*.json")):
        try:
            docs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            console.print(f"[red]Failed to load {p.name}:[/red] {e}")
    return docs


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Crawl Stack Overflow asyncio Q&A")
    ap.add_argument("--max-questions",    type=int,  default=MAX_QUESTIONS)
    ap.add_argument("--min-score",        type=int,  default=MIN_SCORE)
    ap.add_argument("--min-answer-score", type=int,  default=MIN_ANSWER_SCORE)
    ap.add_argument("--force",            action="store_true",
                    help="Re-fetch even if JSON already exists")
    ap.add_argument("--inspect",          type=int,  default=None, metavar="N",
                    help="Print N saved docs after crawling")
    args = ap.parse_args()

    saved = crawl_so(
        max_questions=args.max_questions,
        min_score=args.min_score,
        min_answer_score=args.min_answer_score,
        skip_existing=not args.force,
    )

    if args.inspect and saved:
        import random
        console.print(f"\n[bold]Inspecting {args.inspect} saved docs[/bold]\n")
        for path in random.sample(saved, min(args.inspect, len(saved))):
            doc = json.loads(path.read_text(encoding="utf-8"))
            console.print(f"[bold cyan]{doc['title']}[/bold cyan]")
            console.print(f"[dim]score={doc['score']} answers={doc['answer_count']} url={doc['url']}[/dim]")
            if doc["answers"]:
                preview = doc["answers"][0]["body"][:200].replace("\n", " ")
                console.print(f"[dim]Top answer: {preview}...[/dim]")
            console.rule()

    console.print(f"\n[bold green]Done — {len(saved)} questions saved.[/bold green]")