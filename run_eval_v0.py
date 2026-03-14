"""
run_eval.py — run the 10 golden questions through the pipeline and score results

Scoring per question (0–3 points):
    +1  at least one expected_source_contains URL appears in sources
    +1  at least 3 of the expected_concepts appear in the explanation
    +1  explanation does NOT contain the "not covered" fallback phrase

Usage:
    python run_eval.py                  # run all 10 questions
    python run_eval.py --id q03         # run a single question by id
    python run_eval.py --no-save        # run without saving outputs to disk
    python run_eval.py --threshold 0.5  # override retrieval threshold
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from src.pipeline import explain
from src.config import RETRIEVAL_SCORE_THRESHOLD

console = Console()

GOLDEN_PATH  = Path("eval/golden.json")
OUTPUTS_DIR  = Path("eval/outputs")
NOT_COVERED_PHRASE = "not covered in the provided documentation"


# ── Scoring ────────────────────────────────────────────────────

def score_result(result: dict, golden: dict) -> dict:
    """
    Score a single pipeline result against its golden entry.

    Returns a scoring dict:
    {
        "source_hit":    bool,   # correct source URL was cited
        "concept_hits":  int,    # how many expected concepts found
        "concept_total": int,    # total expected concepts
        "not_fallback":  bool,   # did NOT fall back to "not covered"
        "total":         int,    # 0–3
        "pass":          bool,   # total >= 2
    }
    """
    explanation_lower = result["explanation"].lower()
    sources_text      = " ".join(result["sources"]).lower()

    # +1 source hit — expected source URL fragment appears in cited sources
    expected_src = golden.get("expected_source_contains", "")
    source_hit   = expected_src.lower() in sources_text if expected_src else False

    # +1 concept coverage — at least 3 expected concepts in explanation
    expected_concepts = [c.lower() for c in golden.get("expected_concepts", [])]
    concept_hits = sum(
        1 for concept in expected_concepts
        if concept in explanation_lower
    )
    concept_pass = concept_hits >= min(3, len(expected_concepts))

    # +1 no fallback — LLM didn't say "not covered in the provided documentation"
    not_fallback = NOT_COVERED_PHRASE not in explanation_lower

    total = int(source_hit) + int(concept_pass) + int(not_fallback)

    return {
        "source_hit":    source_hit,
        "concept_hits":  concept_hits,
        "concept_total": len(expected_concepts),
        "not_fallback":  not_fallback,
        "total":         total,
        "pass":          total >= 2,
    }


# ── Run single question ────────────────────────────────────────

def run_question(
    golden: dict,
    threshold: float,
    save: bool,
) -> dict:
    """Run the pipeline on one golden question and return scored output."""
    qid      = golden["id"]
    category = golden["category"]
    question = golden["question"]
    code     = golden.get("code_snippet")

    console.rule(f"[bold]{qid}[/bold] — {category}")
    console.print(f"[dim]Q: {question}[/dim]")
    if code:
        console.print(f"[dim]Code: {code[:80].strip()}...[/dim]" if len(code) > 80 else f"[dim]Code: {code.strip()}[/dim]")
    console.print()

    t0 = time.monotonic()
    result = explain(
        question=question,
        code_snippet=code,
        score_threshold=threshold,
        stream=False,   # suppress token stream in eval mode
        verbose=False,
    )
    elapsed = time.monotonic() - t0

    scoring = score_result(result, golden)

    # Print brief summary
    status = "[green]PASS[/green]" if scoring["pass"] else "[red]FAIL[/red]"
    console.print(f"\n{status}  score={scoring['total']}/3  "
                  f"source={'✓' if scoring['source_hit'] else '✗'}  "
                  f"concepts={scoring['concept_hits']}/{scoring['concept_total']}  "
                  f"no_fallback={'✓' if scoring['not_fallback'] else '✗'}  "
                  f"time={elapsed:.1f}s")

    # Print explanation snippet
    snippet = result["explanation"][:300].replace("\n", " ")
    if len(result["explanation"]) > 300:
        snippet += "..."
    console.print(f"[dim]{snippet}[/dim]")

    if result["sources"]:
        console.print(f"[dim]Sources: {result['sources']}[/dim]")
    console.print()

    output = {
        "id":          qid,
        "category":    category,
        "question":    question,
        "code":        code,
        "explanation": result["explanation"],
        "sources":     result["sources"],
        "chunks_used": result["chunks_used"],
        "scoring":     scoring,
        "tokens": {
            "input":  result["input_tokens"],
            "output": result["output_tokens"],
        },
        "elapsed_sec": round(elapsed, 2),
    }

    # Save individual output
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / f"{qid}.json"
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return output


# ── Summary table ──────────────────────────────────────────────

def print_summary(outputs: list[dict]) -> None:
    """Print a Rich table summarising all results + overall score."""
    table = Table(
        title="Eval summary",
        show_header=True,
        header_style="bold",
        show_lines=False,
    )
    table.add_column("ID",       width=5)
    table.add_column("Category", width=18)
    table.add_column("Score",    justify="center", width=7)
    table.add_column("Src",      justify="center", width=5)
    table.add_column("Concepts", justify="center", width=10)
    table.add_column("No FB",    justify="center", width=7)
    table.add_column("Result",   justify="center", width=7)

    passed = 0
    for o in outputs:
        s    = o["scoring"]
        res  = "[green]PASS[/green]" if s["pass"] else "[red]FAIL[/red]"
        src  = "✓" if s["source_hit"]  else "✗"
        nfb  = "✓" if s["not_fallback"] else "✗"
        cons = f"{s['concept_hits']}/{s['concept_total']}"
        if s["pass"]:
            passed += 1
        table.add_row(
            o["id"],
            o["category"],
            f"{s['total']}/3",
            src,
            cons,
            nfb,
            res,
        )

    total   = len(outputs)
    overall = passed / total * 100 if total else 0
    color   = "green" if overall >= 70 else "yellow" if overall >= 50 else "red"

    table.add_section()
    table.add_row(
        "", "TOTAL",
        f"{sum(o['scoring']['total'] for o in outputs)}/{total * 3}",
        "", "", "",
        f"[bold {color}]{passed}/{total} ({overall:.0f}%)[/bold {color}]",
    )

    console.print(table)

    # Exit criterion check
    console.print()
    if overall >= 70:
        console.print(
            "[bold green]✓ Exit criterion met[/bold green] — "
            f"{passed}/{total} questions passed (≥ 7/10 required)"
        )
    else:
        console.print(
            f"[bold red]✗ Exit criterion not met[/bold red] — "
            f"{passed}/{total} passed, need 7/10."
        )
        fails = [o for o in outputs if not o["scoring"]["pass"]]
        console.print("\n[dim]Failed questions to investigate:[/dim]")
        for f in fails:
            s = f["scoring"]
            hints = []
            if not s["source_hit"]:
                hints.append("wrong source retrieved")
            if s["concept_hits"] < 3:
                hints.append(f"only {s['concept_hits']}/{s['concept_total']} concepts found")
            if not s["not_fallback"]:
                hints.append("LLM fell back to 'not covered'")
            console.print(f"  [red]{f['id']}[/red] {f['category']} — {', '.join(hints)}")
    console.print()


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Run golden eval set through the pipeline")
    ap.add_argument("--id",        type=str,   default=None,  help="Run a single question by id (e.g. q03)")
    ap.add_argument("--no-save",   action="store_true",        help="Don't save outputs to eval/outputs/")
    ap.add_argument("--threshold", type=float, default=RETRIEVAL_SCORE_THRESHOLD,
                    help=f"Retrieval score threshold (default: {RETRIEVAL_SCORE_THRESHOLD})")
    args = ap.parse_args()

    # Load golden set
    if not GOLDEN_PATH.exists():
        console.print(f"[red]Golden file not found: {GOLDEN_PATH}[/red]")
        return

    golden_set: list[dict] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    # Filter to single question if --id passed
    if args.id:
        golden_set = [g for g in golden_set if g["id"] == args.id]
        if not golden_set:
            console.print(f"[red]No question with id '{args.id}' found.[/red]")
            return

    console.print()
    console.rule("[bold]Code Explainer v0 — Eval Run[/bold]")
    console.print(f"  Questions  : {len(golden_set)}")
    console.print(f"  Threshold  : {args.threshold}")
    console.print(f"  Save       : {not args.no_save}")
    console.print(f"  Started    : {datetime.now().strftime('%H:%M:%S')}")
    console.print()

    run_start = time.monotonic()
    outputs: list[dict] = []

    for golden in golden_set:
        output = run_question(
            golden=golden,
            threshold=args.threshold,
            save=not args.no_save,
        )
        outputs.append(output)
        # Small pause between questions to avoid Groq burst limits
        time.sleep(1)

    total_elapsed = time.monotonic() - run_start

    # Save full run summary
    if not args.no_save and outputs:
        summary_path = OUTPUTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "threshold": args.threshold,
                    "results":   outputs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[dim]Full run saved → {summary_path}[/dim]\n")

    console.rule("[bold]Results[/bold]")
    print_summary(outputs)
    console.print(f"[dim]Total eval time: {total_elapsed:.1f}s[/dim]\n")


if __name__ == "__main__":
    main()