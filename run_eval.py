"""
run_eval.py — run 10 DSA golden questions through the hint pipeline

Scoring per question (0–4 points):
    +1  correct pattern identified (sliding window, BFS, DP, etc.)
    +1  at least 3 expected concepts appear in the hint
    +1  correct source type cited (SO or CP-Algorithms URL)
    +1  hint does NOT reveal the full solution (no complete code given)

Usage:
    python run_eval.py
    python run_eval.py --id q03
    python run_eval.py --no-save
    python run_eval.py --threshold 0.4
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from src.pipeline import hint
from src.config   import RETRIEVAL_SCORE_THRESHOLD

console    = Console()
GOLDEN_PATH = Path("eval/golden.json")
OUTPUTS_DIR = Path("eval/outputs")

NOT_COVERED_PHRASE = "not covered in the provided references"
# Heuristic: full solution given if response has 5+ lines of code
FULL_SOLUTION_RE   = re.compile(r"```python.*?```", re.DOTALL)


# ── Scoring ────────────────────────────────────────────────────

def score_result(result: dict, golden: dict) -> dict:
    hint_lower    = result["hint"].lower()
    sources_text  = " ".join(result["sources"]).lower()
    pattern_lower = result["pattern"].lower()

    # +1 correct pattern
    expected_pattern  = golden["expected_pattern"].lower()
    pattern_hit = (
        expected_pattern in pattern_lower
        or expected_pattern in hint_lower
    )

    # +1 concept coverage (at least 3 of expected_concepts)
    expected_concepts = [c.lower() for c in golden.get("expected_concepts", [])]
    concept_hits = sum(1 for c in expected_concepts if c in hint_lower)
    concept_pass = concept_hits >= min(3, len(expected_concepts))

    # +1 source hit
    expected_src = golden.get("expected_source_contains", "")
    source_hit   = expected_src.lower() in sources_text if expected_src else False

    # +1 no full solution revealed
    code_blocks  = FULL_SOLUTION_RE.findall(result["hint"])
    no_full_soln = len(code_blocks) == 0 or all(
        len(b.splitlines()) < 6 for b in code_blocks
    )

    total = int(pattern_hit) + int(concept_pass) + int(source_hit) + int(no_full_soln)

    return {
        "pattern_hit":   pattern_hit,
        "concept_hits":  concept_hits,
        "concept_total": len(expected_concepts),
        "concept_pass":  concept_pass,
        "source_hit":    source_hit,
        "no_full_soln":  no_full_soln,
        "total":         total,
        "pass":          total >= 3,   # need 3/4 to pass
    }


# ── Run single question ────────────────────────────────────────

def run_question(golden: dict, threshold: float, save: bool) -> dict:
    qid      = golden["id"]
    category = golden["category"]
    problem  = golden["problem"]
    code     = golden.get("code_snippet")

    console.rule(f"[bold]{qid}[/bold] — {category}")
    console.print(f"[dim]{problem[:80]}[/dim]\n")

    t0     = time.monotonic()
    result = hint(
        problem=problem,
        code_snippet=code,
        score_threshold=threshold,
        stream=False,
        verbose=False,
    )
    elapsed = time.monotonic() - t0
    scoring = score_result(result, golden)

    status = "[green]PASS[/green]" if scoring["pass"] else "[red]FAIL[/red]"
    console.print(
        f"{status}  score={scoring['total']}/4  "
        f"pattern={'✓' if scoring['pattern_hit'] else '✗'} ({result['pattern']})  "
        f"concepts={scoring['concept_hits']}/{scoring['concept_total']}  "
        f"source={'✓' if scoring['source_hit'] else '✗'}  "
        f"no_soln={'✓' if scoring['no_full_soln'] else '✗'}  "
        f"time={elapsed:.1f}s"
    )

    snippet = result["hint"][:250].replace("\n", " ")
    console.print(f"[dim]{snippet}...[/dim]\n")

    output = {
        "id":        qid,
        "category":  category,
        "problem":   problem,
        "hint":      result["hint"],
        "pattern":   result["pattern"],
        "sources":   result["sources"],
        "scoring":   scoring,
        "tokens":    {"input": result["input_tokens"], "output": result["output_tokens"]},
        "elapsed_sec": round(elapsed, 2),
    }

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / f"{qid}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return output


# ── Summary ────────────────────────────────────────────────────

def print_summary(outputs: list[dict]) -> None:
    table = Table(title="Eval summary", show_header=True, header_style="bold")
    table.add_column("ID",       width=5)
    table.add_column("Category", width=20)
    table.add_column("Score",    justify="center", width=7)
    table.add_column("Pattern",  justify="center", width=8)
    table.add_column("Concepts", justify="center", width=10)
    table.add_column("Source",   justify="center", width=8)
    table.add_column("No soln",  justify="center", width=9)
    table.add_column("Result",   justify="center", width=7)

    passed = 0
    for o in outputs:
        s   = o["scoring"]
        res = "[green]PASS[/green]" if s["pass"] else "[red]FAIL[/red]"
        if s["pass"]:
            passed += 1
        table.add_row(
            o["id"], o["category"],
            f"{s['total']}/4",
            "✓" if s["pattern_hit"]  else "✗",
            f"{s['concept_hits']}/{s['concept_total']}",
            "✓" if s["source_hit"]   else "✗",
            "✓" if s["no_full_soln"] else "✗",
            res,
        )

    total   = len(outputs)
    pct     = passed / total * 100 if total else 0
    color   = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"
    table.add_section()
    table.add_row(
        "", "TOTAL", "", "", "", "", "",
        f"[bold {color}]{passed}/{total} ({pct:.0f}%)[/bold {color}]",
    )
    console.print(table)
    console.print()

    if pct >= 70:
        console.print(
            f"[bold green]✓ Exit criterion met[/bold green] — "
            f"{passed}/{total} questions passed (≥ 7/10 required)"
        )
    else:
        console.print(
            f"[bold red]✗ Exit criterion not met[/bold red] — "
            f"{passed}/{total} passed, need 7/10"
        )
        fails = [o for o in outputs if not o["scoring"]["pass"]]
        for f in fails:
            s = f["scoring"]
            hints_list = []
            if not s["pattern_hit"]:   hints_list.append("wrong pattern")
            if not s["concept_pass"]:  hints_list.append(f"concepts {s['concept_hits']}/{s['concept_total']}")
            if not s["source_hit"]:    hints_list.append("source not cited")
            if not s["no_full_soln"]:  hints_list.append("revealed full solution")
            console.print(f"  [red]{f['id']}[/red] {f['category']} — {', '.join(hints_list)}")
    console.print()


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id",        type=str,   default=None)
    ap.add_argument("--no-save",   action="store_true")
    ap.add_argument("--threshold", type=float, default=RETRIEVAL_SCORE_THRESHOLD)
    args = ap.parse_args()

    golden_set = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.id:
        golden_set = [g for g in golden_set if g["id"] == args.id]
        if not golden_set:
            console.print(f"[red]No question with id '{args.id}'[/red]")
            return

    console.print()
    console.print(Rule("[bold]DSA Hint Engine — Eval Run[/bold]"))
    console.print(f"  Questions : {len(golden_set)}")
    console.print(f"  Threshold : {args.threshold}")
    console.print(f"  Started   : {datetime.now().strftime('%H:%M:%S')}")
    console.print()

    t_start = time.monotonic()
    outputs: list[dict] = []

    for golden in golden_set:
        output = run_question(golden, args.threshold, not args.no_save)
        outputs.append(output)
        time.sleep(1)

    total_elapsed = time.monotonic() - t_start

    if not args.no_save and outputs:
        run_path = OUTPUTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        run_path.write_text(
            json.dumps({"timestamp": datetime.now().isoformat(), "results": outputs},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"[dim]Run saved → {run_path}[/dim]\n")

    console.rule("[bold]Results[/bold]")
    print_summary(outputs)
    console.print(f"[dim]Total time: {total_elapsed:.1f}s[/dim]\n")


if __name__ == "__main__":
    main()