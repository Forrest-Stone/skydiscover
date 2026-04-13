"""Aggregate and plot phase-1 budget traces.

Usage:
  python scripts/plot_budget_curves.py --root outputs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def collect_runs(root: Path) -> List[Dict]:
    runs: List[Dict] = []
    for summary_path in root.rglob("summary.json"):
        run_dir = summary_path.parent
        iter_path = run_dir / "iterations.jsonl"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trace = load_jsonl(iter_path)
        runs.append(
            {
                "run_dir": str(run_dir),
                "summary": summary,
                "trace": trace,
            }
        )
    return runs


def write_runs_csv(runs: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_dir",
        "nominal_budget",
        "total_cost",
        "oob",
        "overshoot",
        "best_score",
        "num_iterations",
        "num_generation_calls",
        "num_retry_calls",
        "num_guide_calls",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for run in runs:
            s = run["summary"]
            writer.writerow(
                {
                    "run_dir": run["run_dir"],
                    "nominal_budget": s.get("nominal_budget"),
                    "total_cost": s.get("total_cost"),
                    "oob": s.get("oob"),
                    "overshoot": s.get("overshoot"),
                    "best_score": s.get("best_score"),
                    "num_iterations": s.get("num_iterations"),
                    "num_generation_calls": s.get("num_generation_calls"),
                    "num_retry_calls": s.get("num_retry_calls"),
                    "num_guide_calls": s.get("num_guide_calls"),
                }
            )


def plot_best_score_vs_cost(runs: List[Dict], out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    has_line = False
    for run in runs:
        trace = run["trace"]
        if not trace:
            continue
        x = [float(row.get("cumulative_cost", 0.0) or 0.0) for row in trace]
        y = [row.get("global_best_after") for row in trace]
        if all(v is None for v in y):
            continue
        has_line = True
        label = Path(run["run_dir"]).name
        plt.plot(x, y, linewidth=1.5, alpha=0.85, label=label)

    plt.xlabel("Cumulative cost (USD)")
    plt.ylabel("Best score")
    plt.title("Best score vs cumulative cost")
    if has_line and len(runs) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", help="Root outputs directory")
    parser.add_argument(
        "--out-dir",
        default="outputs/aggregate_budget",
        help="Where to write csv/plots",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    runs = collect_runs(root)
    write_runs_csv(runs, out_dir / "runs.csv")
    plotted = plot_best_score_vs_cost(runs, out_dir / "best_score_vs_cost.png")

    print(f"Found runs: {len(runs)}")
    print(f"Wrote: {out_dir / 'runs.csv'}")
    if plotted:
        print(f"Wrote: {out_dir / 'best_score_vs_cost.png'}")
    else:
        print("Skipped plotting (matplotlib not available).")


if __name__ == "__main__":
    main()
