from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from skydiscover.budget.core import BudgetLedger, IterationBudgetRecord
from skydiscover.budget.hooks import aggregate_tokens


def write_iteration_record(path: Path, record: IterationBudgetRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = record.calls[-1].model_name if record.calls else "unknown"
    prompt_tokens, completion_tokens = aggregate_tokens(record)
    row: Dict[str, Any] = {
        "iteration": record.iteration,
        "frontier_id": record.meta.get("frontier_id"),
        "generation_cost": record.generation_cost,
        "retry_cost": record.retry_cost,
        "guide_cost": record.guide_cost,
        "iteration_cost": record.iteration_cost,
        "cumulative_cost": record.cumulative_cost,
        "remaining_budget_ratio": record.remaining_budget_ratio,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_name": model_name,
        "candidate_score": record.meta.get("candidate_score"),
        "global_best_before": record.meta.get("global_best_before"),
        "global_best_after": record.meta.get("global_best_after"),
        "meta_triggered": record.meta.get("meta_triggered", False),
        "attempts_used": record.meta.get("attempts_used", 1),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, ledger: BudgetLedger, best_score: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = ledger.summary()
    if best_score is not None:
        summary["best_score"] = float(best_score)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def plot_run_best_score_vs_cost(iterations_path: Path, out_png: Path) -> bool:
    """Plot best-score-vs-cost for a single run.

    Returns False when plotting dependencies are unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not iterations_path.exists():
        return False

    costs = []
    best_scores = []
    with iterations_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            costs.append(float(row.get("cumulative_cost", 0.0) or 0.0))
            best_scores.append(row.get("global_best_after"))

    if not costs:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(costs, best_scores, linewidth=1.8)
    plt.xlabel("Cumulative cost (USD)")
    plt.ylabel("Best score")
    plt.title("Best score vs cumulative cost")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True
