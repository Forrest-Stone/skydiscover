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
    gen_prompt = sum(c.prompt_tokens for c in record.calls if c.role.value == "generation")
    gen_completion = sum(c.completion_tokens for c in record.calls if c.role.value == "generation")
    retry_prompt = sum(c.prompt_tokens for c in record.calls if c.role.value == "retry")
    retry_completion = sum(c.completion_tokens for c in record.calls if c.role.value == "retry")
    guide_prompt = sum(c.prompt_tokens for c in record.calls if c.role.value == "guide")
    guide_completion = sum(c.completion_tokens for c in record.calls if c.role.value == "guide")
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
        # Explicit aliases so downstream consumers can read in/out directly.
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "model_name": model_name,
        "candidate_score": record.meta.get("candidate_score"),
        "global_best_before": record.meta.get("global_best_before"),
        "global_best_after": record.meta.get("global_best_after"),
        "tier": record.meta.get("tier"),
        "recent_improvement_avg": record.meta.get("recent_improvement_avg"),
        "stagnation_steps": record.meta.get("stagnation_steps"),
        "local_gain": record.meta.get("local_gain"),
        "global_gain": record.meta.get("global_gain"),
        "utility": record.meta.get("utility"),
        "frontier_signal": record.meta.get("frontier_signal"),
        "budget_bin": record.meta.get("budget_bin"),
        "action_family": record.meta.get("action_family"),
        "action_tier": record.meta.get("action_tier"),
        "frontier_gain": record.meta.get("frontier_gain"),
        "routing_reward": (
            record.meta.get("routing_reward")
            if record.meta.get("routing_reward") is not None
            else record.meta.get("router_reward")
        ),
        "router_reward": record.meta.get("router_reward"),
        "meta_triggered": record.meta.get("meta_triggered", False),
        "attempts_used": record.meta.get("attempts_used", 1),
        "num_calls": len(record.calls),
        "num_generation_calls_this_iteration": sum(
            1 for c in record.calls if c.role.value == "generation"
        ),
        "num_retry_calls_this_iteration": sum(1 for c in record.calls if c.role.value == "retry"),
        "num_guide_calls_this_iteration": sum(1 for c in record.calls if c.role.value == "guide"),
        "total_tokens": prompt_tokens + completion_tokens,
        "generation_prompt_tokens": gen_prompt,
        "generation_completion_tokens": gen_completion,
        "generation_input_tokens": gen_prompt,
        "generation_output_tokens": gen_completion,
        "retry_prompt_tokens": retry_prompt,
        "retry_completion_tokens": retry_completion,
        "retry_input_tokens": retry_prompt,
        "retry_output_tokens": retry_completion,
        "guide_prompt_tokens": guide_prompt,
        "guide_completion_tokens": guide_completion,
        "guide_input_tokens": guide_prompt,
        "guide_output_tokens": guide_completion,
        "objective_key": record.meta.get("objective_key"),
        "objective_value": record.meta.get("objective_value"),
        "best_so_far_objective": record.meta.get("best_so_far_objective"),
        "target_value": record.meta.get("target_value"),
        "target_ratio": record.meta.get("target_ratio"),
        "best_so_far_target_ratio": record.meta.get("best_so_far_target_ratio"),
        "combined_score": record.meta.get("combined_score"),
        "best_so_far_combined_score": record.meta.get("best_so_far_combined_score"),
        "validity": record.meta.get("validity"),
        "eval_time": record.meta.get("eval_time"),
        "metrics_raw": record.meta.get("metrics_raw"),
        "call_roles": [c.role.value for c in record.calls],
        "call_model_names": [c.model_name for c in record.calls],
        "call_prompt_tokens": [c.prompt_tokens for c in record.calls],
        "call_completion_tokens": [c.completion_tokens for c in record.calls],
        "call_input_tokens": [c.prompt_tokens for c in record.calls],
        "call_output_tokens": [c.completion_tokens for c in record.calls],
        "call_total_tokens": [c.total_tokens for c in record.calls],
        "call_costs": [c.raw_cost for c in record.calls],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    ledger: BudgetLedger,
    best_score: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = ledger.summary()
    if best_score is not None:
        summary["best_score"] = float(best_score)
    if extra:
        summary.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_iterations(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
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


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plot_run_metric_vs_cost(
    iterations_path: Path,
    out_png: Path,
    *,
    y_keys: list[str],
    ylabel: str,
    title: str,
) -> bool:
    """Plot a chosen best-so-far metric vs cumulative cost for a single run.

    Returns False when plotting dependencies are unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not iterations_path.exists():
        return False

    costs = []
    values = []
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
            metric_val = None
            for key in y_keys:
                if row.get(key) is not None:
                    metric_val = row.get(key)
                    break
            values.append(metric_val)

    if not costs:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(costs, values, linewidth=1.8)
    plt.xlabel("Cumulative cost (USD)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def plot_run_best_score_vs_cost(iterations_path: Path, out_png: Path) -> bool:
    """Backward-compatible best-score plot."""
    return plot_run_metric_vs_cost(
        iterations_path,
        out_png,
        y_keys=["global_best_after", "best_so_far_combined_score", "combined_score"],
        ylabel="Best score",
        title="Best score vs cumulative cost",
    )


def plot_run_budget_panels(iterations_path: Path, out_png: Path) -> bool:
    """Create a compact multi-panel budget report for one run."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not iterations_path.exists():
        return False

    rows = []
    with iterations_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return False

    x_iter = [int(r.get("iteration", i)) for i, r in enumerate(rows)]
    y_iter_cost = [float(r.get("iteration_cost", 0.0) or 0.0) for r in rows]
    y_cum_cost = [float(r.get("cumulative_cost", 0.0) or 0.0) for r in rows]
    y_score = [r.get("global_best_after") for r in rows]
    y_tokens = [int(r.get("total_tokens", 0) or 0) for r in rows]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(x_iter, y_iter_cost, linewidth=1.5)
    axes[0, 0].set_title("Iteration cost")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("USD")

    axes[0, 1].plot(x_iter, y_cum_cost, linewidth=1.5)
    axes[0, 1].set_title("Cumulative cost")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("USD")

    axes[1, 0].plot(y_cum_cost, y_score, linewidth=1.5)
    axes[1, 0].set_title("Best score vs cumulative cost")
    axes[1, 0].set_xlabel("Cumulative cost (USD)")
    axes[1, 0].set_ylabel("Best score")

    axes[1, 1].plot(x_iter, y_tokens, linewidth=1.5)
    axes[1, 1].set_title("Tokens per iteration")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Tokens")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True
