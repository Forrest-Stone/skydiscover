from __future__ import annotations

import csv
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
    num_guide_calls = sum(1 for c in record.calls if c.role.value == "guide")
    row: Dict[str, Any] = {
        "iteration": record.iteration,
        "source": record.meta.get("source", "iteration"),
        "frontier_id": record.meta.get("frontier_id"),
        "method": record.meta.get("method"),
        "task_family": record.meta.get("task_family"),
        "task_name": record.meta.get("task_name"),
        "seed": record.meta.get("seed"),
        "generation_cost": record.generation_cost,
        "retry_cost": record.retry_cost,
        "guide_cost": record.guide_cost,
        "iteration_cost": record.iteration_cost,
        "cumulative_cost": record.cumulative_cost,
        "remaining_budget_ratio": record.remaining_budget_ratio,
        "remaining_budget_ratio_before": record.meta.get(
            "remaining_budget_ratio_before",
            record.meta.get("remaining_budget_ratio"),
        ),
        "remaining_budget_ratio_after": record.meta.get(
            "remaining_budget_ratio_after",
            record.remaining_budget_ratio,
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        # Explicit aliases so downstream consumers can read in/out directly.
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "model_name": model_name,
        "candidate_score": record.meta.get("candidate_score"),
        "local_best": record.meta.get("local_best", record.meta.get("candidate_score")),
        "global_best_before": record.meta.get("global_best_before"),
        "global_best_after": record.meta.get("global_best_after"),
        "global_best": record.meta.get("global_best", record.meta.get("global_best_after")),
        "tier": record.meta.get("tier"),
        "base_tier": record.meta.get("base_tier", record.meta.get("tier")),
        "final_tier": record.meta.get(
            "final_tier", record.meta.get("action_tier", record.meta.get("tier"))
        ),
        "tier_override_reason": record.meta.get("tier_override_reason"),
        "intensity": record.meta.get("intensity", record.meta.get("recent_improvement_avg")),
        "lambda_t": record.meta.get("lambda_t"),
        "recent_improvement_avg": record.meta.get("recent_improvement_avg"),
        "stagnation_steps": record.meta.get("stagnation_steps"),
        "local_gain": record.meta.get("local_gain"),
        "global_gain": record.meta.get("global_gain"),
        "frontier_improvement": record.meta.get(
            "frontier_improvement", record.meta.get("frontier_gain")
        ),
        "local_gain_normalized": record.meta.get("local_gain_normalized"),
        "global_gain_normalized": record.meta.get("global_gain_normalized"),
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
        "routing_stat": record.meta.get("routing_stat", record.meta.get("router_reward")),
        "router_reward": record.meta.get("router_reward"),
        "meta_triggered": record.meta.get("meta_triggered", False),
        "meta_sources": record.meta.get("meta_sources", []),
        "guide_triggered": record.meta.get("guide_triggered", bool(num_guide_calls)),
        "paradigm_triggered": record.meta.get("paradigm_triggered", False),
        "paradigm_count": record.meta.get("paradigm_count", 0),
        "attempts_used": record.meta.get("attempts_used", 1),
        "num_calls": len(record.calls),
        "num_generation_calls_this_iteration": sum(
            1 for c in record.calls if c.role.value == "generation"
        ),
        "num_retry_calls_this_iteration": sum(1 for c in record.calls if c.role.value == "retry"),
        "num_guide_calls_this_iteration": num_guide_calls,
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
        "best_so_far_objective_iteration": record.meta.get("best_so_far_objective_iteration"),
        "target_value": record.meta.get("target_value"),
        "target_ratio": record.meta.get("target_ratio"),
        "best_so_far_target_ratio": record.meta.get("best_so_far_target_ratio"),
        "best_so_far_target_ratio_iteration": record.meta.get("best_so_far_target_ratio_iteration"),
        "combined_score": record.meta.get("combined_score"),
        "best_so_far_combined_score": record.meta.get("best_so_far_combined_score"),
        "best_so_far_combined_score_iteration": record.meta.get(
            "best_so_far_combined_score_iteration"
        ),
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
        "calls": [
            {
                "role": c.role.value,
                "model_name": c.model_name,
                "input_tokens": c.prompt_tokens,
                "output_tokens": c.completion_tokens,
                "total_tokens": c.total_tokens,
                "raw_cost": c.raw_cost,
                "meta": c.meta,
            }
            for c in record.calls
        ],
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


_ITERATION_CSV_PREFIX_FIELDS = [
    "iteration",
    "source",
    "method",
    "task_family",
    "task_name",
    "seed",
    "frontier_id",
    "objective_key",
    "objective_value",
    "best_so_far_objective",
    "best_so_far_objective_iteration",
    "combined_score",
    "best_so_far_combined_score",
    "best_so_far_combined_score_iteration",
    "candidate_score",
    "local_best",
    "global_best_before",
    "global_best_after",
    "global_best",
    "target_value",
    "target_ratio",
    "best_so_far_target_ratio",
    "best_so_far_target_ratio_iteration",
    "generation_cost",
    "retry_cost",
    "guide_cost",
    "iteration_cost",
    "cumulative_cost",
    "remaining_budget_ratio",
    "remaining_budget_ratio_before",
    "remaining_budget_ratio_after",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "generation_input_tokens",
    "generation_output_tokens",
    "retry_input_tokens",
    "retry_output_tokens",
    "guide_input_tokens",
    "guide_output_tokens",
    "num_calls",
    "num_generation_calls_this_iteration",
    "num_retry_calls_this_iteration",
    "num_guide_calls_this_iteration",
    "lambda_t",
    "local_gain",
    "global_gain",
    "local_gain_normalized",
    "global_gain_normalized",
    "frontier_improvement",
    "utility",
    "frontier_signal",
    "routing_reward",
    "routing_stat",
    "router_reward",
    "recent_improvement_avg",
    "stagnation_steps",
    "intensity",
    "tier",
    "base_tier",
    "final_tier",
    "tier_override_reason",
    "meta_triggered",
    "guide_triggered",
    "meta_sources",
    "paradigm_triggered",
    "paradigm_count",
    "attempts_used",
    "model_name",
    "validity",
    "eval_time",
    "metrics_raw",
    "calls",
]

_CALL_CSV_PREFIX_FIELDS = [
    "iteration",
    "source",
    "method",
    "task_family",
    "task_name",
    "seed",
    "frontier_id",
    "call_index",
    "role",
    "model_name",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "raw_cost",
    "iteration_cost_before_call",
    "iteration_cost_after_call",
    "cumulative_cost_before_call",
    "cumulative_cost_after_call",
    "cumulative_cost",
    "call_meta",
    "generation_cost",
    "retry_cost",
    "guide_cost",
    "iteration_cost",
    "cumulative_cost_before_iteration",
    "objective_key",
    "combined_score",
    "objective_value",
    "cumulative_cost_after_iteration",
]

_SUMMARY_CSV_PREFIX_FIELDS = [
    "method",
    "task_family",
    "task_name",
    "seed",
    "nominal_budget",
    "total_cost",
    "avg_iteration_cost",
    "oob",
    "overshoot",
    "overshoot_ratio",
    "best_score",
    "objective_key",
    "best_objective",
    "best_objective_iteration",
    "best_combined_score",
    "best_combined_score_iteration",
    "best_target_ratio",
    "best_target_ratio_iteration",
    "target_value",
    "success_target",
    "cost_to_target",
    "iteration_to_target",
    "num_iterations",
    "num_generation_calls",
    "num_retry_calls",
    "num_guide_calls",
    "generation_cost_total",
    "retry_cost_total",
    "guide_cost_total",
    "input_tokens_total",
    "output_tokens_total",
    "total_tokens",
]


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _ordered_fields(rows: list[Dict[str, Any]], preferred: list[str]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    row_keys = {key for row in rows for key in row.keys()}
    for key in preferred:
        if key in row_keys and key not in seen:
            fields.append(key)
            seen.add(key)
    for key in sorted(row_keys - seen):
        fields.append(key)
    return fields


def _write_dict_rows_csv(
    rows: list[Dict[str, Any]],
    out_csv: Path,
    preferred_fields: list[str],
) -> bool:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return False
    fields = _ordered_fields(rows, preferred_fields)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fields})
    return True


def export_iterations_csv(iterations_path: Path, out_csv: Path | None = None) -> bool:
    """Export iterations.jsonl to a flat CSV without dropping diagnostic fields."""
    rows = load_iterations(iterations_path)
    return _write_dict_rows_csv(
        rows,
        out_csv or iterations_path.with_suffix(".csv"),
        _ITERATION_CSV_PREFIX_FIELDS,
    )


def _list_at(values: Any, idx: int, default: Any = None) -> Any:
    if not isinstance(values, list) or idx >= len(values):
        return default
    return values[idx]


def _float_or_zero(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def calls_from_iteration_row(row: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return normalized per-call rows from one iteration JSON row.

    New traces store a structured ``calls`` list. Older traces only have
    parallel arrays such as ``call_roles`` and ``call_costs``; keeping that
    fallback makes aggregate CSV export work for existing runs too.
    """
    calls = row.get("calls")
    if isinstance(calls, list) and calls:
        out: list[Dict[str, Any]] = []
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            out.append(
                {
                    "call_index": idx,
                    "role": call.get("role"),
                    "model_name": call.get("model_name"),
                    "input_tokens": call.get("input_tokens", call.get("prompt_tokens")),
                    "output_tokens": call.get(
                        "output_tokens", call.get("completion_tokens")
                    ),
                    "total_tokens": call.get("total_tokens"),
                    "raw_cost": call.get("raw_cost"),
                    "call_meta": call.get("meta"),
                }
            )
        return out

    roles = row.get("call_roles")
    models = row.get("call_model_names")
    input_tokens = row.get("call_input_tokens", row.get("call_prompt_tokens"))
    output_tokens = row.get("call_output_tokens", row.get("call_completion_tokens"))
    total_tokens = row.get("call_total_tokens")
    costs = row.get("call_costs")
    lengths = [
        len(v)
        for v in (roles, models, input_tokens, output_tokens, total_tokens, costs)
        if isinstance(v, list)
    ]
    if not lengths:
        return []
    out = []
    for idx in range(max(lengths)):
        out.append(
            {
                "call_index": idx,
                "role": _list_at(roles, idx),
                "model_name": _list_at(models, idx),
                "input_tokens": _list_at(input_tokens, idx, 0),
                "output_tokens": _list_at(output_tokens, idx, 0),
                "total_tokens": _list_at(total_tokens, idx, 0),
                "raw_cost": _list_at(costs, idx, 0.0),
                "call_meta": None,
            }
        )
    return out


def export_calls_csv(iterations_path: Path, out_csv: Path | None = None) -> bool:
    """Export one row per search-side LLM call from an iteration trace."""
    rows = load_iterations(iterations_path)
    call_rows: list[Dict[str, Any]] = []
    running_run_cost = 0.0
    for row in rows:
        calls = calls_from_iteration_row(row)
        iteration_cost = _float_or_zero(row.get("iteration_cost"))
        if iteration_cost == 0.0 and calls:
            iteration_cost = sum(_float_or_zero(call.get("raw_cost")) for call in calls)
        if row.get("cumulative_cost") in (None, ""):
            cumulative_before_iteration = running_run_cost
            cumulative_after_iteration = cumulative_before_iteration + iteration_cost
        else:
            cumulative_after_iteration = _float_or_zero(row.get("cumulative_cost"))
            cumulative_before_iteration = max(cumulative_after_iteration - iteration_cost, 0.0)
        run_meta = {
            "iteration": row.get("iteration"),
            "source": row.get("source"),
            "method": row.get("method"),
            "task_family": row.get("task_family"),
            "task_name": row.get("task_name"),
            "seed": row.get("seed"),
            "frontier_id": row.get("frontier_id"),
            "objective_key": row.get("objective_key"),
            "combined_score": row.get("combined_score"),
            "objective_value": row.get("objective_value"),
            "generation_cost": row.get("generation_cost"),
            "retry_cost": row.get("retry_cost"),
            "guide_cost": row.get("guide_cost"),
            "iteration_cost": (
                row.get("iteration_cost")
                if row.get("iteration_cost") not in (None, "")
                else iteration_cost
            ),
            "cumulative_cost_before_iteration": cumulative_before_iteration,
            "cumulative_cost_after_iteration": (
                row.get("cumulative_cost")
                if row.get("cumulative_cost") not in (None, "")
                else cumulative_after_iteration
            ),
        }
        iteration_running_cost = 0.0
        for call in calls:
            raw_cost = _float_or_zero(call.get("raw_cost"))
            before_call = iteration_running_cost
            after_call = before_call + raw_cost
            call_rows.append(
                {
                    **run_meta,
                    **call,
                    "iteration_cost_before_call": before_call,
                    "iteration_cost_after_call": after_call,
                    "cumulative_cost_before_call": cumulative_before_iteration + before_call,
                    "cumulative_cost_after_call": cumulative_before_iteration + after_call,
                    "cumulative_cost": cumulative_before_iteration + after_call,
                }
            )
            iteration_running_cost = after_call
        running_run_cost = cumulative_after_iteration

    default_out = iterations_path.with_name("calls.csv")
    return _write_dict_rows_csv(call_rows, out_csv or default_out, _CALL_CSV_PREFIX_FIELDS)


def export_summary_csv(summary_path: Path, out_csv: Path | None = None) -> bool:
    """Export summary.json as a single-row CSV."""
    summary = load_summary(summary_path)
    if not summary:
        out = out_csv or summary_path.with_suffix(".csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return False
    return _write_dict_rows_csv(
        [summary],
        out_csv or summary_path.with_suffix(".csv"),
        _SUMMARY_CSV_PREFIX_FIELDS,
    )


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_budget_row_for_iteration(
    rows: list[Dict[str, Any]],
    iteration: int | None,
) -> tuple[Dict[str, Any] | None, str | None]:
    if iteration is None:
        return None, None

    for row in rows:
        if _as_int(row.get("iteration")) == iteration:
            return row, "iteration"

    # Older summaries may only expose the iteration that first reached a best
    # value.  Keep this fallback so best_program_info can still link to a trace.
    for key in (
        "best_so_far_combined_score_iteration",
        "best_so_far_objective_iteration",
    ):
        for row in rows:
            if _as_int(row.get(key)) == iteration:
                return row, key

    return None, None


def best_program_budget_info(output_dir: Path | str, program_iteration: Any) -> Dict[str, Any]:
    """Collect cost/budget context for the saved best program.

    The returned dict is intended to be embedded into best_program_info.json.
    It includes the run-level summary plus the exact iteration trace row for
    the best program when that row is available.
    """
    root = Path(output_dir)
    summary = load_summary(root / "summary.json")
    rows = load_iterations(root / "iterations.jsonl")
    iteration = _as_int(program_iteration)
    best_row, matched_by = _find_budget_row_for_iteration(rows, iteration)

    if not summary and best_row is None:
        return {}

    info: Dict[str, Any] = {
        "program_iteration": iteration,
        "matched_iteration_by": matched_by,
    }
    if summary:
        info["run_summary"] = summary
    if best_row is not None:
        info["best_iteration_trace"] = best_row
        info["cost_at_best"] = best_row.get("cumulative_cost")
        info["iteration_cost_at_best"] = best_row.get("iteration_cost")
        info["remaining_budget_ratio_at_best"] = best_row.get(
            "remaining_budget_ratio_after",
            best_row.get("remaining_budget_ratio"),
        )
        info["tokens_at_best"] = {
            "input_tokens": best_row.get("input_tokens"),
            "output_tokens": best_row.get("output_tokens"),
            "total_tokens": best_row.get("total_tokens"),
        }
        info["call_counts_at_best"] = {
            "num_calls": best_row.get("num_calls"),
            "generation": best_row.get("num_generation_calls_this_iteration"),
            "retry": best_row.get("num_retry_calls_this_iteration"),
            "guide": best_row.get("num_guide_calls_this_iteration"),
        }
    return info


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _first_present(row: Dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row.get(key)
    return None


def _series(rows: list[Dict[str, Any]], keys: list[str]) -> list[Any]:
    return [_first_present(row, keys) for row in rows]


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
    plt = _load_plt()
    if plt is None:
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


def plot_run_metric_vs_iteration(
    iterations_path: Path,
    out_png: Path,
    *,
    y_keys: list[str],
    ylabel: str,
    title: str,
) -> bool:
    """Plot a chosen metric vs iteration index for a single run."""
    plt = _load_plt()
    if plt is None:
        return False

    if not iterations_path.exists():
        return False

    x_iter = []
    values = []
    with iterations_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            x_iter.append(int(row.get("iteration", i)))
            metric_val = None
            for key in y_keys:
                if row.get(key) is not None:
                    metric_val = row.get(key)
                    break
            values.append(metric_val)

    if not x_iter:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x_iter, values, linewidth=1.8)
    plt.xlabel("Iteration")
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
    plt = _load_plt()
    if plt is None:
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


def plot_run_performance_panels(iterations_path: Path, out_png: Path) -> bool:
    """Group score/objective vs cost/iteration views into one run-level figure."""
    plt = _load_plt()
    if plt is None:
        return False
    rows = load_iterations(iterations_path)
    if not rows:
        return False

    x_iter = [int(r.get("iteration", i)) for i, r in enumerate(rows)]
    x_cost = [float(r.get("cumulative_cost", 0.0) or 0.0) for r in rows]
    best_score = _series(rows, ["global_best_after", "global_best", "best_so_far_combined_score"])
    best_objective = _series(rows, ["best_so_far_objective", "objective_value", "global_best_after"])
    best_combined = _series(rows, ["best_so_far_combined_score", "combined_score", "global_best_after"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(x_cost, best_score, linewidth=1.7)
    axes[0, 0].set_title("Best score vs cost")
    axes[0, 0].set_xlabel("Cumulative cost (USD)")
    axes[0, 0].set_ylabel("Best score")

    axes[0, 1].plot(x_iter, best_score, linewidth=1.7)
    axes[0, 1].set_title("Best score vs iteration")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Best score")

    axes[1, 0].plot(x_cost, best_objective, linewidth=1.7)
    axes[1, 0].set_title("Best objective vs cost")
    axes[1, 0].set_xlabel("Cumulative cost (USD)")
    axes[1, 0].set_ylabel("Best-so-far objective")

    axes[1, 1].plot(x_cost, best_combined, linewidth=1.7)
    axes[1, 1].set_title("Best combined score vs cost")
    axes[1, 1].set_xlabel("Cumulative cost (USD)")
    axes[1, 1].set_ylabel("Best-so-far combined score")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def plot_run_cost_panels(iterations_path: Path, out_png: Path) -> bool:
    """Group iteration/cumulative cost and token composition views."""
    plt = _load_plt()
    if plt is None:
        return False
    rows = load_iterations(iterations_path)
    if not rows:
        return False

    x_iter = [int(r.get("iteration", i)) for i, r in enumerate(rows)]
    gen = [float(r.get("generation_cost", 0.0) or 0.0) for r in rows]
    retry = [float(r.get("retry_cost", 0.0) or 0.0) for r in rows]
    guide = [float(r.get("guide_cost", 0.0) or 0.0) for r in rows]
    iter_cost = [float(r.get("iteration_cost", 0.0) or 0.0) for r in rows]
    cum_cost = [float(r.get("cumulative_cost", 0.0) or 0.0) for r in rows]
    remain = [float(r.get("remaining_budget_ratio_after", r.get("remaining_budget_ratio", 0.0)) or 0.0) for r in rows]
    in_tokens = [int(r.get("input_tokens", 0) or 0) for r in rows]
    out_tokens = [int(r.get("output_tokens", 0) or 0) for r in rows]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(x_iter, iter_cost, linewidth=1.5)
    axes[0, 0].set_title("Iteration cost")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("USD")

    axes[0, 1].stackplot(x_iter, gen, retry, guide, labels=["generation", "retry", "guide"])
    axes[0, 1].set_title("Cost composition")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("USD")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x_iter, cum_cost, linewidth=1.5, label="cumulative cost")
    axes[1, 0].set_title("Cumulative cost")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("USD")

    axes[1, 1].plot(x_iter, in_tokens, linewidth=1.3, label="input")
    axes[1, 1].plot(x_iter, out_tokens, linewidth=1.3, label="output")
    axes[1, 1].plot(x_iter, remain, linewidth=1.3, label="remaining ratio")
    axes[1, 1].set_title("Tokens and remaining budget")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def plot_run_diagnostic_panels(iterations_path: Path, out_png: Path) -> bool:
    """Group BCHD diagnostics: tiers, meta triggers, frontier share, utility chain."""
    plt = _load_plt()
    if plt is None:
        return False
    rows = load_iterations(iterations_path)
    if not rows:
        return False

    x_iter = [int(r.get("iteration", i)) for i, r in enumerate(rows)]
    tier_map = {"cheap": 0, "standard": 1, "rich": 2}
    tiers = [
        tier_map.get(str(r.get("final_tier", r.get("base_tier", r.get("tier", "")))), None)
        for r in rows
    ]
    meta = [1.0 if r.get("meta_triggered") else 0.0 for r in rows]
    utility_vals = _series(rows, ["utility"])
    frontier_signal = _series(rows, ["frontier_signal"])
    local_gain_vals = _series(rows, ["local_gain_normalized", "local_gain"])
    global_gain_vals = _series(rows, ["global_gain_normalized", "global_gain"])
    frontier_counts: Dict[str, int] = {}
    for row in rows:
        fid = row.get("frontier_id")
        if fid is not None:
            frontier_counts[str(fid)] = frontier_counts.get(str(fid), 0) + 1

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(x_iter, tiers, drawstyle="steps-post", linewidth=1.5)
    axes[0, 0].set_yticks([0, 1, 2], ["cheap", "standard", "rich"])
    axes[0, 0].set_title("Tier usage vs iteration")
    axes[0, 0].set_xlabel("Iteration")

    axes[0, 1].plot(x_iter, meta, linewidth=1.5)
    axes[0, 1].set_title("Meta trigger vs iteration")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Triggered")

    if frontier_counts:
        labels = sorted(frontier_counts.keys())
        vals = [frontier_counts[k] for k in labels]
        axes[1, 0].bar(labels, vals)
    axes[1, 0].set_title("Frontier selection counts")
    axes[1, 0].set_xlabel("Frontier")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].plot(x_iter, utility_vals, linewidth=1.2, label="utility")
    axes[1, 1].plot(x_iter, frontier_signal, linewidth=1.2, label="frontier signal")
    axes[1, 1].plot(x_iter, local_gain_vals, linewidth=1.0, label="local gain")
    axes[1, 1].plot(x_iter, global_gain_vals, linewidth=1.0, label="global gain")
    axes[1, 1].set_title("Utility chain")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True
