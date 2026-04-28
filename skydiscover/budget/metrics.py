from __future__ import annotations

from statistics import mean
from typing import Iterable, Optional


BUDGET_REPORTING_PRECISION = 4


def is_oob(total_cost: float, nominal_budget: float) -> bool:
    return total_cost > nominal_budget


def overshoot(total_cost: float, nominal_budget: float) -> float:
    return max(total_cost - nominal_budget, 0.0)


def best_score_at_budget(points: Iterable[tuple[float, float]], budget: float) -> float | None:
    best = None
    for cost, score in points:
        if cost <= budget:
            best = score if best is None else max(best, score)
    return best


def success_at_target(best_score: Optional[float], target: float) -> float:
    if best_score is None:
        return 0.0
    return 1.0 if float(best_score) >= float(target) else 0.0


def cost_to_target(points: Iterable[tuple[float, float]], target: float) -> float | None:
    for cost, score in points:
        try:
            if score is not None and float(score) >= float(target):
                return float(cost)
        except (TypeError, ValueError):
            continue
    return None


def overshoot_ratio(total_cost: float, nominal_budget: float) -> float:
    if nominal_budget <= 0:
        return 0.0
    return max((total_cost - nominal_budget) / nominal_budget, 0.0)


def avg_cost(costs: Iterable[float]) -> float:
    """Compute average realized cost across runs."""
    vals = [float(c) for c in costs]
    if not vals:
        return 0.0
    return float(mean(vals))


def speedup_at_target(method_cost: Optional[float], baseline_cost: Optional[float]) -> float | None:
    """Compute speedup to target as baseline_cost / method_cost.

    Returns None when either side is missing or non-positive.
    """
    if method_cost is None or baseline_cost is None:
        return None
    m = float(method_cost)
    b = float(baseline_cost)
    if m <= 0.0 or b <= 0.0:
        return None
    return b / m


def rounded_metric_value(
    value: Optional[float],
    *,
    precision: int = BUDGET_REPORTING_PRECISION,
) -> float | None:
    """Round metrics exactly as budget reports/plots present them."""
    if value is None:
        return None
    try:
        return round(float(value), int(precision))
    except (TypeError, ValueError):
        return None


def best_so_far_with_iteration(
    previous_value: Optional[float],
    previous_iteration: Optional[int],
    candidate_value: Optional[float],
    candidate_iteration: int,
    *,
    precision: int = BUDGET_REPORTING_PRECISION,
) -> tuple[float | None, int | None]:
    """Update a best-so-far value while keeping first displayed-tie iteration.

    The raw best value may improve by tiny amounts, but when the old and new
    values round to the same displayed precision we keep the first iteration
    that reached that displayed best. This matches the four-decimal reporting
    semantics used in budget traces.
    """
    candidate_rounded = rounded_metric_value(candidate_value, precision=precision)
    if candidate_rounded is None:
        return previous_value, previous_iteration

    previous_rounded = rounded_metric_value(previous_value, precision=precision)
    if previous_rounded is None:
        return float(candidate_value), int(candidate_iteration)

    if candidate_rounded > previous_rounded:
        return float(candidate_value), int(candidate_iteration)

    if candidate_rounded == previous_rounded:
        if previous_iteration is None:
            return float(candidate_value), int(candidate_iteration)
        raw_previous = float(previous_value) if previous_value is not None else float("-inf")
        raw_candidate = float(candidate_value)
        return max(raw_previous, raw_candidate), previous_iteration

    return previous_value, previous_iteration
