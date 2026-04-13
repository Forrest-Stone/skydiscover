from __future__ import annotations

from typing import Iterable


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
