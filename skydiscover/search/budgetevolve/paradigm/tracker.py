"""BudgetEvolve paradigm tracker wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from skydiscover.search.adaevolve.paradigm.tracker import ParadigmTracker as BaseParadigmTracker


@dataclass
class ParadigmTracker(BaseParadigmTracker):
    """Adds budget gate on top of AdaEvolve's stagnation tracker."""

    def should_trigger_budgeted(
        self,
        remaining_ratio: float,
        meta_budget_threshold: float,
        estimated_meta_cost: int,
        remaining_tokens: int,
        recent_meta_success_rate: Optional[float] = None,
    ) -> bool:
        if not self.is_paradigm_stagnating():
            return False
        if remaining_ratio <= meta_budget_threshold:
            return False
        if estimated_meta_cost >= remaining_tokens:
            return False
        if recent_meta_success_rate is not None and recent_meta_success_rate < 0.05:
            return False
        return True
