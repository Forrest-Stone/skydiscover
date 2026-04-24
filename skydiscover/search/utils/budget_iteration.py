"""Shared helpers for controllers that own custom iteration loops.

These helpers keep budget start/finalize/error-handling consistent across
algorithm-specific controllers (e.g. AdaEvolve/CostAda wrappers).
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from skydiscover.budget import CallRole
from skydiscover.search.utils.discovery_utils import SerializableResult


class BudgetIterationMixin:
    """Reusable budget-iteration lifecycle helpers."""

    def _budget_start_iteration(self, iteration: int, *, frontier_id=None, tier=None):
        record = self.budget_ledger.start_iteration(iteration)
        record.meta["frontier_id"] = (
            frontier_id if frontier_id is not None else getattr(self.database, "current_island", None)
        )
        record.meta["global_best_before"] = self._best_score_or_zero()
        record.meta["tier"] = tier if tier is not None else getattr(self, "_last_sampling_mode", None)
        self._active_budget_record = record
        return record

    def _budget_set_call_role(self, role: CallRole) -> None:
        self._current_call_role = role

    def _budget_finalize_iteration(
        self,
        budget_record,
        result: SerializableResult,
        *,
        extra_meta: Optional[Dict] = None,
    ) -> None:
        if extra_meta:
            budget_record.meta.update(extra_meta)
        self._finalize_budget_iteration(budget_record, result)

    def _budget_error_result(self, iteration: int, start_time: float, message: str) -> SerializableResult:
        return SerializableResult(
            error=message,
            iteration=iteration,
            attempts_used=1,
            iteration_time=(time.time() - start_time),
        )

    def _budget_reset_context(self) -> None:
        self._active_budget_record = None
        self._current_call_role = CallRole.GENERATION
