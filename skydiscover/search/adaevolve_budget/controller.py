from __future__ import annotations

import time
from typing import Any, Optional

from skydiscover.budget import CallRole
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.utils.budget_iteration import BudgetIterationMixin
from skydiscover.search.utils.discovery_utils import SerializableResult


class AdaEvolveBudgetController(BudgetIterationMixin, AdaEvolveController):
    """AdaEvolve wrapper that adds per-iteration budget/cost tracing."""

    def __init__(self, controller_input):
        super().__init__(controller_input)
        self._active_budget_record = None
        self._current_call_role = CallRole.GENERATION

    async def _call_llm(self, system_message: str, user_message: str, **kwargs):
        if "_budget_record" not in kwargs and self._active_budget_record is not None:
            kwargs["_budget_record"] = self._active_budget_record
        if "_call_role" not in kwargs:
            kwargs["_call_role"] = self._current_call_role
        return await super()._call_llm(system_message, user_message, **kwargs)

    async def _run_normal_step(self, iteration: int) -> SerializableResult:
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)

        for attempt in range(attempts):
            self._budget_set_call_role(
                CallRole.GENERATION if attempt == 0 else CallRole.RETRY
            )
            result = await self._generate_child(iteration, error_context=last_error)
            if not result.error:
                result.attempts_used = attempt + 1
                return result
            last_error = result.error

        return SerializableResult(
            error=f"All {attempts} attempts failed: {last_error}",
            iteration=iteration,
            attempts_used=attempts,
        )

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        iteration_start_time = time.time()
        budget_record = self._budget_start_iteration(iteration)
        result: Optional[SerializableResult] = None
        finalized = False

        try:
            if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
                await self._budget_generate_paradigms_if_needed(budget_record)

            result = await self._run_normal_step(iteration)
            result.iteration_time = max(float(result.iteration_time or 0.0), time.time() - iteration_start_time)

            if result.error:
                self._log_iteration_stats(
                    iteration=iteration,
                    sampling_mode=getattr(self, "_last_sampling_mode", None),
                    sampling_intensity=getattr(self, "_last_sampling_intensity", None),
                    child_program=None,
                    iteration_time=result.iteration_time,
                    llm_generation_time=result.llm_generation_time,
                    eval_time=result.eval_time,
                    error=result.error,
                )
            else:
                self._process_result(result, iteration, checkpoint_callback)
                self._log_iteration_stats(
                    iteration=iteration,
                    sampling_mode=getattr(self, "_last_sampling_mode", None),
                    sampling_intensity=getattr(self, "_last_sampling_intensity", None),
                    child_program=result.child_program_dict,
                    iteration_time=result.iteration_time,
                    llm_generation_time=result.llm_generation_time,
                    eval_time=result.eval_time,
                    error=None,
                )

            self._budget_finalize_iteration(
                budget_record,
                result,
                extra_meta={
                    "attempts_used": int(result.attempts_used or 1),
                    "tier": getattr(self, "_last_sampling_mode", None),
                    "recent_improvement_avg": getattr(self, "_last_sampling_intensity", None),
                },
            )
            finalized = True
        except Exception as exc:
            result = self._budget_error_result(
                iteration,
                iteration_start_time,
                f"Budget wrapper iteration error: {exc}",
            )
        finally:
            if not finalized and result is not None:
                try:
                    self._budget_finalize_iteration(budget_record, result)
                except Exception:
                    pass
            self._budget_reset_context()

    async def run_discovery(self, *args: Any, **kwargs: Any):
        out = await super().run_discovery(*args, **kwargs)
        self._write_budget_summary()
        return out
