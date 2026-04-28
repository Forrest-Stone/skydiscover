"""Shared helpers for controllers that own custom iteration loops.

These helpers keep budget start/finalize/error-handling consistent across
algorithm-specific controllers (e.g. AdaEvolve/CostAda wrappers).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from skydiscover.budget import CallRole, call_record_from_response
from skydiscover.search.adaevolve.paradigm.generator import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,
)
from skydiscover.search.utils.discovery_utils import SerializableResult

logger = logging.getLogger(__name__)


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

    async def _budget_generate_paradigms_if_needed(self, budget_record) -> None:
        """AdaEvolve paradigm generation with guide-call budget accounting.

        This mirrors AdaEvolveController._generate_paradigms_if_needed without
        touching the baseline implementation. Every LLM call made for
        high-level guidance is attached to the current iteration as GUIDE cost.
        """
        if getattr(self, "paradigm_generator", None) is None:
            return

        if self.database.has_active_paradigm():
            return

        logger.info("Global paradigm stagnation detected, generating breakthrough ideas...")

        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program) if best_program else 0.0

        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        previously_tried = self.database.get_previously_tried_ideas()
        prompt = self.paradigm_generator._build_prompt(
            best_solution,
            best_score,
            previously_tried or [],
            evaluator_feedback=evaluator_feedback,
        )

        last_error = None
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.guide_llms.generate_with_usage(
                    system_message=self.paradigm_generator._get_system_message(),
                    messages=[{"role": "user", "content": prompt}],
                )
                self.budget_ledger.add_call(
                    budget_record,
                    call_record_from_response(
                        response,
                        CallRole.GUIDE,
                        source="paradigm_generation",
                        attempt=attempt + 1,
                    ),
                )

                if not response.text:
                    logger.warning(
                        "Empty paradigm response from LLM (attempt %s/%s)",
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    last_error = "Empty response"
                    break

                paradigms = self.paradigm_generator._parse_response(response.text)
                if not paradigms:
                    logger.warning(
                        "Failed to parse paradigms (attempt %s/%s)",
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    last_error = "Parse failure"
                    break

                self.database.set_paradigms(paradigms)
                budget_record.meta["paradigm_triggered"] = True
                budget_record.meta["guide_triggered"] = True
                budget_record.meta["paradigm_count"] = len(paradigms)
                logger.info("Generated %s breakthrough paradigms", len(paradigms))
                return

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Paradigm generation failed (attempt %s/%s): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    logger.info("Retrying in %.1fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER

        budget_record.meta.setdefault("paradigm_triggered", False)
        budget_record.meta.setdefault("paradigm_count", 0)
        logger.error("Paradigm generation failed after %s attempts: %s", MAX_RETRIES, last_error)
