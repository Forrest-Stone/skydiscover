from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from skydiscover.budget import CallCostRecord, CallRole, call_record_from_response
from skydiscover.search.evox.controller import CoEvolutionController
from skydiscover.search.evox.utils.template import DEFAULT_DIVERGE_TEMPLATE, DEFAULT_REFINE_TEMPLATE
from skydiscover.search.evox.utils.variation_operator_generator import (
    COMBINED_SYSTEM_PROMPT,
    _build_operator_prompt,
    _operators_from_response,
)
from skydiscover.search.utils.discovery_utils import SerializableResult, load_evaluator_code

logger = logging.getLogger(__name__)


class CoEvolutionBudgetController(CoEvolutionController):
    """EvoX wrapper that adds budget tracing without changing EvoX logic."""

    def __init__(self, controller_input):
        super().__init__(controller_input)
        self._copied_search_budget_records = 0

    def _record_guide_response(self, response, *, source: str, iteration: int) -> None:
        record = self.budget_ledger.start_iteration(iteration)
        record.meta.update(
            {
                "source": source,
                "method": self.config.search.type,
                "global_best_before": self._best_score_or_zero(),
                "guide_triggered": True,
                "meta_triggered": True,
            }
        )
        self.budget_ledger.add_call(
            record,
            call_record_from_response(response, CallRole.GUIDE, source=source),
        )
        self._finalize_budget_iteration(
            record,
            SerializableResult(iteration=iteration, attempts_used=1),
        )

    def _record_inner_search_budget(self, *, solution_iter: int) -> None:
        records = getattr(self.search_controller.budget_ledger, "records", [])
        new_records = records[self._copied_search_budget_records :]
        for source_record in new_records:
            record = self.budget_ledger.start_iteration(solution_iter)
            record.meta.update(
                {
                    "source": "evox_search_algorithm_generation",
                    "method": self.config.search.type,
                    "global_best_before": self._best_score_or_zero(),
                    "guide_triggered": True,
                    "meta_triggered": True,
                    "inner_search_iteration": source_record.iteration,
                }
            )
            for call in source_record.calls:
                self.budget_ledger.add_call(
                    record,
                    CallCostRecord(
                        role=CallRole.GUIDE,
                        model_name=call.model_name,
                        prompt_tokens=call.prompt_tokens,
                        completion_tokens=call.completion_tokens,
                        total_tokens=call.total_tokens,
                        raw_cost=call.raw_cost,
                        meta={
                            **(call.meta or {}),
                            "source": "evox_search_algorithm_generation",
                            "original_role": call.role.value,
                            "inner_search_iteration": source_record.iteration,
                        },
                    ),
                )
            self._finalize_budget_iteration(
                record,
                SerializableResult(iteration=solution_iter, attempts_used=1),
            )
        self._copied_search_budget_records = len(records)

    async def _generate_variation_operators(self) -> None:
        """Generate EvoX labels with guide-call cost recording."""
        if self._diverge_label and self._refine_label:
            self._assign_labels_to_db(self.database)
            return

        db_cfg = self.config.search.database
        if not getattr(db_cfg, "auto_generate_variation_operators", True):
            self._diverge_label = DEFAULT_DIVERGE_TEMPLATE
            self._refine_label = DEFAULT_REFINE_TEMPLATE
            logger.info(
                "Using default variation operators (auto_generate_variation_operators=false)"
            )
            self._assign_labels_to_db(self.database)
            return

        system_message = self.config.context_builder.system_message or ""
        evaluator_code = load_evaluator_code(self.evaluation_file)
        problem_dir = Path(self.evaluation_file).parent if self.evaluation_file else None
        label_llms = self.search_controller.guide_llms
        model_names = ", ".join(m.name for m in label_llms.models_cfg)
        logger.info("Label generation: using guide_model = [%s]", model_names)

        user_prompt = _build_operator_prompt(
            system_message,
            evaluator_code,
            problem_dir=problem_dir,
        )
        try:
            response = await label_llms.generate_with_usage(
                system_message=COMBINED_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            self._record_guide_response(
                response,
                source="evox_variation_operator_generation",
                iteration=-1,
            )
            self._diverge_label, self._refine_label = _operators_from_response(response.text)
            logger.info(
                "Generated variation operator labels (%s/%s chars)",
                len(self._diverge_label),
                len(self._refine_label),
            )
        except Exception as exc:
            self._diverge_label = ""
            self._refine_label = ""
            logger.error("Label generation failed: %s, setting labels to empty strings", exc)

        self._assign_labels_to_db(self.database)

    async def _generate_and_validate_search_algorithm(self, solution_iter: int) -> None:
        await super()._generate_and_validate_search_algorithm(solution_iter)
        self._record_inner_search_budget(solution_iter=solution_iter)

    async def run_discovery(self, *args: Any, **kwargs: Any):
        out = await super().run_discovery(*args, **kwargs)
        self._write_budget_summary()
        return out
