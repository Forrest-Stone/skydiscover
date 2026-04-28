from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from skydiscover.budget import (
    CallCostRecord,
    CallRole,
    call_record_from_response,
    write_iteration_record,
    write_summary,
)
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
        self._pending_iteration_guide_calls: List[CallCostRecord] = []

    def _queue_guide_response(self, response, *, source: str) -> None:
        self._pending_iteration_guide_calls.append(
            call_record_from_response(response, CallRole.GUIDE, source=source)
        )

    def _find_solution_budget_record(self, iteration: int):
        for record in reversed(self.budget_ledger.records):
            if int(record.iteration) != int(iteration):
                continue
            if record.meta.get("source", "iteration") == "iteration":
                return record
        return None

    def _recompute_budget_totals_after_mutation(self) -> None:
        cumulative = 0.0
        nominal = max(float(self.budget_ledger.config.nominal_budget), self.budget_ledger.config.eps)
        for record in self.budget_ledger.records:
            record.iteration_cost = (
                float(record.generation_cost or 0.0)
                + float(record.retry_cost or 0.0)
                + float(record.guide_cost or 0.0)
            )
            cumulative += record.iteration_cost
            record.cumulative_cost = cumulative
            record.remaining_budget_ratio = max(0.0, 1.0 - cumulative / nominal)
            record.meta["remaining_budget_ratio_after"] = record.remaining_budget_ratio
        self.budget_ledger.cumulative_cost = cumulative

    def _rewrite_budget_artifacts_after_mutation(self) -> None:
        self._recompute_budget_totals_after_mutation()
        self._budget_iterations_path.parent.mkdir(parents=True, exist_ok=True)
        self._budget_iterations_path.write_text("", encoding="utf-8")
        for record in self.budget_ledger.records:
            write_iteration_record(self._budget_iterations_path, record)
        write_summary(
            self._budget_summary_path,
            self.budget_ledger,
            best_score=self._best_score_or_zero(),
            extra=self._build_budget_summary_extra(),
        )
        self._export_budget_csv_artifacts()

    def _attach_guide_calls_to_iteration(
        self,
        iteration: int,
        calls: List[CallCostRecord],
        *,
        source: str,
    ) -> None:
        if not calls:
            return
        record = self._find_solution_budget_record(iteration)
        if record is None:
            self._pending_iteration_guide_calls.extend(calls)
            return
        for call in calls:
            self.budget_ledger.add_call(record, call)
        record.meta["guide_triggered"] = True
        record.meta["meta_triggered"] = True
        sources = list(record.meta.get("meta_sources", []) or [])
        sources.append(source)
        record.meta["meta_sources"] = sorted(set(str(s) for s in sources))
        self._rewrite_budget_artifacts_after_mutation()

    def _flush_pending_guide_calls_to_iteration(self, iteration: int) -> None:
        if not self._pending_iteration_guide_calls:
            return
        calls = list(self._pending_iteration_guide_calls)
        self._pending_iteration_guide_calls = []
        self._attach_guide_calls_to_iteration(
            iteration,
            calls,
            source="pending_evox_meta",
        )

    def _record_inner_search_budget(self, *, solution_iter: int) -> None:
        records = getattr(self.search_controller.budget_ledger, "records", [])
        new_records = records[self._copied_search_budget_records :]
        calls: List[CallCostRecord] = []
        for source_record in new_records:
            for call in source_record.calls:
                calls.append(
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
        self._copied_search_budget_records = len(records)
        self._attach_guide_calls_to_iteration(
            solution_iter,
            calls,
            source="evox_search_algorithm_generation",
        )

    async def _run_iteration(self, iteration: int, retry_times: int = 1) -> SerializableResult:
        result = await super()._run_iteration(iteration, retry_times=retry_times)
        self._flush_pending_guide_calls_to_iteration(iteration)
        return result

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
            self._queue_guide_response(response, source="evox_variation_operator_generation")
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
