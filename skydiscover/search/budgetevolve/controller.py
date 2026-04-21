"""BudgetEvolve controller.

Implements a budget-aware layer on top of AdaEvolve without modifying AdaEvolve's
original code.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from skydiscover.budget import CallRole, call_record_from_response
from skydiscover.context_builder.budgetevolve import BudgetEvolveContextBuilder
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.adaevolve.paradigm import ParadigmGenerator as BaseParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import apply_diff, extract_diffs, format_diff_summary, parse_full_rewrite

from .adaptation import (
    BudgetAction,
    BudgetActionScheduler,
    BudgetLedger as TokenBudgetLedger,
    BudgetState,
    BudgetStateBuilder,
    UsageRecord,
)

logger = logging.getLogger(__name__)


class BudgetEvolveController(AdaEvolveController):
    """AdaEvolve + budget state/action loop."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.context_builder = BudgetEvolveContextBuilder(self.config)

        db_cfg = self.config.search.database
        self.budget_enabled = bool(getattr(db_cfg, "budget_enabled", False))
        # Keep shared USD-based budget ledger from DiscoveryController untouched.
        # BudgetEvolve keeps an additional token-budget ledger for method-local control.
        self.token_budget_ledger = TokenBudgetLedger(
            total_budget=getattr(db_cfg, "token_budget_total", 500000),
            strict_stop=getattr(db_cfg, "budget_strict_stop", True),
            cost_budget_total=getattr(db_cfg, "cost_budget_total", 0.0),
            input_token_cost=getattr(db_cfg, "input_token_cost", 0.0),
            output_token_cost=getattr(db_cfg, "output_token_cost", 0.0),
        )
        self.budget_state_builder = BudgetStateBuilder(self.config)
        self.budget_scheduler = BudgetActionScheduler(self.config)

        self._recent_frontier_gains: List[float] = []
        self._no_improve_steps = 0
        self._current_budget_state: Optional[BudgetState] = None
        self._current_budget_action: Optional[BudgetAction] = None
        self._current_usage: Optional[UsageRecord] = None
        self._current_frontier_gain: float = 0.0

        # Use budget-aware paradigm wrapper if available.
        if self.paradigm_generator is not None and isinstance(self.paradigm_generator, BaseParadigmGenerator):
            from .paradigm import ParadigmGenerator

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_cfg, "pareto_objectives", []),
                higher_is_better=getattr(db_cfg, "higher_is_better", {}),
                fitness_key=getattr(db_cfg, "fitness_key", None),
            )

    async def _call_llm(self, system_message: str, user_message: str, **kwargs):
        """Keep shared budget logging and additionally track token-budget spend."""
        call_role = kwargs.get("_call_role")
        result = await super()._call_llm(system_message, user_message, **kwargs)
        if self.budget_enabled and call_role in (CallRole.GENERATION, CallRole.RETRY):
            in_tokens = int(getattr(result, "input_tokens", 0) or 0)
            out_tokens = int(getattr(result, "output_tokens", 0) or 0)
            if in_tokens > 0 or out_tokens > 0:
                self.token_budget_ledger.update(
                    UsageRecord(
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=in_tokens + out_tokens,
                        raw_usage=getattr(result, "raw_usage", None),
                    )
                )
        return result

    def _estimate_prompt_tokens(self, prompt: Dict[str, str]) -> int:
        return max(1, int((prompt.get("system", "") + prompt.get("user", "")).__len__() / 4))

    def _estimate_text_tokens(self, text: str) -> int:
        return max(1, int(len(text or "") / 4))

    def _budget_exhausted(self) -> bool:
        if not self.budget_enabled:
            return False
        if self.token_budget_ledger.remaining_tokens <= 0:
            return True
        if (
            self.token_budget_ledger.cost_budget_total > 0
            and self.token_budget_ledger.remaining_cost <= 0
        ):
            return True
        return False

    def _recent_gain_ma(self) -> float:
        if not self._recent_frontier_gains:
            return 0.0
        n = min(10, len(self._recent_frontier_gains))
        return sum(self._recent_frontier_gains[-n:]) / n

    def _burn_rate(self, iteration: int) -> float:
        return float(self.token_budget_ledger.spent_tokens) / max(1, iteration)

    def _downgrade_action(self, action: BudgetAction) -> BudgetAction:
        db = self.config.search.database
        if action.tier == "rich":
            return BudgetAction(action.family, "standard", db.standard_max_output_tokens)
        if action.tier == "standard":
            return BudgetAction(action.family, "cheap", db.cheap_max_output_tokens)
        return action

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        total = start_iteration + max_iterations
        self._setup_iteration_stats_logging()
        self._ensure_all_islands_seeded()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set() or self._budget_exhausted():
                break
            try:
                await self._run_iteration(iteration, checkpoint_callback)
            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")
            finally:
                self.database.end_iteration(iteration)

        self.database.log_status()
        self._write_budget_summary()
        return self.database.get_best_program()

    async def _generate_paradigms_if_needed(self, budget_record=None) -> None:
        if not self.budget_enabled:
            return await super()._generate_paradigms_if_needed(budget_record=budget_record)
        if self.paradigm_generator is None:
            return

        tracker = getattr(self.database, "paradigm_tracker", None)
        if tracker is None or self.database.has_active_paradigm():
            return
        if not tracker.should_trigger_budgeted(
            remaining_ratio=self.token_budget_ledger.remaining_ratio,
            meta_budget_threshold=getattr(self.config.search.database, "budget_meta_threshold", 0.25),
            estimated_meta_cost=2048,
            remaining_tokens=self.token_budget_ledger.remaining_tokens,
            recent_meta_success_rate=None,
        ):
            return

        best_program = self.database.get_best_program()
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_program.solution if best_program else "",
            current_best_score=self.database.get_program_proxy_score(best_program),
            previously_tried_ideas=self.database.get_previously_tried_ideas(),
            evaluator_feedback=(best_program.artifacts or {}).get("feedback") if best_program else None,
            mode="summary",
            llm_response_callback=lambda response: self._on_guide_response(response, budget_record),
        )
        if paradigms:
            self.database.set_paradigms(paradigms)

    def _on_guide_response(self, response, budget_record) -> None:
        """Record guide-call cost in shared ledger and token ledger."""
        if budget_record is not None:
            self.budget_ledger.add_call(
                budget_record,
                call_record_from_response(response, CallRole.GUIDE),
            )
        in_tokens = int(getattr(response, "input_tokens", 0) or 0)
        out_tokens = int(getattr(response, "output_tokens", 0) or 0)
        if in_tokens > 0 or out_tokens > 0:
            self.token_budget_ledger.update(
                UsageRecord(
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    total_tokens=in_tokens + out_tokens,
                    raw_usage=getattr(response, "raw_usage", None),
                )
            )

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
        call_role=CallRole.GENERATION,
        budget_record=None,
    ) -> SerializableResult:
        if not self.budget_enabled:
            return await super()._generate_child(
                iteration=iteration,
                error_context=error_context,
                force_exploration=force_exploration,
                call_role=call_role,
                budget_record=budget_record,
            )

        if not self.database.programs:
            return await self._run_from_scratch_iteration(iteration, budget_record=budget_record)

        island_id = self.database.current_island
        intensity = self.database.adapter.get_search_intensity(island_id) if self.database.use_adaptive_search else self.database.fixed_intensity
        state = self.budget_state_builder.build(
            island_id=island_id,
            intensity=intensity,
            ledger=self.token_budget_ledger,
            recent_gain_ma=self._recent_gain_ma(),
            no_improve_steps=self._no_improve_steps,
            burn_rate=self._burn_rate(iteration),
        )
        action = self.budget_scheduler.select(state)
        self._current_budget_state = state
        self._current_budget_action = action

        parent_dict, context_programs_dict = self.database.sample(
            self.num_context_programs,
            force_exploration=force_exploration,
            island_id=island_id,
            intensity=intensity,
            family=action.family,
            tier=action.tier,
            budget_bin=state.budget_bin,
        )

        parent_label = list(parent_dict.keys())[0]
        parent = list(parent_dict.values())[0]
        context = {
            "program_metrics": parent.metrics,
            "other_context_programs": context_programs_dict,
            "error_context": error_context,
            "budget_enabled": True,
            "spent_tokens": self.token_budget_ledger.spent_tokens,
            "spent_cost": self.token_budget_ledger.spent_cost,
            "remaining_budget_ratio": state.remaining_ratio,
            "budget_bin": state.budget_bin,
            "budget_action_family": action.family,
            "budget_action_tier": action.tier,
        }
        prompt = self.context_builder.build_prompt(parent_dict, context)

        est_in = self._estimate_prompt_tokens(prompt)
        if not self.token_budget_ledger.feasible(est_in, action.max_output_tokens):
            action = self._downgrade_action(action)
            context["budget_action_tier"] = action.tier
            prompt = self.context_builder.build_prompt(parent_dict, context)
            if not self.token_budget_ledger.feasible(
                self._estimate_prompt_tokens(prompt), action.max_output_tokens
            ):
                return SerializableResult(error="Budget exhausted before generation", iteration=iteration)

        llm_start = time.time()
        llm_result = await self._call_llm(
            prompt["system"],
            prompt["user"],
            max_tokens=action.max_output_tokens,
            _budget_record=budget_record,
            _call_role=call_role,
        )
        llm_generation_time = time.time() - llm_start
        response = llm_result.text or ""
        if not response:
            return SerializableResult(error="Empty LLM response", iteration=iteration)

        in_tokens = llm_result.input_tokens or self._estimate_prompt_tokens(prompt)
        out_tokens = llm_result.output_tokens or self._estimate_text_tokens(response)
        self._current_usage = UsageRecord(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            raw_usage=llm_result.raw_usage,
        )

        if self.config.diff_based_generation:
            diffs = extract_diffs(response)
            child_solution = apply_diff(parent.solution, response) if diffs else parse_full_rewrite(response, self.config.language)
            changes = format_diff_summary(diffs) if diffs else "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(error="No valid solution in response", iteration=iteration)

        child_id = str(uuid.uuid4())
        eval_start = time.time()
        eval_result = await self.evaluator.evaluate_program(child_solution, child_id)
        eval_time = time.time() - eval_start

        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=eval_result.metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metadata={"changes": changes, "parent_metrics": parent.metrics},
            artifacts=eval_result.artifacts,
        )

        new_score = self.database.get_program_proxy_score(child)
        prev_best = self.database.get_program_proxy_score(self.database.get_best_program())
        frontier_gain = max(0.0, float(new_score) - float(prev_best))
        self._current_frontier_gain = frontier_gain
        self._recent_frontier_gains.append(frontier_gain)
        self._recent_frontier_gains = self._recent_frontier_gains[-10:]
        self._no_improve_steps = 0 if frontier_gain > 0 else self._no_improve_steps + 1

        child_dict = child.to_dict()
        child_dict.setdefault("metadata", {})
        child_dict["metadata"]["budget"] = {
            "input_tokens": self._current_usage.input_tokens,
            "output_tokens": self._current_usage.output_tokens,
            "total_tokens": self._current_usage.total_tokens,
            "spent_tokens_after": self.token_budget_ledger.spent_tokens,
            "spent_cost_after": self.token_budget_ledger.spent_cost,
            "remaining_ratio_after": self.token_budget_ledger.remaining_ratio,
            "action_family": action.family,
            "action_tier": action.tier,
            "frontier_gain": frontier_gain,
            "roi": frontier_gain / max(1, self._current_usage.total_tokens),
        }
        if budget_record is not None:
            budget_record.meta["budget_bin"] = state.budget_bin
            budget_record.meta["action_family"] = action.family
            budget_record.meta["action_tier"] = action.tier
            budget_record.meta["frontier_gain"] = frontier_gain
            budget_record.meta["remaining_budget_ratio"] = state.remaining_ratio

        return SerializableResult(
            child_program_dict=child_dict,
            parent_id=parent.id,
            iteration_time=time.time() - llm_start,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            iteration=iteration,
        )

    def _process_result(self, result: SerializableResult, iteration: int, checkpoint_callback) -> None:
        super()._process_result(result, iteration, checkpoint_callback)
        if self._current_budget_state and self._current_budget_action and self._current_usage:
            self.budget_scheduler.update(
                state=self._current_budget_state,
                action=self._current_budget_action,
                frontier_gain=self._current_frontier_gain,
                cost=self._current_usage.total_tokens,
            )
