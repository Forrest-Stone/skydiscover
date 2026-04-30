"""CostAda controller on top of the shared evolutionary search scaffold."""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from typing import Dict, List, Optional

from skydiscover.budget import CallRole
from skydiscover.context_builder.costada import CostAdaContextBuilder
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.costada.adaptation import (
    budget_mix,
    cost_denominator,
    cost_transform,
    global_gain,
    local_gain,
    normalized_cost,
    routing_reward,
    update_signal,
    utility,
)
from skydiscover.search.costada.router import CostAwareFrontierRouter
from skydiscover.search.costada.state import CompactControlState, FrontierState
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.utils.budget_iteration import BudgetIterationMixin
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.metrics import compute_proxy_score

logger = logging.getLogger(__name__)


class CostAdaController(BudgetIterationMixin, AdaEvolveController):
    """Budget-calibrated controller with search-side token-cost accounting."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.context_builder = CostAdaContextBuilder(self.config)

        db_cfg = self.config.search.database
        self.alpha = float(getattr(db_cfg, "costada_alpha", 0.9))
        self.router = CostAwareFrontierRouter(
            c_ucb=float(
                getattr(db_cfg, "costada_c_ucb", getattr(db_cfg, "costada_router_beta", 0.5))
            ),
            gamma=float(getattr(db_cfg, "costada_router_gamma", 0.9)),
        )
        self.intensity_min = float(getattr(db_cfg, "intensity_min", 0.15))
        self.intensity_max = float(getattr(db_cfg, "intensity_max", 0.5))
        self.intensity_eps = float(getattr(db_cfg, "costada_intensity_eps", 1e-8))
        if self.intensity_min > self.intensity_max:
            self.intensity_min, self.intensity_max = self.intensity_max, self.intensity_min

        self.eps_c = float(getattr(db_cfg, "costada_eps_c", 1e-8))
        self.ref_cost = self._resolve_ref_cost(db_cfg)
        self.meta_eta_min = float(getattr(db_cfg, "costada_eta_min", 0.15))
        self.meta_h_threshold = float(getattr(db_cfg, "costada_meta_h_threshold", 0.01))
        self.meta_gain_eps = float(getattr(db_cfg, "costada_significant_gain_eps", 1e-6))
        self.meta_stagnation_steps = int(getattr(db_cfg, "costada_stagnation_steps", 8))
        self.guide_min_cost = float(getattr(db_cfg, "costada_guide_min_cost", 2.0 * self.ref_cost))
        self.meta_window = int(getattr(db_cfg, "costada_meta_window", 8))
        self.improvement_window = int(getattr(db_cfg, "costada_improvement_window", 8))

        num_islands = int(getattr(self.database, "num_islands", 1) or 1)
        self.frontier_states: Dict[int, FrontierState] = {
            i: FrontierState(frontier_id=i) for i in range(num_islands)
        }
        self._recent_frontier_improvements: deque[float] = deque(maxlen=self.improvement_window)
        self._recent_H_values: deque[float] = deque(maxlen=self.meta_window)
        self._recent_utility_values: deque[float] = deque(maxlen=self.meta_window)
        self._current_prompt_budget_mode: str = "standard"
        self._current_explore: bool = False
        self._current_local_mode: str = "balanced"
        self._current_intensity: float = self.intensity_max
        self._current_frontier_id: int = 0
        self._active_budget_record = None
        self._current_call_role = CallRole.GENERATION
        self._pending_meta_intervention = False

    def _resolve_ref_cost(self, db_cfg) -> float:
        """Resolve the benchmark-family reference generation cost."""
        for key in (
            "costada_ref_cost",
            "ref_cost",
            "reference_cost",
            "median_standard_generation_cost",
        ):
            value = getattr(db_cfg, key, None)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0.0:
                return parsed

        nominal = max(
            float(getattr(db_cfg, "nominal_budget", 0.0) or self.budget_ledger.config.nominal_budget),
            self.budget_ledger.config.eps,
        )
        max_iters = max(int(getattr(self.config, "max_iterations", 1) or 1), 1)
        return max(nominal / max_iters, self.budget_ledger.config.eps)

    async def _call_llm(self, system_message: str, user_message: str, **kwargs):
        """Attach CostAda budget metadata while preserving AdaEvolve generation limits."""
        if "_budget_record" not in kwargs and self._active_budget_record is not None:
            kwargs["_budget_record"] = self._active_budget_record
        if "_call_role" not in kwargs:
            kwargs["_call_role"] = self._current_call_role
        return await super()._call_llm(system_message, user_message, **kwargs)

    def _candidate_score(self, result: SerializableResult) -> Optional[float]:
        child = result.child_program_dict or {}
        metrics = child.get("metrics") or {}
        if not metrics:
            return None
        return compute_proxy_score(
            metrics,
            fitness_key=getattr(self.config.search.database, "fitness_key", None),
            pareto_objectives=getattr(self.config.search.database, "pareto_objectives", []) or [],
            higher_is_better=getattr(self.config.search.database, "higher_is_better", {}) or {},
        )

    def _extract_candidate_score(self, result: SerializableResult) -> Optional[float]:
        """Keep budget rows aligned with CostAda's configured proxy score."""
        return self._candidate_score(result)

    def _recent_improvement_avg(self) -> float:
        if not self._recent_frontier_improvements:
            return 0.0
        return float(
            sum(self._recent_frontier_improvements) / len(self._recent_frontier_improvements)
        )

    def _avg_recent_H(self) -> float:
        if not self._recent_H_values:
            return 0.0
        return float(sum(self._recent_H_values) / len(self._recent_H_values))

    def _avg_recent_utility(self) -> float:
        if not self._recent_utility_values:
            return 0.0
        return float(sum(self._recent_utility_values) / len(self._recent_utility_values))

    def _low_recent_utility(self) -> bool:
        """Use realized recent utility, not smoothed H, for intervention gates."""
        if not self._recent_utility_values:
            return False
        return self._avg_recent_utility() <= self.meta_h_threshold

    def _guidance_available(self) -> bool:
        return bool(
            getattr(self.database, "use_paradigm_breakthrough", False)
            and getattr(self, "paradigm_generator", None) is not None
        )

    def _is_stagnant(self, frontier_state: FrontierState) -> bool:
        return (
            frontier_state.stagnation_steps >= self.meta_stagnation_steps
            and self._recent_improvement_avg() <= self.meta_gain_eps
        )

    def _affordable_guidance(self, remaining_ratio: float) -> bool:
        remaining_budget = max(
            0.0,
            float(self.budget_ledger.config.nominal_budget) * float(remaining_ratio),
        )
        return remaining_ratio > self.meta_eta_min and remaining_budget >= self.guide_min_cost

    def _remaining_ratio_including_active_record(self) -> float:
        active_cost = 0.0
        if self._active_budget_record is not None:
            active_cost = (
                float(self._active_budget_record.generation_cost)
                + float(self._active_budget_record.retry_cost)
                + float(self._active_budget_record.guide_cost)
            )
        nominal = max(float(self.budget_ledger.config.nominal_budget), self.budget_ledger.config.eps)
        return max(0.0, 1.0 - (float(self.budget_ledger.cumulative_cost) + active_cost) / nominal)

    def _select_frontier(self) -> int:
        self._sync_frontier_states()
        frontier_ids = list(self.frontier_states.keys())
        if not frontier_ids:
            return 0
        return self.router.select(frontier_ids)

    def _prompt_budget_mode(
        self,
        frontier_state: FrontierState,
        remaining_budget_ratio: float,
    ) -> str:
        """Keep prompt construction at AdaEvolve parity for current experiments."""
        _ = (frontier_state, remaining_budget_ratio)
        return "standard"

    def _select_local_control(
        self,
        frontier_state: FrontierState,
        remaining_budget_ratio: float,
    ) -> CompactControlState:
        compact = CompactControlState(
            remaining_budget_ratio=remaining_budget_ratio,
            recent_improvement_avg=self._recent_improvement_avg(),
            stagnation_steps=frontier_state.stagnation_steps,
            frontier_signal=frontier_state.H,
        )
        self._current_intensity = self._frontier_intensity(frontier_state.H)
        self._current_local_mode = self._sample_local_mode(self._current_intensity)
        self._current_explore = self._current_local_mode == "exploration"
        self._current_prompt_budget_mode = self._prompt_budget_mode(
            frontier_state, remaining_budget_ratio
        )
        return compact

    @staticmethod
    def _sample_local_mode(intensity: float) -> str:
        """Sample the local search mode from the intensity-conditioned policy."""
        p_explore = max(0.0, min(1.0, float(intensity)))
        rand = random.random()
        if rand < p_explore:
            return "exploration"
        if rand < p_explore + (1.0 - p_explore) * 0.7:
            return "exploitation"
        return "balanced"

    def _frontier_actual_best(self, frontier_id: int) -> Optional[float]:
        if not hasattr(self.database, "get_island_population"):
            return None
        try:
            population = self.database.get_island_population(frontier_id)
        except Exception:
            return None
        best_score: Optional[float] = None
        for program in population or []:
            try:
                score = float(self.database.get_program_proxy_score(program))
            except Exception:
                score = None
            if score is None:
                continue
            best_score = score if best_score is None else max(best_score, score)
        return best_score

    def _sync_frontier_states(self) -> None:
        num_islands = int(getattr(self.database, "num_islands", len(self.frontier_states) or 1) or 1)
        for fid in range(num_islands):
            state = self.frontier_states.setdefault(fid, FrontierState(frontier_id=fid))
            actual_best = self._frontier_actual_best(fid)
            if actual_best is None:
                continue
            if state.last_update_iteration < 0:
                state.local_best = float(actual_best)
            else:
                state.local_best = max(float(state.local_best), float(actual_best))

    def _frontier_intensity(self, frontier_signal: float) -> float:
        """Map frontier signal H to exploration intensity."""
        H = max(float(frontier_signal), 0.0)
        intensity = self.intensity_min + (self.intensity_max - self.intensity_min) / (
            1.0 + math.sqrt(H + self.intensity_eps)
        )
        return max(0.0, min(1.0, float(intensity)))

    def _budget_exhausted_before_iteration(self) -> bool:
        nominal = max(
            float(self.budget_ledger.config.nominal_budget),
            self.budget_ledger.config.eps,
        )
        return float(self.budget_ledger.cumulative_cost) >= nominal

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run CostAda until iteration cap or pre-step budget exhaustion.

        A step that starts under budget is allowed to finish even if it overshoots
        the nominal budget. The next step is blocked once the ledger is exhausted.
        """
        total = start_iteration + max_iterations
        logger.info(
            f"CostAda: Running up to {max_iterations} iterations "
            f"across {self.database.num_islands} frontiers"
        )

        self._setup_iteration_stats_logging()
        self._ensure_all_islands_seeded()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break
            if self._budget_exhausted_before_iteration():
                logger.info(
                    "CostAda budget exhausted before iteration %s: "
                    "cumulative=%.6f nominal=%.6f; stopping",
                    iteration,
                    float(self.budget_ledger.cumulative_cost),
                    float(self.budget_ledger.config.nominal_budget),
                )
                break

            try:
                await self._run_iteration(iteration, checkpoint_callback)
            except Exception as exc:
                logger.exception(f"Iteration {iteration} failed: {exc}")
            finally:
                self.database.end_iteration(iteration)

        logger.info("CostAda completed")
        self.database.log_status()

        if self._iteration_stats_log_path:
            logger.info(
                f"CostAda iteration stats saved to: {self._iteration_stats_log_path}"
            )

        self._write_budget_summary()
        return self.database.get_best_program()

    async def _run_normal_step(
        self,
        iteration: int,
        budget_record=None,
    ) -> SerializableResult:
        """Run one CostAda step with explicit call-role labeling for budget accounting."""
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)
        for attempt in range(attempts):
            self._budget_set_call_role(
                CallRole.GENERATION if attempt == 0 else CallRole.RETRY
            )
            result = await self._generate_child(
                iteration,
                error_context=last_error,
                budget_record=budget_record,
            )
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
        """Execute one CostAda iteration with budget-calibrated signal updates."""
        iteration_start_time = time.time()
        budget_record = self._budget_start_iteration(iteration)
        budget_record.meta.pop("tier", None)
        result: Optional[SerializableResult] = None
        finalized = False

        try:
            frontier_id = self._select_frontier()
            self._current_frontier_id = frontier_id
            frontier_state = self.frontier_states[frontier_id]
            self.database.current_island = frontier_id

            remaining_before = self.budget_ledger.remaining_ratio()
            compact_state = self._select_local_control(frontier_state, remaining_before)

            global_best_prev = self._best_score_or_zero()
            local_best_prev = float(frontier_state.local_best)
            budget_record.meta["frontier_id"] = frontier_id
            budget_record.meta["selected_frontier"] = frontier_id
            budget_record.meta["global_best_before"] = global_best_prev
            budget_record.meta["prompt_budget_mode"] = self._current_prompt_budget_mode
            budget_record.meta["remaining_budget_ratio"] = compact_state.remaining_budget_ratio
            budget_record.meta["remaining_budget_ratio_before"] = remaining_before
            budget_record.meta["intensity"] = float(self._current_intensity)
            budget_record.meta["explore"] = bool(self._current_explore)
            budget_record.meta["local_search_mode"] = self._current_local_mode
            budget_record.meta["explore_or_exploit"] = self._current_local_mode
            budget_record.meta["ref_cost"] = float(self.ref_cost)

            if (
                self._guidance_available()
                and self._pending_meta_intervention
                and self._affordable_guidance(remaining_before)
            ):
                await self._budget_generate_paradigms_if_needed(budget_record)
                self._pending_meta_intervention = False
            elif self._pending_meta_intervention and not self._guidance_available():
                budget_record.meta["guide_skipped_reason"] = "guidance_disabled"
                self._pending_meta_intervention = False
            elif self._pending_meta_intervention and not self._affordable_guidance(remaining_before):
                budget_record.meta["guide_skipped_reason"] = "not_affordable"
                self._pending_meta_intervention = False

            result = await self._run_normal_step(iteration, budget_record=budget_record)
            iteration_time = time.time() - iteration_start_time

            if result.error:
                logger.warning(f"Iteration {iteration}: {result.error}")
            else:
                self._process_result(result, iteration, checkpoint_callback)

            # Compute utility stats before finalize writes the iteration row.
            step_cost = (
                float(budget_record.generation_cost)
                + float(budget_record.retry_cost)
                + float(budget_record.guide_cost)
            )
            score_new = self._candidate_score(result)
            if score_new is None:
                d_local = 0.0
                g_global = 0.0
                frontier_improvement = 0.0
                global_best_after = global_best_prev
            else:
                d_local = local_gain(score_new, local_best_prev)
                g_global = global_gain(score_new, global_best_prev)
                frontier_improvement = max(float(score_new) - float(global_best_prev), 0.0)
                global_best_after = max(float(global_best_prev), float(score_new))
                budget_record.meta["candidate_score"] = float(score_new)
            nominal_budget = max(float(self.budget_ledger.config.nominal_budget), self.budget_ledger.config.eps)
            cumulative_after = float(self.budget_ledger.cumulative_cost) + step_cost
            remaining_after = max(
                0.0,
                1.0 - cumulative_after / nominal_budget,
            )
            ctilde = normalized_cost(step_cost, self.ref_cost, eps_c=self.eps_c)
            phi_t = cost_transform(ctilde)
            denom = cost_denominator(remaining_after, ctilde)
            lambda_t = budget_mix(remaining_after)
            util = utility(
                d_local,
                g_global,
                step_cost,
                remaining_after,
                self.ref_cost,
                eps_c=self.eps_c,
            )
            frontier_state.H = update_signal(frontier_state.H, util, alpha=self.alpha)
            if score_new is not None:
                frontier_state.local_best = max(frontier_state.local_best, float(score_new))
            frontier_state.last_update_iteration = iteration

            if frontier_improvement > self.meta_gain_eps:
                frontier_state.stagnation_steps = 0
            else:
                frontier_state.stagnation_steps += 1

            realized_router_reward = routing_reward(g_global, denom)
            self.router.update(frontier_id=frontier_id, routing_reward_value=realized_router_reward)
            frontier_state.routing_reward = self.router.get_reward(frontier_id)
            frontier_state.selection_count += 1

            self._recent_frontier_improvements.append(frontier_improvement)
            self._recent_H_values.append(frontier_state.H)
            self._recent_utility_values.append(util)

            stagnant = self._is_stagnant(frontier_state)
            low_recent_utility = self._low_recent_utility()
            affordable = self._affordable_guidance(remaining_after)
            guidance_available = self._guidance_available()
            meta_triggered = (
                stagnant
                and low_recent_utility
                and affordable
                and guidance_available
            )
            if meta_triggered:
                self._pending_meta_intervention = True

            budget_record.meta["step_cost"] = float(step_cost)
            budget_record.meta["cumulative_cost_after_step"] = float(cumulative_after)
            budget_record.meta["remaining_budget_ratio_after"] = float(remaining_after)
            budget_record.meta["lambda_t"] = float(lambda_t)
            budget_record.meta["ref_cost"] = float(self.ref_cost)
            budget_record.meta["normalized_cost"] = float(ctilde)
            budget_record.meta["ctilde"] = float(ctilde)
            budget_record.meta["cost_phi"] = float(phi_t)
            budget_record.meta["cost_denominator"] = float(denom)
            budget_record.meta["recent_improvement_avg"] = float(self._recent_improvement_avg())
            budget_record.meta["recent_H_avg"] = float(self._avg_recent_H())
            budget_record.meta["recent_utility_avg"] = float(self._avg_recent_utility())
            budget_record.meta["stagnation_steps"] = int(frontier_state.stagnation_steps)
            budget_record.meta["local_gain"] = float(d_local)
            budget_record.meta["global_gain"] = float(g_global)
            budget_record.meta["local_gain_normalized"] = float(d_local)
            budget_record.meta["global_gain_normalized"] = float(g_global)
            budget_record.meta["frontier_improvement"] = float(frontier_improvement)
            budget_record.meta["global_best_after"] = float(global_best_after)
            budget_record.meta["local_best"] = float(frontier_state.local_best)
            budget_record.meta["utility"] = float(util)
            budget_record.meta["frontier_signal"] = float(frontier_state.H)
            budget_record.meta["routing_reward"] = float(realized_router_reward)
            budget_record.meta["router_reward"] = float(realized_router_reward)  # backward-compatible alias
            budget_record.meta["routing_stat"] = float(frontier_state.routing_reward)
            budget_record.meta["stagnant"] = bool(stagnant)
            budget_record.meta["low_recent_utility"] = bool(low_recent_utility)
            budget_record.meta["affordable_guidance"] = bool(affordable)
            budget_record.meta["guidance_available"] = bool(guidance_available)
            budget_record.meta["guidance_scheduled"] = bool(meta_triggered)
            budget_record.meta["meta_triggered"] = bool(meta_triggered)
            budget_record.meta["prompt_component_cost_composition"] = {
                "generation": float(budget_record.generation_cost),
                "retry": float(budget_record.retry_cost),
                "guide": float(budget_record.guide_cost),
            }

            # Keep default summary/trace pipeline from phase-1.
            self._budget_finalize_iteration(budget_record, result)
            finalized = True

            # Keep AdaEvolve iteration stats logging behavior.
            if result.error:
                self._log_iteration_stats(
                    iteration=iteration,
                    sampling_mode=getattr(self, "_last_sampling_mode", None),
                    sampling_intensity=getattr(self, "_last_sampling_intensity", None),
                    child_program=None,
                    iteration_time=iteration_time,
                    llm_generation_time=result.llm_generation_time,
                    eval_time=result.eval_time,
                    error=result.error,
                )
            else:
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
        except Exception as exc:
            # Persist budget/cost trace even when iteration orchestration fails mid-way.
            result = self._budget_error_result(
                iteration,
                iteration_start_time,
                f"CostAda iteration error: {exc}",
            )
        finally:
            if not finalized and result is not None:
                try:
                    self._budget_finalize_iteration(budget_record, result)
                except Exception:
                    pass
            self._budget_reset_context()

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
        budget_record=None,
    ) -> SerializableResult:
        """Generate/evaluate a child with CostAda sampling and AdaEvolve prompt parity."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration, budget_record=budget_record)

            self._ensure_all_islands_seeded()

            local_mode = "exploration" if force_exploration else self._current_local_mode
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
                island_id=self._current_frontier_id,
                intensity=self._current_intensity,
                local_mode=local_mode,
                prompt_budget_mode=self._current_prompt_budget_mode,
            )

            if not parent_dict:
                return SerializableResult(error="Empty parent dict from sample()", iteration=iteration)

            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]
            sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"
            self._last_sampling_mode = sampling_mode

            self._last_sampling_intensity = self._current_intensity

            paradigm = (
                self.database.get_current_paradigm()
                if self.database.use_paradigm_breakthrough
                else None
            )
            if paradigm:
                best_program = self.database.get_best_program()
                if best_program:
                    parent_dict = {parent_label: best_program}
                    parent = best_program

            siblings: List[Program] = []
            if hasattr(self.database, "get_children"):
                try:
                    siblings = self.database.get_children(parent.id)
                except (AttributeError, NotImplementedError):
                    pass

            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                "paradigm": paradigm,
                "siblings": siblings,
                "error_context": error_context,
                "remaining_budget_ratio": self._remaining_ratio_including_active_record(),
                "prompt_budget_mode": self._current_prompt_budget_mode,
                "costada_explore": self._current_explore,
                "costada_local_mode": local_mode,
                "costada_intensity": self._current_intensity,
            }
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            prompt = self.context_builder.build_prompt(parent_dict, context)

            if paradigm:
                self.database.use_paradigm()

            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            return await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
            )
        except Exception as e:
            logger.exception(f"CostAda generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)
