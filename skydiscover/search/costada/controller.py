"""CostAda controller implementing BCHD on top of AdaEvolve scaffold."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional

from skydiscover.budget import CallRole
from skydiscover.context_builder.costada import CostAdaContextBuilder
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.costada.adaptation import (
    global_gain,
    local_gain,
    update_signal,
    utility,
)
from skydiscover.search.costada.router import CostAwareFrontierRouter
from skydiscover.search.costada.state import CompactControlState, FrontierState
from skydiscover.search.costada.tier_scheduler import TierScheduler
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.metrics import compute_proxy_score

logger = logging.getLogger(__name__)


class CostAdaController(AdaEvolveController):
    """Budget-calibrated hierarchical controller (BCHD / CostAda).

    Final design split:
    - step-level spending: deterministic local adaptation from H_t
    - frontier-level allocation: UCB/bandit in CostAwareFrontierRouter
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.context_builder = CostAdaContextBuilder(self.config)

        db_cfg = self.config.search.database
        self.alpha = float(getattr(db_cfg, "costada_alpha", 0.9))
        self.router = CostAwareFrontierRouter(
            beta=float(getattr(db_cfg, "costada_router_beta", 0.5)),
            gamma=float(getattr(db_cfg, "costada_router_gamma", 0.9)),
        )
        self.tier_scheduler = TierScheduler(
            intensity_min=float(getattr(db_cfg, "intensity_min", 0.15)),
            intensity_max=float(getattr(db_cfg, "intensity_max", 0.5)),
            tau_1=float(getattr(db_cfg, "costada_tier_tau_1", 0.24)),
            tau_2=float(getattr(db_cfg, "costada_tier_tau_2", 0.38)),
            eta_low=float(getattr(db_cfg, "costada_eta_low", 0.12)),
            rich_enable_min_budget=float(getattr(db_cfg, "costada_rich_enable_min_budget", 0.28)),
            stagnation_threshold=int(getattr(db_cfg, "costada_stagnation_steps", 8)),
            low_signal_threshold=float(getattr(db_cfg, "costada_meta_h_threshold", 0.01)),
        )
        self.meta_eta_min = float(getattr(db_cfg, "costada_eta_min", 0.15))
        self.meta_h_threshold = float(getattr(db_cfg, "costada_meta_h_threshold", 0.01))
        self.meta_gain_eps = float(getattr(db_cfg, "costada_significant_gain_eps", 1e-6))
        self.meta_stagnation_steps = int(getattr(db_cfg, "costada_stagnation_steps", 8))
        self.meta_window = int(getattr(db_cfg, "costada_meta_window", 8))
        self.improvement_window = int(getattr(db_cfg, "costada_improvement_window", 8))

        num_islands = int(getattr(self.database, "num_islands", 1) or 1)
        self.frontier_states: Dict[int, FrontierState] = {
            i: FrontierState(frontier_id=i) for i in range(num_islands)
        }
        self._recent_global_gains: deque[float] = deque(maxlen=self.improvement_window)
        self._recent_H_values: deque[float] = deque(maxlen=self.meta_window)
        self._current_tier: str = "standard"
        self._current_frontier_id: int = 0
        self._active_budget_record = None
        self._active_call_role = CallRole.GENERATION

    async def _call_llm(self, system_message: str, user_message: str, **kwargs):
        """Inject tier-specific output-token caps while preserving shared budget logging."""
        if "max_tokens" not in kwargs and not kwargs.get("image_output"):
            if self._current_tier == "cheap":
                kwargs["max_tokens"] = int(
                    getattr(self.config.search.database, "cheap_max_output_tokens", 512)
                )
            elif self._current_tier == "rich":
                kwargs["max_tokens"] = int(
                    getattr(self.config.search.database, "rich_max_output_tokens", 2048)
                )
            else:
                kwargs["max_tokens"] = int(
                    getattr(self.config.search.database, "standard_max_output_tokens", 1024)
                )
        if "_budget_record" not in kwargs and self._active_budget_record is not None:
            kwargs["_budget_record"] = self._active_budget_record
        if "_call_role" not in kwargs:
            kwargs["_call_role"] = self._active_call_role
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

    def _recent_improvement_avg(self) -> float:
        if not self._recent_global_gains:
            return 0.0
        return float(sum(self._recent_global_gains) / len(self._recent_global_gains))

    def _avg_recent_H(self) -> float:
        if not self._recent_H_values:
            return 0.0
        return float(sum(self._recent_H_values) / len(self._recent_H_values))

    def _is_stagnant(self, frontier_state: FrontierState) -> bool:
        return (
            frontier_state.stagnation_steps >= self.meta_stagnation_steps
            and self._recent_improvement_avg() <= self.meta_gain_eps
        )

    def _select_frontier(self) -> int:
        frontier_ids = list(self.frontier_states.keys())
        if not frontier_ids:
            return 0
        return self.router.select(frontier_ids)

    def _select_tier(self, frontier_state: FrontierState) -> CompactControlState:
        compact = CompactControlState(
            remaining_budget_ratio=self.budget_ledger.remaining_ratio(),
            recent_improvement_avg=self._recent_improvement_avg(),
            stagnation_steps=frontier_state.stagnation_steps,
            frontier_signal=frontier_state.H,
        )
        self._current_tier = self.tier_scheduler.select(compact)
        return compact

    def _frontier_intensity(self, frontier_signal: float) -> float:
        """Map frontier signal H to exploration intensity via deterministic scheduler."""
        return self.tier_scheduler.compute_intensity(frontier_signal)

    async def run_discovery(self, *args, **kwargs):
        out = await super().run_discovery(*args, **kwargs)
        self._write_budget_summary()
        return out

    async def _run_normal_step(
        self,
        iteration: int,
        budget_record=None,
    ) -> SerializableResult:
        """Run one CostAda step with explicit call-role labeling for budget accounting."""
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)
        for attempt in range(attempts):
            self._active_call_role = CallRole.GENERATION if attempt == 0 else CallRole.RETRY
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
        """Execute one CostAda iteration with unified BCHD signal updates."""
        iteration_start_time = time.time()
        budget_record = self.budget_ledger.start_iteration(iteration)
        self._active_budget_record = budget_record
        result: SerializableResult | None = None
        finalized = False

        try:
            frontier_id = self._select_frontier()
            self._current_frontier_id = frontier_id
            frontier_state = self.frontier_states[frontier_id]
            self.database.current_island = frontier_id

            compact_state = self._select_tier(frontier_state)

            global_best_prev = self._best_score_or_zero()
            budget_record.meta["frontier_id"] = frontier_id
            budget_record.meta["global_best_before"] = global_best_prev
            budget_record.meta["tier"] = self._current_tier
            budget_record.meta["remaining_budget_ratio"] = compact_state.remaining_budget_ratio

            # Optional paradigm generation follows AdaEvolve scaffold.
            if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
                await self._generate_paradigms_if_needed()

            result = await self._run_normal_step(iteration, budget_record=budget_record)
            iteration_time = time.time() - iteration_start_time

            if result.error:
                logger.warning(f"Iteration {iteration}: {result.error}")
            else:
                self._process_result(result, iteration, checkpoint_callback)

            # Compute utility stats before finalize writes the iteration row.
            raw_iteration_cost = (
                float(budget_record.generation_cost)
                + float(budget_record.retry_cost)
                + float(budget_record.guide_cost)
            )
            score_new = self._candidate_score(result)
            if score_new is None:
                score_new = global_best_prev

            d_local = local_gain(score_new, frontier_state.local_best)
            g_global = global_gain(score_new, global_best_prev)
            util = utility(d_local, g_global, raw_iteration_cost, compact_state.remaining_budget_ratio)
            frontier_state.H = update_signal(frontier_state.H, util, alpha=self.alpha)
            frontier_state.local_best = max(frontier_state.local_best, float(score_new))
            frontier_state.last_update_iteration = iteration

            if g_global > self.meta_gain_eps:
                frontier_state.stagnation_steps = 0
            else:
                frontier_state.stagnation_steps += 1

            realized_router_reward = self.router.update(
                frontier_id=frontier_id,
                global_gain_value=g_global,
                raw_iteration_cost=raw_iteration_cost,
            )
            frontier_state.routing_reward = self.router.get_reward(frontier_id)
            frontier_state.selection_count += 1

            self._recent_global_gains.append(g_global)
            self._recent_H_values.append(frontier_state.H)
            self.tier_scheduler.update(compact_state, self._current_tier, util)

            meta_triggered = (
                self._is_stagnant(frontier_state)
                and compact_state.remaining_budget_ratio > self.meta_eta_min
                and self._avg_recent_H() < self.meta_h_threshold
            )

            budget_record.meta["recent_improvement_avg"] = float(self._recent_improvement_avg())
            budget_record.meta["stagnation_steps"] = int(frontier_state.stagnation_steps)
            budget_record.meta["local_gain"] = float(d_local)
            budget_record.meta["global_gain"] = float(g_global)
            budget_record.meta["utility"] = float(util)
            budget_record.meta["frontier_signal"] = float(frontier_state.H)
            budget_record.meta["routing_reward"] = float(realized_router_reward)
            budget_record.meta["router_reward"] = float(realized_router_reward)  # backward-compatible alias
            budget_record.meta["meta_triggered"] = bool(meta_triggered)

            # Keep default summary/trace pipeline from phase-1.
            self._finalize_budget_iteration(budget_record, result)
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
            result = SerializableResult(
                error=f"CostAda iteration error: {exc}",
                iteration=iteration,
                attempts_used=1,
                iteration_time=(time.time() - iteration_start_time),
            )
        finally:
            if not finalized and result is not None:
                try:
                    self._finalize_budget_iteration(budget_record, result)
                except Exception:
                    pass
            self._active_budget_record = None
            self._active_call_role = CallRole.GENERATION

            budget_record.meta["recent_improvement_avg"] = float(self._recent_improvement_avg())
            budget_record.meta["stagnation_steps"] = int(frontier_state.stagnation_steps)
            budget_record.meta["local_gain"] = float(d_local)
            budget_record.meta["global_gain"] = float(g_global)
            budget_record.meta["utility"] = float(util)
            budget_record.meta["frontier_signal"] = float(frontier_state.H)
            budget_record.meta["routing_reward"] = float(realized_router_reward)
            budget_record.meta["router_reward"] = float(realized_router_reward)  # backward-compatible alias
            budget_record.meta["meta_triggered"] = bool(meta_triggered)

            # Keep default summary/trace pipeline from phase-1.
            self._finalize_budget_iteration(budget_record, result)

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
        finally:
            self._active_budget_record = None
            self._active_call_role = CallRole.GENERATION

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
        budget_record=None,
    ) -> SerializableResult:
        """Generate/evaluate a child with tier-aware sampling + prompt context."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration, budget_record=budget_record)

            self._ensure_all_islands_seeded()

            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
                island_id=self._current_frontier_id,
                intensity=self._frontier_intensity(
                    self.frontier_states[self._current_frontier_id].H
                ),
                tier=self._current_tier,
            )

            if not parent_dict:
                return SerializableResult(error="Empty parent dict from sample()", iteration=iteration)

            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]
            sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"
            self._last_sampling_mode = sampling_mode

            self._last_sampling_intensity = self._frontier_intensity(
                self.frontier_states[self._current_frontier_id].H
            )

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
                "remaining_budget_ratio": self.budget_ledger.remaining_ratio(),
                "costada_tier": self._current_tier,
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
