"""CostAda database: AdaEvolve archive with CostAda sampling controls."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)


class CostAdaDatabase(AdaEvolveDatabase):
    """AdaEvolve DB with H-intensity local sampling and prompt budget gates."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self._costada_feedback_chars: Optional[int] = None

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
        island_id: Optional[int] = None,
        intensity: Optional[float] = None,
        explore: Optional[bool] = None,
        local_mode: Optional[str] = None,
        family: Optional[str] = None,
        tier: Optional[str] = None,
        prompt_budget_mode: Optional[str] = None,
        budget_bin: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """Sample parent/context from the selected frontier.

        CostAda samples the local mode from the intensity-conditioned policy:
        exploration gets probability ``intensity``; the remaining mass is split
        between exploitation and balanced sampling.
        """
        _ = (family, budget_bin, kwargs)
        island_idx = self.current_island if island_id is None else int(island_id)
        self.current_island = island_idx

        prompt_mode = self._resolve_prompt_budget_mode(prompt_budget_mode, tier)
        num = self._context_count_for_prompt_mode(int(num_context_programs or 4), prompt_mode)
        self._set_feedback_limit(prompt_mode)

        mode = self._resolve_local_mode(local_mode)
        if mode is None and explore is not None:
            mode = "exploration" if explore else "exploitation"

        if mode is None:
            if force_exploration:
                mode = "exploration"
            elif intensity is not None:
                mode = self._sample_mode_from_intensity(float(intensity))
            else:
                return super().sample(
                    num_context_programs=num,
                    force_exploration=force_exploration,
                )
        elif force_exploration:
            mode = "exploration"

        if self.use_unified_archive and self.archives:
            return self._sample_from_archive_costada(island_idx, num, mode)
        return self._sample_legacy_costada(island_idx, num, mode)

    @staticmethod
    def _resolve_local_mode(local_mode: Optional[str]) -> Optional[str]:
        if local_mode is None:
            return None
        mode = str(local_mode).strip().lower()
        aliases = {
            "explore": "exploration",
            "exploit": "exploitation",
            "refine": "exploitation",
        }
        mode = aliases.get(mode, mode)
        if mode in {"exploration", "exploitation", "balanced"}:
            return mode
        return None

    @staticmethod
    def _sample_mode_from_intensity(intensity: float) -> str:
        p_explore = max(0.0, min(1.0, float(intensity)))
        rand = random.random()
        if rand < p_explore:
            return "exploration"
        if rand < p_explore + (1.0 - p_explore) * 0.7:
            return "exploitation"
        return "balanced"

    @staticmethod
    def _resolve_prompt_budget_mode(prompt_budget_mode: Optional[str], tier: Optional[str]) -> str:
        mode = (prompt_budget_mode or tier or "standard").strip().lower()
        if mode == "cheap":
            mode = "lean"
        if mode not in {"lean", "standard", "rich"}:
            mode = "standard"
        return mode

    def _context_count_for_prompt_mode(self, requested: int, prompt_mode: str) -> int:
        if prompt_mode == "lean":
            return min(requested, int(getattr(self.config, "costada_lean_context_programs", 2)))
        if prompt_mode == "rich":
            return max(requested, int(getattr(self.config, "costada_rich_context_programs", 6)))
        return requested

    def _set_feedback_limit(self, prompt_mode: str) -> None:
        if prompt_mode == "lean":
            self._costada_feedback_chars = int(
                getattr(
                    self.config,
                    "costada_lean_feedback_chars",
                    getattr(self.config, "costada_cheap_feedback_chars", 300),
                )
            )
        elif prompt_mode == "rich":
            self._costada_feedback_chars = int(getattr(self.config, "costada_rich_feedback_chars", 1500))
        else:
            self._costada_feedback_chars = int(
                getattr(self.config, "costada_standard_feedback_chars", 800)
            )

    def _sample_from_archive_costada(
        self,
        island_idx: int,
        num_context_programs: int,
        mode: str,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        archive = self.archives[island_idx]
        if archive.size() == 0:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        population = archive.get_all()
        if mode == "exploitation" and archive.config.pareto_objectives and archive._pareto_ranks:
            parent = self._sample_pareto_front(archive, population)
        elif mode == "exploitation":
            parent = self._sample_top(population)
        else:
            parent = archive.sample_parent(mode)
        if parent is None:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        num = max(0, int(num_context_programs or 0))
        local_count = max(1, int(num * self.local_context_program_ratio)) if num > 0 else 0
        global_count = max(0, num - local_count)
        local_context_programs = archive.sample_other_context_programs(parent, local_count)
        global_context_programs = self._sample_global_top(parent.id, global_count)
        other_context_programs = local_context_programs + global_context_programs

        explore_label, exploit_label = self._get_mode_labels()
        if mode == "exploration":
            label = explore_label
        elif mode == "exploitation":
            label = exploit_label
        else:
            label = ""
        self._last_sampling_mode = mode
        logger.debug(
            "CostAda sampled parent %s from frontier %s in %s mode",
            parent.id[:8],
            island_idx,
            mode,
        )
        return {label: parent}, {"": other_context_programs}

    def _sample_legacy_costada(
        self,
        island_idx: int,
        num_context_programs: int,
        mode: str,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        population = self.islands[island_idx]
        if not population:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        if mode == "exploration":
            parent = self._sample_random(population)
        elif mode == "exploitation":
            parent = self._sample_top(population)
        else:
            parent = self._sample_weighted(population)
        other_context_programs = self._sample_global_top(parent.id, int(num_context_programs or 0))

        explore_label, exploit_label = self._get_mode_labels()
        if mode == "exploration":
            label = explore_label
        elif mode == "exploitation":
            label = exploit_label
        else:
            label = ""
        self._last_sampling_mode = mode
        logger.debug(
            "CostAda sampled parent %s from frontier %s in %s mode",
            parent.id[:8],
            island_idx,
            mode,
        )
        return {label: parent}, {"": other_context_programs}
