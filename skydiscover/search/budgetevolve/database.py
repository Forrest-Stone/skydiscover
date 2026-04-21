"""BudgetEvolve database.

Keeps AdaEvolve storage/routing logic and adds budget-conditioned sampling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program


class BudgetEvolveDatabase(AdaEvolveDatabase):
    """AdaEvolve database + family/tier-conditioned sampling hooks."""

    def __init__(self, name, config):
        super().__init__(name, config)
        if self.use_paradigm_breakthrough:
            from .paradigm import ParadigmTracker

            self.paradigm_tracker = ParadigmTracker(
                window_size=getattr(config, "paradigm_window_size", 30),
                improvement_threshold=getattr(
                    config, "paradigm_improvement_threshold", 0.05),
                max_paradigm_uses=getattr(config, "paradigm_max_uses", 5),
                max_tried_paradigms=getattr(config, "paradigm_max_tried", 10),
                num_paradigms_to_generate=getattr(
                    config, "paradigm_num_to_generate", 3),
            )

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
        island_id: Optional[int] = None,
        intensity: Optional[float] = None,
        family: Optional[str] = None,
        tier: Optional[str] = None,
        budget_bin: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        island_idx = self.current_island if island_id is None else island_id
        if self.use_unified_archive and self.archives:
            return self._sample_from_archive_budget(
                island_idx,
                num_context_programs,
                force_exploration,
                intensity=intensity,
                family=family,
                tier=tier,
            )
        return self._sample_legacy_budget(
            island_idx,
            num_context_programs,
            force_exploration,
            intensity=intensity,
            family=family,
            tier=tier,
        )

    def _sample_from_archive_budget(
        self,
        island_idx: int,
        num_context_programs: Optional[int],
        force_exploration: bool,
        intensity: Optional[float],
        family: Optional[str],
        tier: Optional[str],
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        archive = self.archives[island_idx]
        if archive.size() == 0:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        if intensity is None:
            intensity = self.adapter.get_search_intensity(
                island_idx) if self.use_adaptive_search else self.fixed_intensity
        if force_exploration:
            intensity = self.intensity_max

        mode = "balanced"
        population = archive.get_all()
        if family == "refine":
            parent = self._sample_top(population)
            mode = "budget_refine"
        elif family == "structural":
            parent = archive.sample_parent("exploration")
            mode = "budget_structural"
        elif family == "tactic_guided":
            parent = archive.sample_parent("balanced")
            mode = "budget_tactic_guided"
        else:
            return super()._sample_from_archive(island_idx, num_context_programs, force_exploration)

        num = num_context_programs or 4
        if tier == "cheap":
            num = min(num, 2)
        elif tier == "rich":
            num = max(num, 5)

        local_count = max(1, int(num * self.local_context_program_ratio))
        global_count = num - local_count
        local_context_programs = archive.sample_other_context_programs(
            parent, local_count)
        global_context_programs = self._sample_global_top(
            parent.id, global_count)
        other_context_programs = local_context_programs + global_context_programs

        self._last_sampling_mode = mode
        return {"": parent}, {"": other_context_programs}

    def _sample_legacy_budget(
        self,
        island_idx: int,
        num_context_programs: Optional[int],
        force_exploration: bool,
        intensity: Optional[float],
        family: Optional[str],
        tier: Optional[str],
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        population = self.islands[island_idx]
        if not population:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        if family == "refine":
            parent = self._sample_top(population)
            mode = "budget_refine"
        elif family == "structural":
            parent = self._sample_random(population)
            mode = "budget_structural"
        elif family == "tactic_guided":
            parent = self._sample_weighted(population)
            mode = "budget_tactic_guided"
        else:
            return super()._sample_legacy(island_idx, num_context_programs, force_exploration)

        num = num_context_programs or 4
        if tier == "cheap":
            num = min(num, 2)
        elif tier == "rich":
            num = max(num, 5)

        other_context_programs = self._sample_global_top(parent.id, num)
        self._last_sampling_mode = mode
        return {"": parent}, {"": other_context_programs}
