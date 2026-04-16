"""CostAda database: minimal tier-aware extension over AdaEvolveDatabase."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program


class CostAdaDatabase(AdaEvolveDatabase):
    """AdaEvolve DB with lightweight tier-conditioned sampling knobs."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self._costada_feedback_chars: Optional[int] = None

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
        """Sample parent/context with tier-aware context-depth controls."""
        num = int(num_context_programs or 4)
        if tier == "cheap":
            num = min(num, int(getattr(self.config, "costada_cheap_context_programs", 2)))
            self._costada_feedback_chars = int(getattr(self.config, "costada_cheap_feedback_chars", 300))
        elif tier == "rich":
            num = max(num, int(getattr(self.config, "costada_rich_context_programs", 6)))
            self._costada_feedback_chars = int(getattr(self.config, "costada_rich_feedback_chars", 1500))
        else:
            self._costada_feedback_chars = int(
                getattr(self.config, "costada_standard_feedback_chars", 800)
            )

        return super().sample(
            num_context_programs=num,
            force_exploration=force_exploration,
            island_id=island_id,
            intensity=intensity,
            family=family,
            tier=tier,
            budget_bin=budget_bin,
            **kwargs,
        )
