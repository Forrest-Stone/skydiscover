# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search strategy with stagnation detection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score_history: List[float] = []
        self.stagnation_count: int = 0
        self.parent_usage: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program and track progress."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track best score history for stagnation detection
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            if not self.best_score_history or score > max(self.best_score_history):
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            self.best_score_history.append(max(self.best_score_history[-1] if self.best_score_history else score, score))

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract score from program."""
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def _weighted_parent_selection(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Select parent weighted by score with usage penalty."""
        if len(candidates) == 1:
            return candidates[0]
        
        weights = []
        for p in candidates:
            score = self._get_score(p)
            usage = self.parent_usage.get(p.id, 0)
            # Weight = score^2 with usage penalty
            weight = (score ** 2) * (0.7 ** usage)
            weights.append(max(weight, 0.01))
        
        return random.choices(candidates, weights=weights, k=1)[0]

    def _diverse_context(self, candidates: List[EvolvedProgram], parent: EvolvedProgram, count: int) -> List[EvolvedProgram]:
        """Select diverse context from different score tiers."""
        scored = [(self._get_score(p), p) for p in candidates if p.id != parent.id]
        if not scored:
            return []
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        context = []
        n = len(scored)
        
        # Top performer
        if n > 0:
            context.append(scored[0][1])
        
        # Middle tier
        if n > 2:
            mid_idx = n // 2
            if scored[mid_idx][1].id not in [c.id for c in context]:
                context.append(scored[mid_idx][1])
        
        # Lower tier (but not worst)
        if n > 4:
            low_idx = int(n * 0.75)
            if scored[low_idx][1].id not in [c.id for c in context]:
                context.append(scored[low_idx][1])
        
        return context[:count]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Sample parent with adaptive strategy and diverse context."""
        candidates = list(self.programs.values())
        
        if not candidates:
            raise ValueError("No candidates available for sampling")
        
        parent = self._weighted_parent_selection(candidates)
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Determine label based on stagnation
        label = ""
        if self.stagnation_count >= 5:
            # Extended stagnation - try divergence
            label = self.DIVERGE_LABEL
            self.stagnation_count = 0
        elif self.stagnation_count >= 3 and self._get_score(parent) >= 0.9:
            # Good program but stuck - refine it
            label = self.REFINE_LABEL
        
        context = [] if label else self._diverse_context(candidates, parent, num_context_programs)
        
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END