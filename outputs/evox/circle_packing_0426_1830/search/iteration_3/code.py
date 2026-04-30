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
    """Score-weighted selection with stagnation-aware label application."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.best_score_history: List[float] = []
        self.stagnation_count: int = 0
        self.last_best_score: float = 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track stagnation state."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track best score and stagnation
        current_best = self._get_best_score()
        if current_best > self.last_best_score + 0.01:
            self.stagnation_count = 0
            self.last_best_score = current_best
        else:
            self.stagnation_count += 1
        
        self.best_score_history.append(current_best)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def _get_best_score(self) -> float:
        """Get current best score from population."""
        best = 0.0
        for p in self.programs.values():
            score = p.metrics.get("combined_score", 0)
            if isinstance(score, (int, float)):
                best = max(best, float(score))
        return best

    def _score_weighted_select(self, programs: List[EvolvedProgram], penalty: bool = True) -> EvolvedProgram:
        """Select program with score-based weights and usage penalty."""
        weights = []
        for p in programs:
            score = p.metrics.get("combined_score", 0)
            if not isinstance(score, (int, float)):
                score = 0.0
            score = float(score)
            
            # Apply usage penalty
            usage = self.parent_usage.get(p.id, 0)
            weight = score * (0.5 ** usage) if penalty else score
            weights.append(max(weight, 0.01))
        
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for p, w in zip(programs, weights):
            cumulative += w
            if r <= cumulative:
                self.parent_usage[p.id] = self.parent_usage.get(p.id, 0) + 1
                return p
        return programs[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent with score-weighting; apply labels on stagnation."""
        candidates = list(self.programs.values())
        
        if len(candidates) == 0:
            raise ValueError("No candidates available")
        
        # Sort by score for context selection
        sorted_by_score = sorted(
            candidates,
            key=lambda p: float(p.metrics.get("combined_score", 0)) if isinstance(p.metrics.get("combined_score"), (int, float)) else 0,
            reverse=True
        )
        
        # Stagnation handling: use DIVERGE_LABEL to break out
        if self.stagnation_count >= 8:
            # Pick a mid-tier program to diverge from
            mid_idx = len(sorted_by_score) // 3
            parent = sorted_by_score[mid_idx]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}
        
        # Normal: score-weighted parent selection favoring top performers
        parent = self._score_weighted_select(candidates)
        
        # Context: mix of top performers (excluding parent)
        context = [p for p in sorted_by_score if p.id != parent.id][:num_context_programs]
        
        # Pad with random if needed
        if len(context) < num_context_programs:
            remaining = [p for p in candidates if p.id != parent.id and p not in context]
            if remaining:
                context.extend(random.sample(remaining, min(num_context_programs - len(context), len(remaining))))
        
        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END