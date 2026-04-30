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
    """Adaptive search with exploitation and stagnation handling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage_count: Dict[str, int] = {}
        self.iterations_without_improvement: int = 0
        self.best_score: float = 0.0

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract combined_score from program metrics."""
        s = program.metrics.get("combined_score", 0)
        return float(s) if isinstance(s, (int, float)) else 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track improvement state."""
        score = self._get_score(program)
        if score > self.best_score:
            if score - self.best_score > 0.01:
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1
            self.best_score = score
        else:
            self.iterations_without_improvement += 1

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent from top performers; context from top programs."""
        programs = list(self.programs.values())
        if not programs:
            raise ValueError("No candidates available for sampling")

        # Sort by score descending
        sorted_programs = sorted(programs, key=self._get_score, reverse=True)
        
        # Initialize best_score from existing population
        if self.best_score == 0.0:
            self.best_score = self._get_score(sorted_programs[0])
        
        best_score = self._get_score(sorted_programs[0])

        # Parent: top half, weighted by score with usage penalty
        top_half = sorted_programs[:max(1, len(sorted_programs) // 2)]
        weights = [self._get_score(p) / (1 + self.parent_usage_count.get(p.id, 0)) for p in top_half]
        parent = random.choices(top_half, weights=weights, k=1)[0]
        self.parent_usage_count[parent.id] = self.parent_usage_count.get(parent.id, 0) + 1

        # Context: top programs excluding parent
        context = [p for p in sorted_programs if p.id != parent.id][:num_context_programs]

        # Stagnation handling
        parent_label = ""
        if self.iterations_without_improvement >= 3:
            parent_score = self._get_score(parent)
            if best_score > 0 and parent_score >= best_score * 0.98:
                parent_label = self.REFINE_LABEL
            else:
                parent_label = self.DIVERGE_LABEL

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END