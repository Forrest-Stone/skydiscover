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
    """Search strategy with score-weighted parent selection and stagnation handling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.stagnation_count: int = 0
        self.best_score: float = 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program and track improvement state."""
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track stagnation based on meaningful improvement (> 0.01)
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > self.best_score + 0.01:
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            if score > self.best_score:
                self.best_score = score

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Score-weighted parent selection with usage penalty.
        Context includes top performers and random diverse programs.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        # Score-weighted parent selection with usage penalty
        weights = []
        for p in candidates:
            raw_score = p.metrics.get("combined_score", 0)
            score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
            usage = self.parent_usage.get(p.id, 0)
            weight = max(0.01, score) / (1 + usage)
            weights.append(weight)

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(candidates)] * len(candidates)

        parent = random.choices(candidates, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Select context: mix of top scorers and random diverse programs
        others = [p for p in candidates if p.id != parent.id]

        def get_score(p):
            s = p.metrics.get("combined_score", 0)
            return float(s) if isinstance(s, (int, float)) else 0.0

        sorted_others = sorted(others, key=get_score, reverse=True)
        n_context = min(num_context_programs, len(sorted_others))
        n_top = n_context // 2

        context = sorted_others[:n_top] if n_top > 0 else []
        remaining = sorted_others[n_top:]
        n_random = n_context - len(context)
        if remaining and n_random > 0:
            context.extend(random.sample(remaining, min(n_random, len(remaining))))

        # Determine parent label based on stagnation
        parent_label = ""
        raw_parent_score = parent.metrics.get("combined_score", 0)
        parent_score = float(raw_parent_score) if isinstance(raw_parent_score, (int, float)) else 0.0

        if self.stagnation_count >= 5:
            # Deeply stuck - signal divergence
            parent_label = self.DIVERGE_LABEL
            self.stagnation_count = 0
            context = []  # Empty context for targeted divergence
        elif self.stagnation_count >= 3 and parent_score > 0.85:
            # Stuck but parent is promising - signal refinement
            parent_label = self.REFINE_LABEL
            self.stagnation_count = 0

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END