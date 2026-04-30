# EVOLVE-BLOCK-START
import logging
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search strategy for breaking stagnation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.best_score_history: List[float] = []
        self.stagnation_count: int = 0
        self.diverge_attempts: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track progress."""
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track best score and stagnation
        current_score = program.metrics.get("combined_score", 0)
        if isinstance(current_score, (int, float)):
            if not self.best_score_history or current_score > max(self.best_score_history):
                self.best_score_history.append(current_score)
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

        self._update_best_program(program)

        if self.config.db_path:
            self._save_program(program)

        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract score from program."""
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context with stagnation-aware strategy."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")

        # Score all programs
        scored = [(p, self._get_score(p)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        best_score = scored[0][1] if scored else 0.0

        # Select parent: weighted by score, penalized by usage
        weights = []
        for p, score in scored:
            usage = self.parent_usage.get(p.id, 0)
            # Higher score = higher weight, but penalize overuse
            weight = max(0.01, score * (0.5 ** usage))
            weights.append(weight)

        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        parent = scored[0][0]
        parent_idx = 0
        for i, (p, score) in enumerate(scored):
            cumulative += weights[i]
            if r <= cumulative:
                parent = p
                parent_idx = i
                break

        # Track usage
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Build diverse context: top performers + some lower scores
        context = []
        top_programs = [p for p, s in scored[:5] if p.id != parent.id]
        low_programs = [p for p, s in scored[5:] if p.id != parent.id]
        
        # Add 2-3 from top, 1-2 from bottom
        if top_programs:
            context.extend(random.sample(top_programs, min(2, len(top_programs))))
        if low_programs:
            context.extend(random.sample(low_programs, min(2, len(low_programs))))
        
        context = context[:num_context_programs]

        # Determine label: DIVERGE on stagnation for under-explored mid-tier parents
        label = ""
        parent_score = self._get_score(parent)
        diverge_count = self.diverge_attempts.get(parent.id, 0)
        
        # Trigger divergence if stagnating and parent is mid-tier not heavily diverged
        if self.stagnation_count >= 5 and diverge_count < 2:
            if 0.90 <= parent_score < best_score:
                label = self.DIVERGE_LABEL
                self.diverge_attempts[parent.id] = diverge_count + 1
                self.stagnation_count = max(0, self.stagnation_count - 3)
            elif self.stagnation_count >= 10 and parent_score < best_score:
                label = self.DIVERGE_LABEL
                self.diverge_attempts[parent.id] = diverge_count + 1
                self.stagnation_count = max(0, self.stagnation_count - 3)

        parent_dict = {label: parent}
        context_programs_dict = {"": context}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END