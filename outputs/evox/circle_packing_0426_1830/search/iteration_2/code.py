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
    """Adaptive search with stagnation detection and diverse selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.iterations_without_improvement = 0
        self.parent_usage_count: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track progress state."""
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > self.best_score:
                if score - self.best_score > 0.01:
                    self.iterations_without_improvement = 0
                else:
                    self.iterations_without_improvement += 1
                self.best_score = score
            else:
                self.iterations_without_improvement += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context with adaptive strategy."""
        if not self.programs:
            raise ValueError("No programs available")

        sorted_programs = sorted(self.programs.values(), key=self._get_score, reverse=True)
        
        # Score-weighted parent selection with usage penalty
        weights = []
        for p in sorted_programs:
            score = self._get_score(p)
            usage = self.parent_usage_count.get(p.id, 0)
            weights.append((score + 0.1) * (0.5 ** usage))
        
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        parent = sorted_programs[0]
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                parent = sorted_programs[i]
                break

        self.parent_usage_count[parent.id] = self.parent_usage_count.get(parent.id, 0) + 1

        # Determine label based on stagnation
        label = ""
        if self.iterations_without_improvement >= 8:
            label = self.DIVERGE_LABEL
            self.parent_usage_count.clear()
        elif self.iterations_without_improvement >= 4:
            label = self.REFINE_LABEL

        # Context: best + diverse from different tiers
        context = []
        parent_id = parent.id
        
        # Always include best if different from parent
        if sorted_programs[0].id != parent_id:
            context.append(sorted_programs[0])

        # Add diverse programs from different score ranges
        remaining = [p for p in sorted_programs if p.id != parent_id and p not in context]
        if remaining:
            tier_size = max(1, len(remaining) // 3)
            for i in range(3):
                tier = remaining[i * tier_size:(i + 1) * tier_size if i < 2 else len(remaining)]
                if tier and len(context) < (num_context_programs or 4):
                    context.append(random.choice(tier))

        return {label: parent}, {"": context[:num_context_programs]}


# EVOLVE-BLOCK-END