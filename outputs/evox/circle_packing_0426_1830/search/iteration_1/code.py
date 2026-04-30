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
    """Adaptive search with score-weighted selection and stagnation handling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.stagnation_count: int = 0
        self.last_best_score: float = 0.0
        self.label_attempts: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        # Track best score and stagnation
        current_best = self._get_best_score()
        if current_best > self.last_best_score + 0.01:
            self.stagnation_count = 0
            self.last_best_score = current_best
        else:
            self.stagnation_count += 1

        # Track parent usage
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def _get_best_score(self) -> float:
        best = 0.0
        for p in self.programs.values():
            score = p.metrics.get("combined_score", 0)
            if isinstance(score, (int, float)):
                best = max(best, float(score))
        return best

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")

        parent = self._select_parent(candidates)
        context = self._select_context(candidates, parent, num_context_programs or 4)

        # Use labels strategically during stagnation
        label = ""
        if self.stagnation_count >= 6:
            attempts = self.label_attempts.get(parent.id, 0)
            if attempts < 2:
                label = self.DIVERGE_LABEL if self.stagnation_count >= 10 else self.REFINE_LABEL
                self.label_attempts[parent.id] = attempts + 1

        return {label: parent}, {"": context}

    def _select_parent(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        weights = []
        for p in candidates:
            score = p.metrics.get("combined_score", 0)
            weight = float(score) + 0.1 if isinstance(score, (int, float)) else 0.1
            # Penalize overused parents exponentially
            usage = self.parent_usage.get(p.id, 0)
            weight *= 0.5 ** usage
            weights.append(max(weight, 0.01))

        total = sum(weights)
        r = random.random() * total
        cumsum = 0.0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return candidates[i]
        return candidates[-1]

    def _select_context(
        self, candidates: List[EvolvedProgram], parent: EvolvedProgram, num_context: int
    ) -> List[EvolvedProgram]:
        others = [p for p in candidates if p.id != parent.id]
        if len(others) <= num_context:
            return others

        # Sort by score
        scored = [(p, float(p.metrics.get("combined_score", 0))) 
                  for p in others if isinstance(p.metrics.get("combined_score"), (int, float))]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Half from top performers, half random for diversity
        n_top = min(num_context // 2, len(scored))
        n_random = num_context - n_top

        context = [p for p, _ in scored[:n_top]]
        remaining = [p for p, _ in scored[n_top:]]
        if remaining and n_random > 0:
            context.extend(random.sample(remaining, min(n_random, len(remaining))))

        return context[:num_context]


# EVOLVE-BLOCK-END