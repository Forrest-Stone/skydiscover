# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Stagnation-aware search with DIVERGE/REFINE progression."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_seen = 0.0
        self.iterations_stuck = 0
        self.parent_counts = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > self.best_seen + 0.01:
                self.iterations_stuck = 0
                self.best_seen = score
            else:
                self.iterations_stuck += 1

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
        programs = list(self.programs.values())

        if not programs:
            raise ValueError("No programs available")

        # Initialize best_seen from existing programs
        if self.best_seen == 0.0:
            scores = [float(p.metrics.get("combined_score", 0)) 
                      for p in programs 
                      if isinstance(p.metrics.get("combined_score"), (int, float))]
            if scores:
                self.best_seen = max(scores)

        # Get programs with valid scores
        scored = [(p, float(p.metrics.get("combined_score", 0))) 
                  for p in programs if isinstance(p.metrics.get("combined_score"), (int, float))]

        if not scored:
            parent = random.choice(programs)
            return {"": parent}, {"": []}

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]

        parent_label = ""
        parent = None
        context = []

        # Stagnation handling: DIVERGE first, then REFINE
        if self.iterations_stuck > 3:
            # DIVERGE from low/mid-tier for exploration
            parent_label = self.DIVERGE_LABEL
            lower_tier = [p for p, s in scored if s < scored[len(scored)//2][1]]
            if lower_tier:
                # Apply usage penalty for diversity
                weights = [1.0 / (1 + self.parent_counts.get(p.id, 0)) for p in lower_tier]
                parent = random.choices(lower_tier, weights=weights, k=1)[0]
            else:
                parent = random.choice([p for p, _ in scored])
            self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1
            return {parent_label: parent}, {"": []}
        
        if self.iterations_stuck > 6:
            # REFINE near-best for focused optimization
            parent_label = self.REFINE_LABEL
            top_tier = [p for p, s in scored if s > 0.98]
            if top_tier:
                weights = [1.0 / (1 + self.parent_counts.get(p.id, 0)) for p in top_tier]
                parent = random.choices(top_tier, weights=weights, k=1)[0]
            else:
                parent = best
            self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1
            return {parent_label: parent}, {"": []}

        # Normal: score-weighted selection with usage penalty
        weights = [(s + 0.1) / (1 + self.parent_counts.get(p.id, 0)) for p, s in scored]
        parent = random.choices([p for p, _ in scored], weights=weights, k=1)[0]
        self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1

        # Context: best + diverse from different tiers
        if best.id != parent.id:
            context.append(best)
        others = [p for p, _ in scored if p.id != parent.id and p.id not in [c.id for c in context]]
        if others:
            n = len(others)
            for i in [0, n // 2, n - 1]:
                if len(context) < num_context_programs and i < n:
                    context.append(others[i])

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END