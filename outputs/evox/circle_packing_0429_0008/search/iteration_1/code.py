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
    """Score-weighted selection with diverse context and stagnation handling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.iterations_stuck = 0
        self.best_seen = 0.0
        self.parent_counts = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        # Track stagnation: meaningful improvement = > 0.01 gain
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

        # Initialize best_seen from existing programs (handles checkpoint resume)
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

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]

        # Score-weighted parent selection with usage penalty for diversity
        weights = [(s + 0.1) * (0.7 ** self.parent_counts.get(p.id, 0)) for p, s in scored]
        parent = random.choices([p for p, s in scored], weights=weights, k=1)[0]
        self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1

        # Context: best + diverse programs from different score tiers
        context = [best] if best.id != parent.id else []
        others = [p for p, s in scored if p.id != parent.id and p not in context]
        if others:
            n = len(others)
            # Sample from top, middle, and bottom score ranges
            for i in [0, n // 2, n - 1]:
                if len(context) < num_context_programs and i < n:
                    context.append(others[i])

        # Stagnation handling: use DIVERGE label when stuck
        parent_label = ""
        if self.iterations_stuck > 5:
            parent_label = self.DIVERGE_LABEL
            context = []

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END