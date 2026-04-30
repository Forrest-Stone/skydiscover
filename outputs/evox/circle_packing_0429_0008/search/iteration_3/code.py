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
    """Focus on top-tier programs with strategic refinement."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_seen = 0.0
        self.iterations_stuck = 0
        self.parent_counts = {}
        self.refine_attempts = {}

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

        # Get valid scored programs
        scored = [(p, float(p.metrics.get("combined_score", 0))) 
                  for p in programs if isinstance(p.metrics.get("combined_score"), (int, float))]

        if not scored:
            parent = random.choice(programs)
            return {"": parent}, {"": []}

        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Separate into tiers
        top_tier = [(p, s) for p, s in scored if s >= 0.98]
        
        # Default: select from top tier with usage penalty
        if top_tier:
            candidates = top_tier
        else:
            candidates = scored[:max(1, len(scored) // 4)]

        # Weight by score and penalize overused parents
        weights = [(s + 0.1) * (0.5 ** self.parent_counts.get(p.id, 0)) for p, s in candidates]
        total = sum(weights)
        if total > 0:
            parent = random.choices([p for p, s in candidates], weights=weights, k=1)[0]
        else:
            parent = candidates[0][0]
        
        self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1

        # Context: best + diverse from different tiers
        best = scored[0][0]
        context = []
        if best.id != parent.id:
            context.append(best)
        
        # Add diverse samples from top, middle, bottom
        n = len(scored)
        for idx in [0, n // 2, n - 1]:
            if len(context) < num_context_programs and idx < n:
                prog = scored[idx][0]
                if prog.id not in [c.id for c in context]:
                    context.append(prog)

        # Stagnation handling: REFINE near-best, DIVERGE only when deeply stuck
        parent_label = ""
        if self.iterations_stuck > 8:
            parent_label = self.DIVERGE_LABEL
            context = []
        elif self.iterations_stuck > 4 and top_tier:
            # Try refining a top-tier program that hasn't been refined much
            refine_candidates = [(p, s) for p, s in top_tier 
                                  if self.refine_attempts.get(p.id, 0) < 2]
            if refine_candidates:
                parent = min(refine_candidates, key=lambda x: self.refine_attempts.get(x[0].id, 0))[0]
                self.refine_attempts[parent.id] = self.refine_attempts.get(parent.id, 0) + 1
                parent_label = self.REFINE_LABEL
                context = [scored[0][0]] if scored[0][0].id != parent.id else []

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END