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
    """Exploit top performers with diverse context; explore when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stuck_count = 0
        self.parent_uses = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > self.best_score + 0.01:
                self.best_score = score
                self.stuck_count = 0
            else:
                self.stuck_count += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        programs = list(self.programs.values())

        if not programs:
            raise ValueError("No programs available")

        # Get programs with valid scores, sorted descending
        scored = [(p, float(p.metrics.get("combined_score", 0))) 
                  for p in programs 
                  if isinstance(p.metrics.get("combined_score"), (int, float))]
        
        if not scored:
            return {"": random.choice(programs)}, {"": []}
        
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]

        # Initialize best_score on first call
        if self.best_score == 0.0:
            self.best_score = scored[0][1]

        # Parent selection: exploit top tier (score >= 0.98) with usage penalty
        top_tier = [(p, s) for p, s in scored if s >= 0.98]
        
        if top_tier and self.stuck_count < 6:
            # Weight by score and inverse usage
            weights = [s * (0.5 ** self.parent_uses.get(p.id, 0)) for p, s in top_tier]
            parent = random.choices([p for p, s in top_tier], weights=weights, k=1)[0]
        else:
            # Explore: pick from middle or lower tier
            others = [(p, s) for p, s in scored if s < 0.98]
            if others:
                parent = random.choice([p for p, s in others])
            else:
                parent = best

        self.parent_uses[parent.id] = self.parent_uses.get(parent.id, 0) + 1

        # Context: best + diverse samples from different tiers
        context = []
        if best.id != parent.id:
            context.append(best)
        
        remaining = [p for p, s in scored if p.id != parent.id and p.id != best.id]
        if remaining:
            # Add diverse context from top, middle, bottom
            for idx in [0, len(remaining) // 2, len(remaining) - 1]:
                if len(context) < num_context_programs and idx < len(remaining):
                    context.append(remaining[idx])

        # Stagnation handling
        parent_label = ""
        if self.stuck_count >= 6:
            parent_label = self.DIVERGE_LABEL
            context = []

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END