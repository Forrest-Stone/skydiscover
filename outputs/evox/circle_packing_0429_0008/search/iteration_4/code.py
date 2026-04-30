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
    """Targeted DIVERGE on low-tier, REFINE on high-tier when stuck."""

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
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs):
        programs = list(self.programs.values())
        
        scored = [(p, float(p.metrics.get("combined_score", 0))) 
                  for p in programs 
                  if isinstance(p.metrics.get("combined_score"), (int, float))]
        
        if not scored:
            return {"": random.choice(programs)}, {"": []}

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        best_score = scored[0][1]
        
        # Initialize from existing population - assume stagnation if already high
        if self.best_seen == 0.0:
            self.best_seen = best_score
            if best_score > 0.95:
                self.iterations_stuck = 5

        # When stuck, alternate DIVERGE on low-tier and REFINE on high-tier
        if self.iterations_stuck > 2:
            if self.iterations_stuck % 2 == 0:
                low_tier = [p for p, s in scored if s < 0.7]
                if low_tier:
                    parent = random.choice(low_tier)
                    return {self.DIVERGE_LABEL: parent}, {"": []}
            else:
                high_tier = [p for p, s in scored if s > 0.95]
                if high_tier:
                    parent = random.choice(high_tier)
                    return {self.REFINE_LABEL: parent}, {"": []}

        # Normal: score-weighted selection with usage penalty
        weights = [(s + 0.1) * (0.5 ** self.parent_counts.get(p.id, 0)) 
                   for p, s in scored]
        parent = random.choices([p for p, s in scored], weights=weights, k=1)[0]
        self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1

        # Context: best + diverse samples from different tiers
        context = [best] if best.id != parent.id else []
        others = [p for p, s in scored if p.id != parent.id and p not in context]
        for i in [0, len(others)//2, len(others)-1]:
            if len(context) < num_context_programs and 0 <= i < len(others):
                context.append(others[i])

        return {"": parent}, {"": context}
# EVOLVE-BLOCK-END