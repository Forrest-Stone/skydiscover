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
    """Strategic search focusing on top performers with adaptive refinement."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stagnation = 0
        self.parent_counts: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            if score > self.best_score + 0.01:
                self.stagnation = 0
                self.best_score = score
            else:
                self.stagnation += 1
        
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def _score(self, p: EvolvedProgram) -> float:
        s = p.metrics.get("combined_score", 0)
        return float(s) if isinstance(s, (int, float)) else 0.0

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        progs = list(self.programs.values())
        if not progs:
            raise ValueError("No programs available")
        
        scored = sorted([(p, self._score(p)) for p in progs], key=lambda x: -x[1])
        top = [p for p, s in scored if s >= 0.95]
        mid = [p for p, s in scored if 0.7 <= s < 0.95]
        
        label = ""
        parent = None
        
        # When stuck: REFINE least-used top performer
        if self.stagnation > 5 and top:
            top.sort(key=lambda p: self.parent_counts.get(p.id, 0))
            parent = top[0]
            label = self.REFINE_LABEL
        # Default: weighted selection from top tier
        elif top:
            weights = [1.0 / (1 + self.parent_counts.get(p.id, 0)) for p in top]
            parent = random.choices(top, weights=weights)[0]
        elif mid:
            parent = random.choice(mid)
        else:
            parent = random.choice(progs)
        
        self.parent_counts[parent.id] = self.parent_counts.get(parent.id, 0) + 1
        
        # Context: top performers for guidance
        context = []
        top_others = [p for p in top if p.id != parent.id]
        if top_others:
            context.extend(random.sample(top_others, min(2, len(top_others))))
        if mid and len(context) < (num_context_programs or 4):
            context.extend(random.sample(mid, min(1, len(mid))))
        if len(context) < (num_context_programs or 4):
            remaining = [p for p in progs if p.id != parent.id and p not in context]
            if remaining:
                context.extend(random.sample(remaining, min((num_context_programs or 4) - len(context), len(remaining))))
        
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END