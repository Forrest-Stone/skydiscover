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
    """Simple exploitation-focused search."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score: float = 0.0
        self.stagnation: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        
        # Track stagnation
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            if float(score) > self.best_score + 0.01:
                self.best_score = float(score)
                self.stagnation = 0
            else:
                self.stagnation += 1
        
        self._update_best_program(program)
        return program.id

    def _get_sorted(self) -> List[EvolvedProgram]:
        scored = [(float(p.metrics.get("combined_score", 0)), p) 
                  for p in self.programs.values()
                  if isinstance(p.metrics.get("combined_score"), (int, float))]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        sorted_progs = self._get_sorted()
        if not sorted_progs:
            raise ValueError("No candidates")

        # Stagnation -> DIVERGE from best to explore new directions
        if self.stagnation >= 5:
            parent = sorted_progs[0]
            return {self.DIVERGE_LABEL: parent}, {"": []}

        # Normal: pick from top 3 with preference for less-used
        top = sorted_progs[:min(3, len(sorted_progs))]
        parent = random.choice(top)

        # Context: best + diverse samples
        others = [p for p in sorted_progs if p.id != parent.id]
        context = []
        if others:
            context.append(others[0])  # Best available
        if len(others) > 1:
            context.append(others[-1])  # Worst for contrast
        if len(others) > 2:
            context.append(others[len(others)//2])  # Median
        context = context[:num_context_programs]

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END