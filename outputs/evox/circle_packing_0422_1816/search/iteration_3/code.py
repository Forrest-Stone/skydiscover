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
    """Exploit top performers with strategic refinement."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.stagnation: int = 0
        self.best_score: float = 0.0
        self.refine_tried: set = set()

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        
        current_best = self._get_best_score()
        if current_best > self.best_score + 0.01:
            self.stagnation = 0
            self.best_score = current_best
            self.refine_tried.clear()
        else:
            self.stagnation += 1
        
        self._update_best_program(program)
        return program.id

    def _get_best_score(self) -> float:
        best = 0.0
        for p in self.programs.values():
            s = p.metrics.get("combined_score")
            if isinstance(s, (int, float)):
                best = max(best, float(s))
        return best

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

        # During stagnation, try REFINE on promising mid-tier program
        if self.stagnation >= 3:
            for p in sorted_progs[1:6]:  # Skip best, try ranks 2-6
                if p.id not in self.refine_tried:
                    self.refine_tried.add(p.id)
                    self.parent_usage[p.id] = self.parent_usage.get(p.id, 0) + 1
                    context = sorted_progs[:3] + [sorted_progs[-1]]
                    context = [x for x in context if x.id != p.id][:num_context_programs]
                    return {self.REFINE_LABEL: p}, {"": context}

        # Normal: select from TOP tier with usage weighting
        top = sorted_progs[:min(5, len(sorted_progs))]
        weights = [1.0 / (1 + self.parent_usage.get(p.id, 0)) for p in top]
        parent = random.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Diverse context: best + mid + worst
        others = [p for p in sorted_progs if p.id != parent.id]
        context = []
        if others:
            context.append(others[0])
        if len(others) >= 2:
            context.append(others[-1])
        if len(others) >= 3:
            context.append(others[len(others)//2])
        if len(others) >= 4:
            context.append(others[1])
        context = context[:num_context_programs]

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END