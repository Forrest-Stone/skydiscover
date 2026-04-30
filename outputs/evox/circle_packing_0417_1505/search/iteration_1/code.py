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
    """Search strategy prioritizing top-tier programs."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = 0.0
        self.no_improvement: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track improvement."""
        self.programs[program.id] = program
        
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > self.best_score + 0.01:
                self.best_score = score
                self.no_improvement = 0
            else:
                self.no_improvement += 1
        
        if program.id not in self.parent_usage:
            self.parent_usage[program.id] = 0
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent from top tier, context from best programs."""
        programs = list(self.programs.values())
        
        # Score and sort descending
        scored = []
        for p in programs:
            s = p.metrics.get("combined_score", 0)
            if isinstance(s, (int, float)):
                scored.append((p, float(s)))
        if not scored:
            scored = [(p, 0.0) for p in programs]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Parent: from top half, weighted by score and inverse usage
        top_half = scored[:max(1, len(scored) // 2)]
        weights = [(s + 0.1) / (self.parent_usage.get(p.id, 0) + 1) for p, s in top_half]
        idx = random.choices(range(len(top_half)), weights=weights, k=1)[0]
        parent = top_half[idx][0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Label: DIVERGE when stuck for several iterations
        label = ""
        if self.no_improvement > 3 and random.random() < 0.25:
            label = self.DIVERGE_LABEL
        
        # Context: best programs excluding parent (not random, not bottom-heavy)
        context = [p for p, _ in scored if p.id != parent.id][:num_context_programs]
        
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END