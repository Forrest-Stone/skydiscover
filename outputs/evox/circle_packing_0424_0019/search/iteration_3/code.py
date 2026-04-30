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
    """
    Strategic search for breaking plateaus.
    
    Key principles:
    1. Use DIVERGE_LABEL early when in a known plateau
    2. Select mid-tier parents for divergence to explore new directions
    3. Track stagnation and use DIVERGE when stuck
    4. Normal selection exploits top performers with diverse context
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.stagnation = 0
        self.best_score: Optional[float] = None
        self.diverge_count = 0
        self.first_sample = True

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track stagnation based on meaningful improvement."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        
        # Track stagnation - reset only on meaningful improvement (>0.01)
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if self.best_score is None or score > self.best_score + 0.01:
                self.stagnation = 0
                self.best_score = score
            else:
                self.stagnation += 1
        
        if self.config.db_path:
            self._save_program(program)
        
        self._update_best_program(program)
        return program.id

    def _get_score(self, p: EvolvedProgram) -> float:
        """Safely extract score from program metrics."""
        s = p.metrics.get("combined_score", 0)
        return float(s) if isinstance(s, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context strategically based on search state."""
        if not self.programs:
            raise ValueError("No programs available")
        
        # Initialize best_score from existing population
        if self.best_score is None:
            self.best_score = max(self._get_score(p) for p in self.programs.values())
        
        sorted_p = sorted(self.programs.values(), key=self._get_score, reverse=True)
        
        # Use DIVERGE on first sample (known plateau) or when stagnating
        # Limit usage to avoid over-exploration
        if (self.first_sample or self.stagnation >= 3) and self.diverge_count < 4:
            self.first_sample = False
            # Select mid-tier parent for divergence (based on what worked before)
            mid_idx = len(sorted_p) // 3
            parent = sorted_p[mid_idx]
            self.diverge_count += 1
            return {self.DIVERGE_LABEL: parent}, {}
        
        # Normal selection: exploit top performers
        top_n = max(1, len(sorted_p) // 3)
        parent = random.choice(sorted_p[:top_n])
        
        # Context: include best program + diverse samples
        context = []
        if sorted_p[0].id != parent.id:
            context.append(sorted_p[0])
        
        others = [p for p in sorted_p if p.id != parent.id and (not context or p.id != context[0].id)]
        if others:
            n = min(num_context_programs - len(context), len(others))
            context.extend(random.sample(others, n))
        
        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END