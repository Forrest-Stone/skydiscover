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
    """Adaptive search strategy for circle packing optimization."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.my_start_iteration: Optional[int] = None
        self.last_diverge_iteration: int = -100

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track parent usage for diversity."""
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track parent usage to avoid overuse
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely get combined_score from program metrics."""
        score = program.metrics.get("combined_score", 0.0)
        if not isinstance(score, (int, float)):
            return 0.0
        return float(score)

    def _weighted_select(self, programs: List[EvolvedProgram]) -> EvolvedProgram:
        """Weighted random selection favoring high scores, penalizing overuse."""
        if len(programs) == 1:
            return programs[0]
        
        weights = []
        for p in programs:
            score = self._get_score(p)
            usage = self.parent_usage.get(p.id, 0)
            w = score * (0.6 ** usage)  # Decay weight for overused parents
            weights.append(max(w, 0.01))
        
        total = sum(weights)
        r = random.uniform(0, total)
        cumsum = 0.0
        for p, w in zip(programs, weights):
            cumsum += w
            if r <= cumsum:
                return p
        return programs[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Adaptive selection with phase-based strategy.
        
        Phase 1 (iterations 0-2): Default - exploit top performers with rich context
        Phase 2 (iterations 3-5): REFINE on best candidates  
        Phase 3 (iterations 6+): DIVERGE sparingly on mid-tier
        """
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")

        # Track when this algorithm started running
        if self.my_start_iteration is None:
            current_iter = max((p.iteration_found for p in candidates), default=0)
            self.my_start_iteration = current_iter
        
        my_iterations = self.last_iteration - self.my_start_iteration

        # Sort by score descending
        sorted_progs = sorted(candidates, key=self._get_score, reverse=True)
        best_score = self._get_score(sorted_progs[0])

        # Strategy selection based on algorithm's run time
        label = ""
        if my_iterations >= 6 and (self.last_iteration - self.last_diverge_iteration) >= 3 and best_score > 0.95:
            label = self.DIVERGE_LABEL
            self.last_diverge_iteration = self.last_iteration
        elif my_iterations >= 3 and best_score > 0.95:
            label = self.REFINE_LABEL

        # Parent and context selection
        if label == self.DIVERGE_LABEL:
            # Mid-tier parent for exploration
            start = len(sorted_progs) // 4
            end = len(sorted_progs) // 2
            mid_tier = sorted_progs[start:end] if end > start else sorted_progs[start:]
            parent = random.choice(mid_tier) if mid_tier else sorted_progs[start]
            context = []  # Empty context for pure divergence
        else:
            # Top performers with usage penalty
            top_n = min(8, len(sorted_progs))
            parent = self._weighted_select(sorted_progs[:top_n])
            
            # Context: prioritize high-scoring programs (breakthrough pattern insight)
            context = [p for p in sorted_progs[:6] if p.id != parent.id][:num_context_programs]
            
            # Add diverse programs if space remains
            if len(context) < num_context_programs:
                remaining = [p for p in sorted_progs if p.id != parent.id and p not in context]
                if remaining:
                    context.extend(random.sample(remaining, min(num_context_programs - len(context), len(remaining))))

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END