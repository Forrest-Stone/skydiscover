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
    """Search strategy focused on breaking stagnation through diverse parent/context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = 0.0
        self.stagnation_count = 0
        self.parent_usage = {}  # Track how often each program is used as parent
        self.diverge_count = 0
        self.refine_count = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program and track progress state."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track stagnation - only count meaningful improvements
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)) and score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract score from program."""
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def _select_parent(self, programs: List[EvolvedProgram], top_programs: List[EvolvedProgram]) -> EvolvedProgram:
        """Select parent with diversity - prefer less-used, good-scoring programs."""
        # Sort by score
        scored = [(p, self._get_score(p)) for p in programs]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Focus on top half for exploitation
        top_half = [p for p, s in scored[:max(1, len(scored) // 2)]]
        
        # Apply usage penalty for diversity
        min_usage = min(self.parent_usage.get(p.id, 0) for p in top_half)
        underused = [p for p in top_half if self.parent_usage.get(p.id, 0) <= min_usage + 1]
        
        if underused:
            parent = random.choice(underused)
        else:
            parent = random.choice(top_half)
        
        # Track usage
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        return parent

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context with adaptive strategy based on stagnation."""
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        # Sort programs by score
        scored = [(p, self._get_score(p)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_programs = [p for p, s in scored]
        top_programs = sorted_programs[:max(1, len(sorted_programs) // 4)]

        # Stagnation recovery: use REFINE_LABEL on best program periodically
        if self.stagnation_count >= 5 and self.refine_count < 3:
            self.refine_count += 1
            parent = sorted_programs[0]  # Best program
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.REFINE_LABEL: parent}, {}

        # Deeper stagnation: use DIVERGE_LABEL on mid-tier program
        if self.stagnation_count >= 10 and self.diverge_count < 2:
            self.diverge_count += 1
            mid_idx = len(sorted_programs) // 2
            parent = sorted_programs[mid_idx]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}

        # Normal selection: diverse parent + varied context
        parent = self._select_parent(candidates, top_programs)
        
        # Context: best + some variety (different score ranges)
        context = [sorted_programs[0]]  # Always include best
        if len(sorted_programs) > 2:
            # Add a mid-tier program
            mid_idx = len(sorted_programs) // 2
            if sorted_programs[mid_idx].id != parent.id:
                context.append(sorted_programs[mid_idx])
        if len(sorted_programs) > 4:
            # Add a lower-tier program for diversity
            low_idx = 3 * len(sorted_programs) // 4
            if sorted_programs[low_idx].id != parent.id and sorted_programs[low_idx] not in context:
                context.append(sorted_programs[low_idx])
        
        # Remove parent from context if present
        context = [p for p in context if p.id != parent.id][:num_context_programs]

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END