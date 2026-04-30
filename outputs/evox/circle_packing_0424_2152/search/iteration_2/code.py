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
    """Strategic search for breaking through high-performing plateaus."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score: float = 0.0
        self.stagnation_count: int = 0
        self.parent_usage: Dict[str, int] = {}
        self.last_parent_id: Optional[str] = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track progress."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track best score and stagnation
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > self.best_score:
                improvement = score - self.best_score
                # Count meaningful improvement (>1% relative or >0.01 absolute)
                if improvement > 0.01 or (self.best_score > 0 and improvement / self.best_score > 0.01):
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1
                self.best_score = score
            else:
                self.stagnation_count += 1
        
        # Track parent usage
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract numeric score."""
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Strategic sampling: exploit best, diverse context, signal divergence when stuck."""
        candidates = list(self.programs.values())
        
        if not candidates:
            raise ValueError("No candidates available for sampling")

        # Sort by score
        sorted_programs = sorted(candidates, key=self._get_score, reverse=True)
        
        # Select parent from top performers, with some randomness
        top_n = min(5, len(sorted_programs))
        # Weight towards best but add randomness
        weights = [max(1, top_n - i) for i in range(top_n)]
        parent = random.choices(sorted_programs[:top_n], weights=weights, k=1)[0]
        
        # Avoid overusing same parent
        if self.parent_usage.get(parent.id, 0) >= 3 and top_n > 1:
            parent = random.choice(sorted_programs[:top_n])
        
        # Build diverse context: mix of high-scoring and contrasting examples
        context = []
        others = [p for p in sorted_programs if p.id != parent.id]
        
        # Add top performers (excluding parent)
        top_context = [p for p in others[:4] if p.id != parent.id]
        context.extend(top_context[:2])
        
        # Add from different score ranges for diversity
        if len(others) > 4:
            mid_idx = len(others) // 2
            low_idx = int(len(others) * 0.75)
            if mid_idx < len(others) and others[mid_idx].id not in [c.id for c in context]:
                context.append(others[mid_idx])
            if low_idx < len(others) and others[low_idx].id not in [c.id for c in context]:
                context.append(others[low_idx])
        
        context = context[:num_context_programs]
        
        # Determine label - use DIVERGE when stuck
        label = ""
        if self.stagnation_count >= 8:
            label = self.DIVERGE_LABEL
            # Reset stagnation after signaling divergence
            self.stagnation_count = 0
        
        parent_dict = {label: parent}
        context_programs_dict = {"": context}
        
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END