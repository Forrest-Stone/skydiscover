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
    """Search strategy with balanced exploration/exploitation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.iterations_without_improvement = 0
        self.last_best_score = -float('inf')

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program and track improvement state."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        
        # Track meaningful improvement (>0.01)
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            if score > self.last_best_score + 0.01:
                self.last_best_score = score
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1
        
        if self.config.db_path:
            self._save_program(program)
        
        self._update_best_program(program)
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract combined_score."""
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context with diversity to break stagnation."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")
        
        candidates.sort(key=self._get_score, reverse=True)
        parent_label = ""
        parent = None
        
        # When stuck for 5+ iterations, use DIVERGE on less-used mid-tier program
        if self.iterations_without_improvement >= 5:
            mid_start = len(candidates) // 4
            mid_end = 3 * len(candidates) // 4
            mid_tier = candidates[mid_start:mid_end]
            if mid_tier:
                mid_tier.sort(key=lambda p: self.parent_usage.get(p.id, 0))
                parent = mid_tier[0]
                parent_label = self.DIVERGE_LABEL
        
        # Normal selection: weight by score and inverse usage
        if parent is None:
            weights = []
            for p in candidates:
                score = self._get_score(p)
                usage = self.parent_usage.get(p.id, 0)
                weights.append((score + 0.1) / (usage + 1))
            
            total = sum(weights)
            if total > 0:
                r = random.random() * total
                cumsum = 0
                for i, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        parent = candidates[i]
                        break
            if parent is None:
                parent = candidates[0]
        
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Diverse context: best, mid, worst + random
        others = [p for p in candidates if p.id != parent.id]
        context = []
        if others:
            context.append(others[0])  # Best other
            if len(others) > 2:
                context.append(others[len(others) // 2])  # Mid
            if len(others) > 1:
                context.append(others[-1])  # Worst (shows what fails)
            # Fill with random unused
            remaining = [p for p in others if p not in context]
            random.shuffle(remaining)
            context.extend(remaining[:max(0, (num_context_programs or 4) - len(context))])
        
        return {parent_label: parent}, {"": context[:num_context_programs]}


# EVOLVE-BLOCK-END