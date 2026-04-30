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
    """Adaptive search strategy with stagnation detection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stagnation_count = 0
        self.parent_usage: Dict[str, int] = {}

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract score from program metrics."""
        score = program.metrics.get("combined_score", 0.0)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track progress."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._get_score(program)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context with adaptive strategy."""
        candidates = list(self.programs.values())
        n = len(candidates)
        
        if n == 0:
            raise ValueError("No candidates available")

        # Sort by score
        sorted_candidates = sorted(candidates, key=self._get_score, reverse=True)
        
        # Weighted selection favoring better scores
        weights = []
        for p in sorted_candidates:
            s = self._get_score(p)
            usage = self.parent_usage.get(p.id, 0)
            # Higher score = higher weight, lower usage = higher weight
            w = (s + 0.1) * (1.0 / (1.0 + usage))
            weights.append(w)
        
        total = sum(weights)
        r = random.random() * total
        cumsum = 0.0
        parent = sorted_candidates[0]
        for i, p in enumerate(sorted_candidates):
            cumsum += weights[i]
            if cumsum >= r:
                parent = p
                break
        
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Build context: best program + diverse others
        context = []
        best = sorted_candidates[0]
        if best.id != parent.id:
            context.append(best)
        
        # Add mid-tier and lower-tier for diversity
        remaining = [p for p in sorted_candidates if p.id != parent.id and p.id not in [c.id for c in context]]
        if remaining:
            mid_idx = len(remaining) // 2
            if mid_idx < len(remaining):
                context.append(remaining[mid_idx])
            if len(remaining) > 2 and len(context) < num_context_programs:
                context.append(remaining[-1])  # Add worst
        
        # Fill with random if needed
        while len(context) < num_context_programs:
            avail = [p for p in sorted_candidates if p.id != parent.id and p.id not in [c.id for c in context]]
            if not avail:
                break
            context.append(random.choice(avail))

        # Use DIVERGE_LABEL during stagnation
        label = ""
        if self.stagnation_count >= 5:
            label = self.DIVERGE_LABEL
            self.stagnation_count = 0  # Reset after applying

        parent_dict = {label: parent}
        context_programs_dict = {"": context[:num_context_programs]}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END