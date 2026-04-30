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
    """Adaptive search with stagnation-aware parent/context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.best_iteration = 0
        self.parent_usage = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track progress state."""
        self.programs[program.id] = program
        self.parent_usage[program.id] = self.parent_usage.get(program.id, 0)

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._get_score(program)
        if score > self.best_score:
            self.best_score = score
            self.best_iteration = program.iteration_found

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract combined_score from program metrics."""
        if program.metrics and "combined_score" in program.metrics:
            val = program.metrics["combined_score"]
            if isinstance(val, (int, float)):
                return float(val)
        return 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context adaptively based on stagnation state."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")

        # Sort programs by score
        scored = [(p, self._get_score(p)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate stagnation (iterations since last improvement)
        max_iter = max((p.iteration_found for p in candidates), default=0)
        current_iter = max(getattr(self, 'last_iteration', 0), max_iter)
        stagnation = current_iter - self.best_iteration
        
        # Heavy stagnation: use DIVERGE on less-used mid-tier program
        if stagnation >= 5:
            mid_tier = scored[len(scored)//3:2*len(scored)//3] or scored[len(scored)//2:]
            mid_sorted = sorted(mid_tier, key=lambda x: self.parent_usage.get(x[0].id, 0))
            if mid_sorted:
                parent = mid_sorted[0][0]
                self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
                return {self.DIVERGE_LABEL: parent}, {}

        # Normal mode: weighted selection from top performers
        top_n = max(2, len(scored) // 3)
        top_tier = scored[:top_n]
        
        # Weight by score / (1 + usage) to balance quality and exploration
        weighted = []
        for p, s in top_tier:
            usage = self.parent_usage.get(p.id, 0)
            weight = (s + 0.1) / (1 + usage)
            weighted.append((p, weight))
        
        total = sum(w for _, w in weighted)
        r = random.random() * total
        cumsum = 0.0
        parent = weighted[0][0]
        for p, w in weighted:
            cumsum += w
            if cumsum >= r:
                parent = p
                break
        
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Context: best performer + diverse lower performers
        context = []
        if scored[0][0].id != parent.id:
            context.append(scored[0][0])
        
        lower_tier = [p for p, _ in scored[len(scored)//2:] if p.id != parent.id]
        if lower_tier:
            context.extend(random.sample(lower_tier, min(2, len(lower_tier))))
        
        return {"": parent}, {"": context[:num_context_programs]}


# EVOLVE-BLOCK-END