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
    """Simple search with strategic stagnation handling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stagnation = 0
        self.parent_counts = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track stagnation."""
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        
        # Track parent usage
        if program.parent_id:
            self.parent_counts[program.parent_id] = self.parent_counts.get(program.parent_id, 0) + 1
        
        # Update stagnation tracking
        score = program.metrics.get("combined_score", 0.0)
        if isinstance(score, (int, float)):
            if score > self.best_score + 0.01:
                self.stagnation = 0
                self.best_score = float(score)
            else:
                self.stagnation += 1
        
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent strategically based on stagnation state."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")
        
        # Get scores and sort
        scored = []
        for p in candidates:
            s = p.metrics.get("combined_score", 0.0)
            s = float(s) if isinstance(s, (int, float)) else 0.0
            scored.append((p, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Deep stagnation: DIVERGE from underused mid-tier program
        if self.stagnation >= 10:
            mid_start = len(scored) // 4
            mid_end = len(scored) * 3 // 4
            mid_tier = scored[mid_start:mid_end]
            # Find least used
            mid_tier.sort(key=lambda x: self.parent_counts.get(x[0].id, 0))
            if mid_tier:
                return {self.DIVERGE_LABEL: mid_tier[0][0]}, {}
        
        # Moderate stagnation: REFINE best unused program
        if self.stagnation >= 5 and random.random() < 0.3:
            scored.sort(key=lambda x: self.parent_counts.get(x[0].id, 0))
            for p, s in scored:
                if s >= self.best_score * 0.95:
                    return {self.REFINE_LABEL: p}, {}
        
        # Normal: weighted selection favoring good scores, penalizing overuse
        weights = []
        for p, s in scored:
            use_penalty = 0.1 * self.parent_counts.get(p.id, 0)
            w = max(0.01, s - use_penalty)
            weights.append(w)
        
        total = sum(weights)
        r = random.random() * total
        cumsum = 0.0
        parent = scored[0][0]
        for (p, _), w in zip(scored, weights):
            cumsum += w
            if cumsum >= r:
                parent = p
                break
        
        # Context: one from each score quartile
        context = []
        n = len(scored)
        for i in range(4):
            start = i * n // 4
            end = (i + 1) * n // 4
            tier = scored[start:end]
            if tier:
                p = random.choice(tier)[0]
                if p.id != parent.id:
                    context.append(p)
        
        return {"": parent}, {"": context[:num_context_programs]}


# EVOLVE-BLOCK-END