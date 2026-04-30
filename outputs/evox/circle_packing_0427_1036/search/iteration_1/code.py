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
    """Adaptive search with stagnation detection and diverse selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stagnation_count = 0
        self.parent_usage = {}  # Track usage for diversity

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track search state."""
        self.programs[program.id] = program
        
        # Extract score safely
        score = 0.0
        if program.metrics and "combined_score" in program.metrics:
            val = program.metrics["combined_score"]
            if isinstance(val, (int, float)):
                score = float(val)
        
        # Track meaningful improvements (>0.01)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        
        if self.config.db_path:
            self._save_program(program)
        
        self._update_best_program(program)
        return program.id

    def _get_score(self, program: EvolvedProgram) -> float:
        """Safely extract score."""
        if program.metrics and "combined_score" in program.metrics:
            val = program.metrics["combined_score"]
            if isinstance(val, (int, float)):
                return float(val)
        return 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context based on search state."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")
        
        sorted_by_score = sorted(candidates, key=self._get_score, reverse=True)
        
        # Stuck for 5+ iterations: diverge from best
        if self.stagnation_count >= 5:
            parent = sorted_by_score[0]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}
        
        # Stuck for 3-4 iterations: refine best with context
        if self.stagnation_count >= 3:
            parent = sorted_by_score[0]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            context = [p for p in sorted_by_score[1:num_context_programs+1]]
            return {self.REFINE_LABEL: parent}, {"": context}
        
        # Normal: weighted selection favoring good programs, penalizing overuse
        weights = []
        for p in sorted_by_score:
            score = self._get_score(p)
            usage_penalty = 0.1 * self.parent_usage.get(p.id, 0)
            weights.append(max(0.1, score - usage_penalty))
        
        total = sum(weights)
        probs = [w / total for w in weights] if total > 0 else None
        
        parent = random.choices(sorted_by_score, weights=probs, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Context: diverse programs from different score tiers
        others = [p for p in sorted_by_score if p.id != parent.id]
        context = []
        
        if others:
            n = min(num_context_programs, len(others))
            third = max(1, len(others) // 3)
            
            # Pick from top, middle, and bottom regions for diversity
            regions = [others[:third], others[third:2*third], others[2*third:]]
            for region in regions:
                if region and len(context) < n:
                    context.append(random.choice(region))
            
            # Fill remaining
            remaining = [p for p in others if p not in context]
            while len(context) < n and remaining:
                pick = random.choice(remaining)
                context.append(pick)
                remaining.remove(pick)
        
        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END