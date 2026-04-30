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
    """Strategic search for plateau breakthrough."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stagnation_count = 0
        self.parent_usage: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            if score > self.best_score + 0.01:
                self.stagnation_count = 0
                self.best_score = float(score)
            else:
                self.stagnation_count += 1
        
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1
        
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        
        # Weighted parent selection: favor high scores, penalize overuse
        def weight(p):
            score = p.metrics.get("combined_score", 0)
            if not isinstance(score, (int, float)):
                score = 0
            penalty = self.parent_usage.get(p.id, 0) * 0.15
            return max(0.1, float(score) - penalty)
        
        weights = [weight(p) for p in candidates]
        parent = random.choices(candidates, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Use REFINE_LABEL for top-tier parents when stagnating
        parent_score = parent.metrics.get("combined_score", 0)
        label = ""
        if isinstance(parent_score, (int, float)):
            if float(parent_score) > 0.95 and self.stagnation_count >= 5:
                label = self.REFINE_LABEL
        
        # Context: top performers + diverse scores
        sorted_progs = sorted(candidates, key=lambda p: p.metrics.get("combined_score", 0) if isinstance(p.metrics.get("combined_score", 0), (int, float)) else 0, reverse=True)
        context = []
        seen_scores = set()
        for p in sorted_progs:
            if p.id == parent.id:
                continue
            score = p.metrics.get("combined_score", 0)
            if isinstance(score, (int, float)):
                if score not in seen_scores or len(context) < 2:
                    context.append(p)
                    seen_scores.add(score)
            if len(context) >= num_context_programs:
                break
        
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END