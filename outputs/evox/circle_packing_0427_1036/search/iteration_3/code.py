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
    """Adaptive search with stagnation detection and diverse sampling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.parent_usage: Dict[str, int] = {}
        self.stagnation: int = 0
        self.best_score: float = 0.0
        self.diverge_used: set = set()
        self.refine_used: set = set()
        self._initialized: bool = False

    def _init_from_programs(self) -> None:
        """Initialize state from existing programs."""
        if self._initialized or not self.programs:
            return
        self._initialized = True
        
        for p in self.programs.values():
            score = p.metrics.get("combined_score", 0)
            if isinstance(score, (int, float)):
                self.best_score = max(self.best_score, float(score))
            if p.parent_id:
                self.parent_usage[p.parent_id] = self.parent_usage.get(p.parent_id, 0) + 1

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and update tracking."""
        self._init_from_programs()
        
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1
        
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)):
            score = float(score)
        else:
            score = 0.0
        
        # Track meaningful improvement (> 0.01)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation = 0
            self.diverge_used.clear()
            self.refine_used.clear()
        else:
            self.stagnation += 1
        
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        
        logger.debug(f"Added program {program.id}")
        return program.id

    def _get_score(self, p: EvolvedProgram) -> float:
        """Safely extract score from program."""
        score = p.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent and context adaptively."""
        self._init_from_programs()
        
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available")
        
        sorted_candidates = sorted(candidates, key=self._get_score, reverse=True)
        
        parent = None
        label = ""
        
        # Deep stagnation: DIVERGE on best unused program
        if self.stagnation >= 5:
            for p in sorted_candidates[:5]:
                if p.id not in self.diverge_used:
                    parent = p
                    label = self.DIVERGE_LABEL
                    self.diverge_used.add(p.id)
                    break
        
        # Moderate stagnation: REFINE on best unused program
        if parent is None and self.stagnation >= 3:
            for p in sorted_candidates[:5]:
                if p.id not in self.refine_used:
                    parent = p
                    label = self.REFINE_LABEL
                    self.refine_used.add(p.id)
                    break
        
        # Default: weighted selection favoring high scores, penalizing overuse
        if parent is None:
            weights = []
            for p in candidates:
                s = self._get_score(p)
                usage = self.parent_usage.get(p.id, 0)
                weights.append(max(0.01, s) / (1 + usage * 0.5))
            
            total = sum(weights)
            if total > 0:
                r = random.random() * total
                cum = 0.0
                for i, p in enumerate(candidates):
                    cum += weights[i]
                    if cum >= r:
                        parent = p
                        break
            
            if parent is None:
                parent = random.choice(candidates)
        
        # Context: diverse from score tiers (only for default sampling)
        context = []
        if label == "":
            n = len(sorted_candidates)
            thirds = [sorted_candidates[:max(1, n//3)], 
                      sorted_candidates[max(1, n//3):max(2, 2*n//3)], 
                      sorted_candidates[max(2, 2*n//3):]]
            
            for tier in thirds:
                if tier:
                    choices = [p for p in tier if p.id != parent.id]
                    if choices:
                        context.append(random.choice(choices))
            
            remaining = num_context_programs - len(context)
            if remaining > 0:
                others = [p for p in candidates if p.id != parent.id and p not in context]
                if others:
                    context.extend(random.sample(others, min(remaining, len(others))))
        
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END