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
    """Adaptive search with stagnation-aware label usage."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score = 0.0
        self.stagnation_count = 0
        self.parent_usage: Dict[str, int] = {}
        self.label_attempts: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add program and track progress state."""
        self.programs[program.id] = program
        
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        
        score = program.metrics.get("combined_score", 0)
        if isinstance(score, (int, float)) and score > self.best_score:
            improvement = float(score) - self.best_score
            if improvement > 0.01:
                self.stagnation_count = 0
            self.best_score = float(score)
        else:
            self.stagnation_count += 1
        
        if self.config.db_path:
            self._save_program(program)
        
        self._update_best_program(program)
        return program.id

    def _get_score(self, program: Program) -> float:
        score = program.metrics.get("combined_score", 0)
        return float(score) if isinstance(score, (int, float)) else 0.0

    def _select_parent(self, candidates: List[Program]) -> Tuple[Program, str]:
        """Select parent with usage penalty; return label."""
        scored = [(p, self._get_score(p)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Weight by score, penalize overused parents
        weights = []
        for p, score in scored:
            usage = self.parent_usage.get(p.id, 0)
            penalty = max(0.1, 1.0 - 0.15 * usage)
            weights.append(max(0.01, score * penalty))
        
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        parent = scored[0][0]
        for (p, _), w in zip(scored, weights):
            cumulative += w
            if r <= cumulative:
                parent = p
                break
        
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        
        # Determine label based on stagnation
        label = ""
        parent_score = self._get_score(parent)
        
        if self.stagnation_count >= 15:
            # Deep stagnation - try divergence on mid-tier programs
            if 0.85 < parent_score < 0.99:
                label_key = f"diverge_{parent.id}"
                attempts = self.label_attempts.get(label_key, 0)
                if attempts < 2:
                    label = self.DIVERGE_LABEL
                    self.label_attempts[label_key] = attempts + 1
        elif self.stagnation_count >= 8:
            # Moderate stagnation - refine near-best
            if parent_score > 0.98:
                label_key = f"refine_{parent.id}"
                attempts = self.label_attempts.get(label_key, 0)
                if attempts < 2:
                    label = self.REFINE_LABEL
                    self.label_attempts[label_key] = attempts + 1
        
        return parent, label

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Sample parent and context with adaptive strategy."""
        candidates = list(self.programs.values())
        
        if not candidates:
            raise ValueError("No candidates available")
        
        parent, label = self._select_parent(candidates)
        
        # Build context: top performers + diverse programs
        scored = [(p, self._get_score(p)) for p in candidates if p.id != parent.id]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        context = []
        used_ids = {parent.id}
        
        # Add top 2 performers
        for p, _ in scored[:2]:
            if p.id not in used_ids:
                context.append(p)
                used_ids.add(p.id)
        
        # Add diverse programs from different score ranges
        random.shuffle(scored)
        for p, _ in scored:
            if len(context) >= num_context_programs:
                break
            if p.id not in used_ids:
                context.append(p)
                used_ids.add(p.id)
        
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END