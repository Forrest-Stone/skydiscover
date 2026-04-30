# EVOLVE-BLOCK-START
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase


@dataclass
class EvolvedProgram(Program):
    pass


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search with stagnation-aware parent selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score: Optional[float] = None
        self.stagnation = 0
        self.parent_use: Dict[str, int] = {}

    def _score(self, p: Program) -> float:
        v = p.metrics.get("combined_score", 0)
        return float(v) if isinstance(v, (int, float)) else 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        s = self._score(program)
        if self.best_score is None:
            self.best_score = max(s, max(self._score(p) for p in self.programs.values()))
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1
        if program.parent_id:
            self.parent_use[program.parent_id] = self.parent_use.get(program.parent_id, 0) + 1
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if num_context_programs is None:
            num_context_programs = 4
        progs = list(self.programs.values())
        if not progs:
            raise ValueError("No candidates")
        if self.best_score is None:
            self.best_score = max(self._score(p) for p in progs)
        progs.sort(key=self._score, reverse=True)
        if self.stagnation > 10:
            return {self.DIVERGE_LABEL: progs[len(progs) // 3]}, {}
        weights = [max(0.1, self._score(p) - 0.05 * self.parent_use.get(p.id, 0)) for p in progs]
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        parent = progs[0]
        for p, w in zip(progs, weights):
            cumsum += w
            if r <= cumsum:
                parent = p
                break
        others = [p for p in progs if p.id != parent.id]
        ctx = others[:2] if others else []
        rest = [p for p in others if p not in ctx]
        if len(ctx) < num_context_programs and rest:
            ctx.extend(random.sample(rest, min(num_context_programs - len(ctx), len(rest))))
        return {"": parent}, {"": ctx}
# EVOLVE-BLOCK-END