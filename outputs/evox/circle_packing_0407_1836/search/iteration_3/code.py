# EVOLVE-BLOCK-START
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

@dataclass
class EvolvedProgram(Program):
    pass

class EvolvedProgramDatabase(ProgramDatabase):
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.usage = {}
        self.stagnant = 0
        self.best = 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        s = program.metrics.get("combined_score", 0)
        if isinstance(s, (int, float)) and s > self.best + 0.01:
            self.stagnant, self.best = 0, s
        else:
            self.stagnant += 1
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        progs = list(self.programs.values())
        if not progs:
            raise ValueError("No programs to sample")
        progs.sort(key=lambda p: -float(p.metrics.get("combined_score", 0) or 0))
        
        # Stagnation: use DIVERGE_LABEL on mid-tier program for exploration
        if self.stagnant > 10 and len(progs) > 3:
            mid = progs[len(progs)//3]
            self.usage[mid.id] = self.usage.get(mid.id, 0) + 1
            return {self.DIVERGE_LABEL: mid}, {"": []}
        
        # Normal: weighted selection favoring high scores, penalizing overuse
        w = [max(0.1, (p.metrics.get("combined_score", 0) or 0) - 0.1*self.usage.get(p.id,0)) for p in progs]
        parent = random.choices(progs, weights=w, k=1)[0]
        self.usage[parent.id] = self.usage.get(parent.id, 0) + 1
        
        # Context: top 2 performers + random diverse programs
        ctx = [p for p in progs[:2] if p.id != parent.id]
        rest = [p for p in progs if p.id != parent.id and p not in ctx]
        if rest:
            ctx += random.sample(rest, min(num_context_programs-len(ctx), len(rest)))
        return {"": parent}, {"": ctx}
# EVOLVE-BLOCK-END