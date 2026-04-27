from __future__ import annotations

from typing import Any

from skydiscover.search.evox.controller import CoEvolutionController


class CoEvolutionBudgetController(CoEvolutionController):
    """EvoX wrapper that ensures final budget plots/statistics are emitted."""

    async def run_discovery(self, *args: Any, **kwargs: Any):
        out = await super().run_discovery(*args, **kwargs)
        self._write_budget_summary()
        return out
