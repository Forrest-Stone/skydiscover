"""CostAda context builder.

CostAda currently keeps prompt construction identical to AdaEvolve so budget
experiments do not change prompt verbosity or context shape.
"""

from __future__ import annotations

from skydiscover.context_builder.adaevolve.builder import AdaEvolveContextBuilder


class CostAdaContextBuilder(AdaEvolveContextBuilder):
    """Alias of AdaEvolveContextBuilder for prompt parity."""

    pass
