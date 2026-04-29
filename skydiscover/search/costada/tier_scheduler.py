"""Deprecated compatibility helpers for the older CostAda tier scheduler.

Current CostAda local control uses the continuous intensity directly to sample
the local search mode.  This module is kept only for older imports and
checkpoint/tooling compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from skydiscover.search.costada.state import CompactControlState


@dataclass
class TierDecision:
    """Debug-friendly decision bundle for one tier selection."""

    intensity: float
    base_tier: str
    final_tier: str
    override_reason: str = ""


class TierScheduler:
    """Deterministic mapping: H -> intensity -> tier with budget overrides."""

    TIERS = ("cheap", "standard", "rich")

    def __init__(
        self,
        *,
        intensity_min: float = 0.15,
        intensity_max: float = 0.5,
        tau_1: float = 0.24,
        tau_2: float = 0.38,
        eta_low: float = 0.12,
        rich_enable_min_budget: float = 0.28,
        stagnation_threshold: int = 8,
        low_signal_threshold: float = 0.01,
        eps: float = 1e-8,
    ):
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.tau_1 = float(tau_1)
        self.tau_2 = float(tau_2)
        self.eta_low = float(eta_low)
        self.rich_enable_min_budget = float(rich_enable_min_budget)
        self.stagnation_threshold = int(stagnation_threshold)
        self.low_signal_threshold = float(low_signal_threshold)
        self.eps = float(eps)
        self._last_decision: TierDecision | None = None

    def compute_intensity(self, frontier_signal: float) -> float:
        """Compute I_t = I_min + (I_max - I_min) / (1 + sqrt(H + eps))."""
        H = max(float(frontier_signal), 0.0)
        return self.intensity_min + (self.intensity_max - self.intensity_min) / (
            1.0 + math.sqrt(H + self.eps)
        )

    def base_tier_from_intensity(self, intensity: float) -> str:
        """Map continuous intensity to discrete base tier."""
        if intensity < self.tau_1:
            return "cheap"
        if intensity < self.tau_2:
            return "standard"
        return "rich"

    def apply_budget_override(self, state: CompactControlState, base_tier: str) -> tuple[str, str]:
        """Apply budget-aware deterministic overrides.

        Rules:
        - if remaining budget <= eta_low: force cheap
        - else if budget sufficient and frontier stagnant and signal low: allow rich
        - otherwise keep base tier
        """
        rho = float(state.remaining_budget_ratio)
        stagnant = int(state.stagnation_steps) >= self.stagnation_threshold
        low_signal = float(state.frontier_signal) <= self.low_signal_threshold

        if rho <= self.eta_low:
            return "cheap", "low_budget_force_cheap"

        if rho >= self.rich_enable_min_budget and stagnant and low_signal:
            return "rich", "stagnation_rich_override"

        return base_tier, ""

    def select(self, state: CompactControlState) -> str:
        """Deterministically select a tier from compact state."""
        intensity = self.compute_intensity(state.frontier_signal)
        base_tier = self.base_tier_from_intensity(intensity)
        final_tier, reason = self.apply_budget_override(state, base_tier)
        self._last_decision = TierDecision(
            intensity=float(intensity),
            base_tier=base_tier,
            final_tier=final_tier,
            override_reason=reason,
        )
        return final_tier

    def update(self, state: CompactControlState, chosen_tier: str, realized_utility: float) -> None:
        """Compatibility no-op: deterministic policy has no bandit table updates."""
        _ = (state, chosen_tier, realized_utility)

    def last_decision(self) -> TierDecision | None:
        """Return the most recent decision metadata (for debugging/logging)."""
        return self._last_decision
