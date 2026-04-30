import math
from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, _ROOT / rel_path)
    mod = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_adaptation = _load_module("costada_adaptation", "skydiscover/search/costada/adaptation.py")
_router = _load_module("costada_router", "skydiscover/search/costada/router.py")
try:
    _controller = _load_module("costada_controller", "skydiscover/search/costada/controller.py")
except ModuleNotFoundError:
    _controller = None


def test_costada_utility_uses_budget_gated_normalized_cost():
    ctilde = _adaptation.normalized_cost(step_cost=0.02, ref_cost=0.01, eps_c=0.0)
    denom = _adaptation.cost_denominator(remaining_ratio=0.75, ctilde=ctilde)
    util = _adaptation.utility(
        local_gain_value=0.2,
        global_gain_value=0.05,
        step_cost=0.02,
        remaining_ratio=0.75,
        ref_cost=0.01,
        eps_c=0.0,
    )

    expected_denom = 1.0 + 0.25 * math.log1p(2.0)
    expected_numerator = 0.25 * 0.05 + 0.75 * 0.2

    assert ctilde == pytest.approx(2.0)
    assert denom == pytest.approx(expected_denom)
    assert util == pytest.approx(expected_numerator / expected_denom)


def test_costada_utility_becomes_more_global_late_in_budget():
    early = _adaptation.utility(
        local_gain_value=0.2,
        global_gain_value=0.0,
        step_cost=0.01,
        remaining_ratio=0.9,
        ref_cost=0.01,
        eps_c=0.0,
    )
    late = _adaptation.utility(
        local_gain_value=0.2,
        global_gain_value=0.0,
        step_cost=0.01,
        remaining_ratio=0.1,
        ref_cost=0.01,
        eps_c=0.0,
    )

    assert early > late


def test_router_updates_from_precomputed_routing_reward():
    router = _router.CostAwareFrontierRouter(c_ucb=0.0, gamma=0.5)
    reward = _adaptation.routing_reward(global_gain_value=0.4, denominator=2.0)

    observed = router.update(frontier_id=3, routing_reward_value=reward)

    assert observed == pytest.approx(0.2)
    assert router.get_reward(3) == pytest.approx(0.1)


def test_low_recent_utility_uses_realized_utility_window_not_h_signal():
    if _controller is None:
        pytest.skip("CostAdaController import requires optional runtime dependencies")
    controller = _controller.CostAdaController.__new__(_controller.CostAdaController)
    controller.meta_h_threshold = 0.01
    controller._recent_utility_values = deque([0.0, 0.0, 0.0], maxlen=8)

    assert controller._low_recent_utility() is True

    controller._recent_utility_values = deque([0.0, 0.05], maxlen=8)
    assert controller._low_recent_utility() is False


def test_budget_exhaustion_gate_blocks_only_next_iteration():
    if _controller is None:
        pytest.skip("CostAdaController import requires optional runtime dependencies")
    controller = _controller.CostAdaController.__new__(_controller.CostAdaController)
    controller.budget_ledger = SimpleNamespace(
        config=SimpleNamespace(nominal_budget=1.0, eps=1e-8),
        cumulative_cost=0.999999,
    )

    assert controller._budget_exhausted_before_iteration() is False

    controller.budget_ledger.cumulative_cost = 1.0
    assert controller._budget_exhausted_before_iteration() is True

    controller.budget_ledger.cumulative_cost = 1.000111
    assert controller._budget_exhausted_before_iteration() is True
