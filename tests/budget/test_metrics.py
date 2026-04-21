from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_METRICS_PATH = Path(__file__).resolve().parents[2] / "skydiscover" / "budget" / "metrics.py"
_spec = spec_from_file_location("budget_metrics", _METRICS_PATH)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

avg_cost = _mod.avg_cost
speedup_at_target = _mod.speedup_at_target


def test_avg_cost_basic():
    assert avg_cost([1.0, 2.0, 3.0]) == 2.0


def test_avg_cost_empty():
    assert avg_cost([]) == 0.0


def test_speedup_at_target():
    assert speedup_at_target(2.0, 6.0) == 3.0


def test_speedup_at_target_invalid_inputs():
    assert speedup_at_target(None, 6.0) is None
    assert speedup_at_target(2.0, None) is None
    assert speedup_at_target(0.0, 6.0) is None
