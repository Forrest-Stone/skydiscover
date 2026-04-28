from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_METRICS_PATH = Path(__file__).resolve().parents[2] / "skydiscover" / "budget" / "metrics.py"
_spec = spec_from_file_location("budget_metrics_best_iteration", _METRICS_PATH)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

best_so_far_with_iteration = _mod.best_so_far_with_iteration
rounded_metric_value = _mod.rounded_metric_value


def test_rounded_display_tie_keeps_first_iteration_but_updates_raw_best():
    value, iteration = best_so_far_with_iteration(1.23491, 3, 1.23494, 5)

    assert rounded_metric_value(value) == 1.2349
    assert value == 1.23494
    assert iteration == 3


def test_rounded_display_improvement_updates_iteration():
    value, iteration = best_so_far_with_iteration(1.23494, 3, 1.23516, 5)

    assert rounded_metric_value(value) == 1.2352
    assert value == 1.23516
    assert iteration == 5


def test_missing_candidate_keeps_previous_best():
    value, iteration = best_so_far_with_iteration(2.0, 7, None, 9)

    assert value == 2.0
    assert iteration == 7
