from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_OBJECTIVE_PATH = Path(__file__).resolve().parents[2] / "skydiscover" / "budget" / "objective.py"
_spec = spec_from_file_location("budget_objective", _OBJECTIVE_PATH)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["budget_objective"] = _mod
_spec.loader.exec_module(_mod)

resolve_objective_from_metrics = _mod.resolve_objective_from_metrics


def test_resolve_objective_keeps_target_ratio_separate():
    snapshot = resolve_objective_from_metrics(
        {
            "combined_score": 0.75,
            "target_ratio": 0.75,
            "sum_radii": 2.1,
            "validity": 1.0,
        }
    )

    assert snapshot.objective_key == "sum_radii"
    assert snapshot.objective_value == 2.1
    assert snapshot.target_ratio == 0.75
    assert snapshot.combined_score == 0.75


def test_resolve_objective_accepts_target_value_alias():
    snapshot = resolve_objective_from_metrics(
        {
            "combined_score": 0.5,
            "target_value": 10.0,
            "ratio": 0.5,
            "objective": 5.0,
        }
    )

    assert snapshot.objective_key == "objective"
    assert snapshot.target_value == 10.0
    assert snapshot.target_ratio == 0.5
