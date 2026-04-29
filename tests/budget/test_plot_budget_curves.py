import csv
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_budget_curves.py"
_spec = spec_from_file_location("plot_budget_curves_script", _SCRIPT_PATH)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["plot_budget_curves_script"] = _mod
_spec.loader.exec_module(_mod)


def test_write_all_calls_csv_includes_call_level_cumulative_cost(tmp_path):
    out_csv = tmp_path / "all_calls.csv"
    runs = [
        {
            "run_dir": str(tmp_path / "run"),
            "method": "unit",
            "task_family": "budget",
            "task_name": "trace",
            "seed": "1",
            "trace": [
                {
                    "iteration": 1,
                    "iteration_cost": 0.3,
                    "cumulative_cost": 0.3,
                    "calls": [
                        {
                            "role": "generation",
                            "model_name": "m",
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                            "raw_cost": 0.1,
                        },
                        {
                            "role": "retry",
                            "model_name": "m",
                            "input_tokens": 11,
                            "output_tokens": 3,
                            "total_tokens": 14,
                            "raw_cost": 0.2,
                        },
                    ],
                },
                {
                    "iteration": 2,
                    "iteration_cost": 0.4,
                    "cumulative_cost": 0.7,
                    "calls": [
                        {
                            "role": "generation",
                            "model_name": "m",
                            "input_tokens": 12,
                            "output_tokens": 4,
                            "total_tokens": 16,
                            "raw_cost": 0.4,
                        },
                    ],
                },
            ],
        }
    ]

    _mod.write_all_calls_csv(runs, out_csv)

    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert float(rows[0]["cumulative_cost_after_call"]) == pytest.approx(0.1)
    assert float(rows[1]["cumulative_cost_before_call"]) == pytest.approx(0.1)
    assert float(rows[1]["cumulative_cost_after_call"]) == pytest.approx(0.3)
    assert float(rows[2]["cumulative_cost_before_iteration"]) == pytest.approx(0.3)
    assert float(rows[2]["cumulative_cost_before_call"]) == pytest.approx(0.3)
    assert float(rows[2]["cumulative_cost_after_call"]) == pytest.approx(0.7)


def test_main_skips_plotting_when_no_runs(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "aggregate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_budget_curves.py",
            "--root",
            str(tmp_path / "missing_outputs"),
            "--out-dir",
            str(out_dir),
        ],
    )

    _mod.main()

    captured = capsys.readouterr()
    assert "Found runs: 0" in captured.out
    assert "Skipped plotting (no runs found)." in captured.out
    assert (out_dir / "runs.csv").exists()
    assert not list(out_dir.glob("*.png"))
