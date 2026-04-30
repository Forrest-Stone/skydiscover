import csv
import json
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


def test_read_default_nominal_budget_from_model_pricing(tmp_path):
    config = tmp_path / "model_pricing.yaml"
    config.write_text(
        """
model_pricing:
  m:
    input_price_per_1m: 1

budget_defaults:
  nominal_budget: 1.25  # shared default

budget_comparison:
  budget_multipliers: [0.25, 0.5, 1.0, 1.5, 2.0]

adaevolve_budget:
  nominal_budget: 2.5
""",
        encoding="utf-8",
    )

    assert _mod.read_default_nominal_budget(config) == pytest.approx(1.25)
    assert _mod.read_default_budget_multipliers(config) == pytest.approx(
        [0.25, 0.5, 1.0, 1.5, 2.0]
    )


def test_compute_common_budget_curve_rows_truncates_runs_to_shared_budget(tmp_path):
    runs = [
        {
            "run_dir": str(tmp_path / "run_a"),
            "method": "adaevolve",
            "task_family": "packing",
            "task_name": "case",
            "seed": "1",
            "trace": [
                {
                    "cumulative_cost": 0.2,
                    "best_so_far_objective": 1.0,
                    "best_so_far_combined_score": 0.1,
                },
                {
                    "cumulative_cost": 1.2,
                    "best_so_far_objective": 2.0,
                    "best_so_far_combined_score": 0.2,
                },
            ],
        },
        {
            "run_dir": str(tmp_path / "run_b"),
            "method": "adaevolve",
            "task_family": "packing",
            "task_name": "case",
            "seed": "2",
            "trace": [
                {
                    "cumulative_cost": 0.4,
                    "best_so_far_objective": 3.0,
                    "best_so_far_combined_score": 0.3,
                },
                {
                    "cumulative_cost": 0.8,
                    "best_so_far_objective": 4.0,
                    "best_so_far_combined_score": 0.4,
                },
            ],
        },
    ]

    per_run_rows, aggregate_rows = _mod.compute_common_budget_curve_rows(
        runs,
        common_budget=1.0,
        grid_size=3,
    )

    assert len(per_run_rows) == 6
    at_half = [
        row
        for row in aggregate_rows
        if row["method"] == "adaevolve" and row["budget_fraction"] == pytest.approx(0.5)
    ][0]
    at_full = [
        row
        for row in aggregate_rows
        if row["method"] == "adaevolve" and row["budget_fraction"] == pytest.approx(1.0)
    ][0]

    assert at_half["best_so_far_objective_mean"] == pytest.approx(2.0)
    assert at_half["best_so_far_combined_score_mean"] == pytest.approx(0.2)
    assert at_full["best_so_far_objective_mean"] == pytest.approx(2.5)
    assert at_full["best_so_far_combined_score_mean"] == pytest.approx(0.25)
    assert at_full["best_so_far_objective_n"] == 2


def test_resolve_metric_budgets_prefers_absolute_budgets_over_multipliers():
    assert _mod.resolve_metric_budgets("", "0.5,1.0,1.5", 2.0) == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert _mod.resolve_metric_budgets("0.25,0.75", "0.5,1.0", 2.0) == pytest.approx(
        [0.25, 0.75]
    )


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
    assert (out_dir / "summary_budget_per_run_metrics.csv").exists()
    assert (out_dir / "summary_budget_aggregate_metrics.csv").exists()
    assert (out_dir / "common_budget_per_run.csv").exists()
    assert (out_dir / "common_budget_curves.csv").exists()
    assert not list(out_dir.glob("*.png"))


def test_main_defaults_metric_slices_to_common_budget(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    run_dir = root / "adaevolve" / "case_0101_0000"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "nominal_budget": 100.0,
                "total_cost": 1.2,
                "best_score": 0.2,
                "num_iterations": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "iterations.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "iteration": 1,
                        "cumulative_cost": 0.5,
                        "best_so_far_objective": 0.1,
                        "best_so_far_combined_score": 0.1,
                    }
                ),
                json.dumps(
                    {
                        "iteration": 2,
                        "cumulative_cost": 1.2,
                        "best_so_far_objective": 0.2,
                        "best_so_far_combined_score": 0.2,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "aggregate"
    monkeypatch.setattr(_mod, "_load_plt", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_budget_curves.py",
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
            "--common-budget",
            "1.0",
            "--budget-multipliers",
            "0.5,1.0",
        ],
    )

    _mod.main()

    with (out_dir / "per_run_metrics.csv").open(newline="", encoding="utf-8") as f:
        per_run_rows = list(csv.DictReader(f))
    with (out_dir / "summary_budget_per_run_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        summary_budget_rows = list(csv.DictReader(f))

    assert len(per_run_rows) == 2
    assert float(per_run_rows[0]["budget"]) == pytest.approx(0.5)
    assert float(per_run_rows[0]["BudgetMultiplier"]) == pytest.approx(0.5)
    assert float(per_run_rows[0]["BestObjective@Budget"]) == pytest.approx(0.1)
    assert float(per_run_rows[1]["budget"]) == pytest.approx(1.0)
    assert float(per_run_rows[1]["BudgetMultiplier"]) == pytest.approx(1.0)
    assert float(per_run_rows[1]["BestObjective@Budget"]) == pytest.approx(0.1)
    assert float(per_run_rows[1]["OOBRate"]) == pytest.approx(1.0)
    assert float(per_run_rows[1]["OvershootRatio"]) == pytest.approx(0.2)

    assert len(summary_budget_rows) == 1
    assert float(summary_budget_rows[0]["budget"]) == pytest.approx(100.0)
    assert float(summary_budget_rows[0]["OOBRate"]) == pytest.approx(0.0)
