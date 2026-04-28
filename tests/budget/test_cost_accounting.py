import csv
import json
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_budget_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, _ROOT / rel_path)
    mod = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.modules.setdefault("skydiscover", types.ModuleType("skydiscover"))
sys.modules.setdefault("skydiscover.budget", types.ModuleType("skydiscover.budget"))

_core = _load_budget_module("skydiscover.budget.core", "skydiscover/budget/core.py")
_hooks = _load_budget_module("skydiscover.budget.hooks", "skydiscover/budget/hooks.py")
_io = _load_budget_module("skydiscover.budget.io", "skydiscover/budget/io.py")

BudgetConfig = _core.BudgetConfig
BudgetLedger = _core.BudgetLedger
CallCostRecord = _core.CallCostRecord
CallRole = _core.CallRole
write_iteration_record = _io.write_iteration_record
write_summary = _io.write_summary
export_iterations_csv = _io.export_iterations_csv
export_calls_csv = _io.export_calls_csv
export_summary_csv = _io.export_summary_csv
calls_from_iteration_row = _io.calls_from_iteration_row


def test_iteration_cost_sums_generation_retry_and_guide_with_role_tokens(tmp_path):
    ledger = BudgetLedger(BudgetConfig(nominal_budget=1.0))
    record = ledger.start_iteration(4)
    ledger.add_call(
        record,
        CallCostRecord(
            role=CallRole.GENERATION,
            model_name="m",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            raw_cost=0.001,
        ),
    )
    ledger.add_call(
        record,
        CallCostRecord(
            role=CallRole.RETRY,
            model_name="m",
            prompt_tokens=30,
            completion_tokens=10,
            total_tokens=40,
            raw_cost=0.0004,
        ),
    )
    ledger.add_call(
        record,
        CallCostRecord(
            role=CallRole.GUIDE,
            model_name="g",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            raw_cost=0.002,
        ),
    )

    ledger.finalize_iteration(record)

    assert record.iteration_cost == pytest.approx(0.0034)
    assert record.cumulative_cost == pytest.approx(0.0034)

    summary = ledger.summary()
    assert summary["generation_input_tokens_total"] == 100
    assert summary["generation_output_tokens_total"] == 20
    assert summary["retry_input_tokens_total"] == 30
    assert summary["retry_output_tokens_total"] == 10
    assert summary["guide_input_tokens_total"] == 50
    assert summary["guide_output_tokens_total"] == 25
    assert summary["input_tokens_total"] == 180
    assert summary["output_tokens_total"] == 55

    out = tmp_path / "iterations.jsonl"
    record.meta.update(
        {
            "remaining_budget_ratio_before": 1.0,
            "remaining_budget_ratio_after": record.remaining_budget_ratio,
            "frontier_improvement": 0.5,
            "meta_triggered": True,
        }
    )
    write_iteration_record(out, record)
    row = json.loads(out.read_text(encoding="utf-8").strip())

    assert row["iteration_cost"] == pytest.approx(0.0034)
    assert row["generation_input_tokens"] == 100
    assert row["retry_output_tokens"] == 10
    assert row["guide_input_tokens"] == 50
    assert row["num_generation_calls_this_iteration"] == 1
    assert row["num_retry_calls_this_iteration"] == 1
    assert row["num_guide_calls_this_iteration"] == 1
    assert row["guide_triggered"] is True
    assert row["frontier_improvement"] == 0.5

    iterations_csv = tmp_path / "iterations.csv"
    calls_csv = tmp_path / "calls.csv"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    write_summary(summary_json, ledger, best_score=0.5, extra={"method": "unit"})

    assert export_iterations_csv(out, iterations_csv) is True
    assert export_calls_csv(out, calls_csv) is True
    assert export_summary_csv(summary_json, summary_csv) is True

    with iterations_csv.open(newline="", encoding="utf-8") as f:
        iteration_rows = list(csv.DictReader(f))
    with calls_csv.open(newline="", encoding="utf-8") as f:
        call_rows = list(csv.DictReader(f))
    with summary_csv.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    assert float(iteration_rows[0]["iteration_cost"]) == pytest.approx(0.0034)
    assert len(call_rows) == 3
    assert [r["role"] for r in call_rows] == ["generation", "retry", "guide"]
    assert call_rows[2]["input_tokens"] == "50"
    assert summary_rows[0]["method"] == "unit"
    assert summary_rows[0]["input_tokens_total"] == "180"


def test_calls_from_iteration_row_supports_legacy_arrays():
    calls = calls_from_iteration_row(
        {
            "call_roles": ["generation", "guide"],
            "call_model_names": ["m1", "m2"],
            "call_input_tokens": [10, 20],
            "call_output_tokens": [3, 4],
            "call_total_tokens": [13, 24],
            "call_costs": [0.1, 0.2],
        }
    )

    assert calls == [
        {
            "call_index": 0,
            "role": "generation",
            "model_name": "m1",
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
            "raw_cost": 0.1,
            "call_meta": None,
        },
        {
            "call_index": 1,
            "role": "guide",
            "model_name": "m2",
            "input_tokens": 20,
            "output_tokens": 4,
            "total_tokens": 24,
            "raw_cost": 0.2,
            "call_meta": None,
        },
    ]
