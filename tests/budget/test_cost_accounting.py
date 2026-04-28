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
