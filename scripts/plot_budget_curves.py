"""Aggregate budget traces and compute reusable evaluation metrics/plots.

Usage:
  python scripts/plot_budget_curves.py --root outputs \
      --out-dir outputs/aggregate_budget \
      --budgets 0.5,1.0,2.0 --target 0.92 --baselines adaevolve,evox
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


def best_score_at_budget(points: Iterable[tuple[float, float]], budget: float) -> float | None:
    best = None
    for cost, score in points:
        if cost <= budget:
            best = score if best is None else max(best, score)
    return best


def success_at_target(best_score: Optional[float], target: float) -> float:
    if best_score is None:
        return 0.0
    return 1.0 if float(best_score) >= float(target) else 0.0


def cost_to_target(points: Iterable[tuple[float, float]], target: float) -> float | None:
    for cost, score in points:
        try:
            if score is not None and float(score) >= float(target):
                return float(cost)
        except (TypeError, ValueError):
            continue
    return None


def overshoot_ratio(total_cost: float, nominal_budget: float) -> float:
    if nominal_budget <= 0:
        return 0.0
    return max((total_cost - nominal_budget) / nominal_budget, 0.0)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def infer_method(run_dir: Path) -> str:
    known = [
        "costada",
        "adaevolve",
        "evox",
        "budgetevolve",
        "topk",
        "best_of_n",
    ]
    parts = [p.lower() for p in run_dir.parts]
    for name in known:
        if name in parts:
            return name
    return run_dir.parent.name.lower() if run_dir.parent.name else "unknown"


def collect_runs(root: Path) -> List[Dict]:
    runs: List[Dict] = []
    for summary_path in root.rglob("summary.json"):
        run_dir = summary_path.parent
        iter_path = run_dir / "iterations.jsonl"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trace = load_jsonl(iter_path)
        points = []
        tiers = []
        meta_flags = []
        for row in trace:
            cost = float(row.get("cumulative_cost", 0.0) or 0.0)
            score = row.get("global_best_after")
            points.append((cost, score))
            tier = row.get("tier")
            if tier:
                tiers.append(str(tier))
            meta_flags.append(1.0 if row.get("meta_triggered") else 0.0)
        runs.append(
            {
                "run_dir": str(run_dir),
                "method": infer_method(run_dir),
                "task_family": infer_task_family(run_dir),
                "task_name": infer_task_name(run_dir),
                "seed": infer_seed(summary, run_dir),
                "summary": summary,
                "trace": trace,
                "points": points,
                "tiers": tiers,
                "meta_trigger_rate": (sum(meta_flags) / len(meta_flags)) if meta_flags else 0.0,
            }
        )
    return runs


def infer_task_family(run_dir: Path) -> str:
    """Best-effort task-family inference from output path shape."""
    parts = [p for p in run_dir.parts]
    method = infer_method(run_dir)
    lowered = [p.lower() for p in parts]
    if method in lowered:
        idx = lowered.index(method)
        # outputs/<method>/<task_name_...>
        if idx + 2 < len(parts):
            return parts[idx - 1] if idx > 0 else "unknown"
    return run_dir.parent.parent.name if run_dir.parent.parent.name else "unknown"


def infer_task_name(run_dir: Path) -> str:
    """Infer task name from run directory name (strip trailing timestamp when possible)."""
    name = run_dir.name
    # Matches common suffixes like _MMDD_HHMM
    m = re.match(r"^(.*)_\d{4}_\d{4}$", name)
    return m.group(1) if m else name


def infer_seed(summary: Dict, run_dir: Path) -> str:
    """Extract seed from summary if present, else parse from folder name."""
    if "seed" in summary:
        return str(summary.get("seed"))
    m = re.search(r"(?:seed|s)(\d+)", run_dir.name.lower())
    return m.group(1) if m else ""


def write_runs_csv(runs: List[Dict], out_csv: Path, target: Optional[float] = None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_dir",
        "method",
        "task_family",
        "task_name",
        "seed",
        "nominal_budget",
        "total_cost",
        "oob",
        "overshoot",
        "overshoot_ratio",
        "best_score",
        "success_target",
        "cost_to_target",
        "num_iterations",
        "num_generation_calls",
        "num_retry_calls",
        "num_guide_calls",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for run in runs:
            s = run["summary"]
            writer.writerow(
                {
                    "run_dir": run["run_dir"],
                    "method": run["method"],
                    "task_family": run.get("task_family", "unknown"),
                    "task_name": run.get("task_name", ""),
                    "seed": run.get("seed", ""),
                    "nominal_budget": s.get("nominal_budget"),
                    "total_cost": s.get("total_cost"),
                    "oob": s.get("oob"),
                    "overshoot": s.get("overshoot"),
                    "overshoot_ratio": s.get("overshoot_ratio"),
                    "best_score": s.get("best_score"),
                    "success_target": (
                        success_at_target(s.get("best_score"), target) if target is not None else ""
                    ),
                    "cost_to_target": (
                        cost_to_target(run["points"], target) if target is not None else ""
                    ),
                    "num_iterations": s.get("num_iterations"),
                    "num_generation_calls": s.get("num_generation_calls"),
                    "num_retry_calls": s.get("num_retry_calls"),
                    "num_guide_calls": s.get("num_guide_calls"),
                }
            )


def compute_metric_rows(runs: List[Dict], budgets: List[float], target: Optional[float]) -> List[Dict]:
    rows: List[Dict] = []
    for run in runs:
        summary = run["summary"]
        total_cost = float(summary.get("total_cost", 0.0) or 0.0)
        nominal = float(summary.get("nominal_budget", 0.0) or 0.0)
        points: Iterable[Tuple[float, float]] = run["points"]
        tier_entropy = compute_tier_entropy(run.get("tiers", []))
        meta_trigger_rate = float(run.get("meta_trigger_rate", 0.0) or 0.0)

        for b in budgets:
            bscore = best_score_at_budget(points, b)
            success = success_at_target(bscore, target) if target is not None else ""
            c2t = cost_to_target(points, target) if target is not None else ""
            rows.append(
                {
                    "run_dir": run["run_dir"],
                    "method": run["method"],
                    "task_family": run.get("task_family", "unknown"),
                    "task_name": run.get("task_name", ""),
                    "seed": run.get("seed", ""),
                    "budget": b,
                    "BestScore@Budget": bscore if bscore is not None else "",
                    "Success@Target": success,
                    "Cost-to-Target": c2t if c2t is not None else "",
                    "AvgCost": total_cost,
                    "OOBRate": 1.0 if total_cost > b else 0.0,
                    "OvershootRatio": overshoot_ratio(total_cost, b),
                    "TierEntropy": tier_entropy,
                    "MetaTriggerRate": meta_trigger_rate,
                    "SummaryNominalBudget": nominal,
                }
            )
    return rows


def aggregate_metric_rows(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, float], List[Dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], float(r["budget"]))].append(r)

    out: List[Dict] = []
    for (method, budget), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        def _vals(key: str) -> List[float]:
            vals = []
            for it in items:
                v = it.get(key)
                if v == "" or v is None:
                    continue
                vals.append(float(v))
            return vals

        best_vals = _vals("BestScore@Budget")
        success_vals = _vals("Success@Target")
        c2t_vals = _vals("Cost-to-Target")
        avg_cost_vals = _vals("AvgCost")
        oob_vals = _vals("OOBRate")
        over_vals = _vals("OvershootRatio")
        tier_entropy_vals = _vals("TierEntropy")
        meta_trigger_vals = _vals("MetaTriggerRate")

        out.append(
            {
                "method": method,
                "budget": budget,
                "BestScore@Budget": mean(best_vals) if best_vals else "",
                "Success@Target": mean(success_vals) if success_vals else "",
                "Cost-to-Target": mean(c2t_vals) if c2t_vals else "",
                "AvgCost": mean(avg_cost_vals) if avg_cost_vals else "",
                "OOBRate": mean(oob_vals) if oob_vals else "",
                "OvershootRatio": mean(over_vals) if over_vals else "",
                "TierEntropy": mean(tier_entropy_vals) if tier_entropy_vals else "",
                "MetaTriggerRate": mean(meta_trigger_vals) if meta_trigger_vals else "",
                "num_runs": len(items),
            }
        )
    return out


def compute_tier_entropy(tiers: List[str]) -> float:
    """Compute entropy of tier usage distribution for one run."""
    if not tiers:
        return 0.0
    counts: Dict[str, int] = defaultdict(int)
    for t in tiers:
        counts[t] += 1
    n = len(tiers)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log(max(p, 1e-12))
    return entropy


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_plt():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_best_score_vs_cost(runs: List[Dict], out_png: Path) -> bool:
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for run in runs:
        trace = run["trace"]
        if not trace:
            continue
        x = [float(row.get("cumulative_cost", 0.0) or 0.0) for row in trace]
        y = [row.get("global_best_after") for row in trace]
        if all(v is None for v in y):
            continue
        plt.plot(x, y, linewidth=1.2, alpha=0.8, label=run["method"])
    plt.xlabel("Cumulative cost (USD)")
    plt.ylabel("Best score")
    plt.title("Best-so-far score vs cumulative cost")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def plot_success_vs_budget(agg_rows: List[Dict], out_png: Path) -> bool:
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in agg_rows:
        v = row.get("Success@Target")
        if v == "" or v is None:
            continue
        by_method[row["method"]].append((float(row["budget"]), float(v)))
    if not by_method:
        return False

    plt.figure(figsize=(7, 5))
    for method, pts in by_method.items():
        pts.sort(key=lambda x: x[0])
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    plt.xlabel("Budget")
    plt.ylabel("Success@Target")
    plt.title("Success@Target vs budget")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def plot_cost_to_target(agg_rows: List[Dict], out_png: Path) -> bool:
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in agg_rows:
        v = row.get("Cost-to-Target")
        if v == "" or v is None:
            continue
        by_method[row["method"]].append((float(row["budget"]), float(v)))
    if not by_method:
        return False

    plt.figure(figsize=(7, 5))
    for method, pts in by_method.items():
        pts.sort(key=lambda x: x[0])
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    plt.xlabel("Budget")
    plt.ylabel("Cost-to-Target (USD)")
    plt.title("Cost-to-Target vs budget")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def plot_budget_adherence(agg_rows: List[Dict], out_png: Path) -> bool:
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: {"oob": [], "over": []})
    for row in agg_rows:
        by_method[row["method"]]["oob"].append((float(row["budget"]), float(row.get("OOBRate") or 0.0)))
        by_method[row["method"]]["over"].append((float(row["budget"]), float(row.get("OvershootRatio") or 0.0)))

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for method, d in by_method.items():
        o = sorted(d["oob"], key=lambda x: x[0])
        ov = sorted(d["over"], key=lambda x: x[0])
        axes[0].plot([x for x, _ in o], [y for _, y in o], marker="o", label=method)
        axes[1].plot([x for x, _ in ov], [y for _, y in ov], marker="o", label=method)

    axes[0].set_ylabel("OOBRate")
    axes[0].set_title("Budget adherence")
    axes[1].set_ylabel("OvershootRatio")
    axes[1].set_xlabel("Budget")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def plot_tier_usage(runs: List[Dict], out_png: Path) -> bool:
    """Plot aggregate tier usage per method."""
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)

    by_method: Dict[str, Dict[str, int]] = defaultdict(lambda: {"cheap": 0, "standard": 0, "rich": 0})
    for run in runs:
        method = run["method"]
        for t in run.get("tiers", []):
            if t in by_method[method]:
                by_method[method][t] += 1
    if not by_method:
        return False

    methods = sorted(by_method.keys())
    x = range(len(methods))
    cheap = [by_method[m]["cheap"] for m in methods]
    standard = [by_method[m]["standard"] for m in methods]
    rich = [by_method[m]["rich"] for m in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(x, cheap, label="cheap")
    plt.bar(x, standard, bottom=cheap, label="standard")
    plt.bar(x, rich, bottom=[cheap[i] + standard[i] for i in range(len(methods))], label="rich")
    plt.xticks(list(x), methods, rotation=25)
    plt.ylabel("Tier usage count")
    plt.title("Tier usage by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def plot_meta_trigger_rate(agg_rows: List[Dict], out_png: Path) -> bool:
    """Plot average meta-trigger rate vs budget by method."""
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in agg_rows:
        v = row.get("MetaTriggerRate")
        if v == "" or v is None:
            continue
        by_method[row["method"]].append((float(row["budget"]), float(v)))
    if not by_method:
        return False

    plt.figure(figsize=(7, 5))
    for method, pts in by_method.items():
        pts.sort(key=lambda x: x[0])
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    plt.xlabel("Budget")
    plt.ylabel("MetaTriggerRate")
    plt.title("Intervention frequency vs budget")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def plot_metric_vs_budget(agg_rows: List[Dict], metric_key: str, title: str, out_png: Path) -> bool:
    """Generic plot helper for budget-sliced aggregated metrics."""
    plt = _load_plt()
    if plt is None:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in agg_rows:
        v = row.get(metric_key)
        if v == "" or v is None:
            continue
        by_method[row["method"]].append((float(row["budget"]), float(v)))
    if not by_method:
        return False

    plt.figure(figsize=(7, 5))
    for method, pts in by_method.items():
        pts.sort(key=lambda x: x[0])
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    plt.xlabel("Budget")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def compute_speedup_vs_baseline(
    per_run_rows: List[Dict], baselines: List[str], target: Optional[float]
) -> List[Dict]:
    """Compute generic pairwise Speedup@Target against one or more baselines.

    Speedup@Target(tau; m, b) =
      median(CostToTarget_b(tau)) / median(CostToTarget_m(tau))

    - Computed on successful runs only (rows with non-empty Cost-to-Target)
    - Uses median for robustness
    - Emits NaN when one side has no successful runs
    """
    if target is None:
        return []

    groups: Dict[Tuple[str, str, float], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in per_run_rows:
        c2t = r.get("Cost-to-Target")
        if c2t == "" or c2t is None:
            continue
        key = (str(r.get("task_family", "all")), str(r.get("task_name", "all")), float(r["budget"]))
        groups[key][str(r["method"])].append(float(c2t))

    out: List[Dict] = []
    for (task_family, task_name, budget), by_method in sorted(groups.items()):
        methods = sorted(by_method.keys())
        for baseline in baselines:
            b_vals = by_method.get(baseline, [])
            b_med = median(b_vals) if b_vals else float("nan")
            for method in methods:
                if method == baseline:
                    continue
                m_vals = by_method.get(method, [])
                m_med = median(m_vals) if m_vals else float("nan")
                speedup = float("nan")
                if not math.isnan(b_med) and not math.isnan(m_med) and m_med > 0:
                    speedup = b_med / m_med
                out.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "task_family": task_family,
                        "task_name": task_name,
                        "nominal_budget": budget,
                        "target": target,
                        "baseline_median_cost_to_target": b_med,
                        "method_median_cost_to_target": m_med,
                        "speedup_target": speedup,
                    }
                )
    return out


def plot_speedup_vs_baseline(speedup_rows: List[Dict], out_png: Path) -> bool:
    """Plot Speedup@Target by method, grouped by baseline."""
    plt = _load_plt()
    if plt is None or not speedup_rows:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, float], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in speedup_rows:
        s = r.get("speedup_target")
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        grouped[(str(r["baseline"]), float(r["nominal_budget"]))][str(r["method"])].append(float(s))
    if not grouped:
        return False

    labels = []
    values = []
    for (baseline, budget), by_method in sorted(grouped.items()):
        for method, vals in sorted(by_method.items()):
            labels.append(f"{method}\nvs {baseline}\nB={budget:g}")
            values.append(mean(vals))

    plt.figure(figsize=(max(8, len(labels) * 0.9), 5))
    plt.bar(range(len(labels)), values)
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Speedup@Target")
    plt.title("Pairwise speedup vs baseline")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def parse_budgets(text: str, fallback: List[float]) -> List[float]:
    if not text:
        return fallback
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(float(chunk))
    return out or fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", help="Root outputs directory")
    parser.add_argument("--out-dir", default="outputs/aggregate_budget", help="Where to write csv/plots")
    parser.add_argument(
        "--budgets",
        default="",
        help="Comma-separated nominal budgets for metric slicing, e.g. 0.5,1.0,2.0",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Target threshold for Success@Target and Cost-to-Target",
    )
    parser.add_argument(
        "--baselines",
        default="",
        help="Comma-separated baseline methods for Speedup@Target, e.g. adaevolve,evox",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    runs = collect_runs(root)
    write_runs_csv(runs, out_dir / "runs.csv", target=args.target)

    default_budgets = sorted(
        {
            float((r["summary"] or {}).get("nominal_budget", 0.0) or 0.0)
            for r in runs
            if float((r["summary"] or {}).get("nominal_budget", 0.0) or 0.0) > 0
        }
    )
    budgets = parse_budgets(args.budgets, default_budgets or [1.0])

    per_run_rows = compute_metric_rows(runs, budgets, args.target)
    agg_rows = aggregate_metric_rows(per_run_rows)

    write_csv(per_run_rows, out_dir / "per_run_metrics.csv")
    write_csv(agg_rows, out_dir / "aggregate_metrics.csv")
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    speedup_rows = compute_speedup_vs_baseline(per_run_rows, baselines, args.target)
    write_csv(speedup_rows, out_dir / "speedup_vs_baseline.csv")

    p1 = plot_best_score_vs_cost(runs, out_dir / "best_score_vs_cost.png")
    p1_alias = plot_best_score_vs_cost(runs, out_dir / "best_vs_cost.png")
    p2 = plot_success_vs_budget(agg_rows, out_dir / "success_vs_budget.png")
    p3 = plot_cost_to_target(agg_rows, out_dir / "cost_to_target.png")
    p4 = plot_budget_adherence(agg_rows, out_dir / "budget_adherence.png")
    p4_alias = plot_metric_vs_budget(agg_rows, "OOBRate", "OOB rate vs budget", out_dir / "oob_vs_budget.png")
    p8 = plot_metric_vs_budget(
        agg_rows, "OvershootRatio", "Overshoot ratio vs budget", out_dir / "overshoot_ratio_vs_budget.png"
    )
    p9 = plot_metric_vs_budget(agg_rows, "AvgCost", "Average cost vs budget", out_dir / "avg_cost_vs_budget.png")
    p5 = plot_tier_usage(runs, out_dir / "tier_usage.png")
    p6 = plot_meta_trigger_rate(agg_rows, out_dir / "meta_trigger_rate.png")
    p7 = plot_speedup_vs_baseline(speedup_rows, out_dir / "speedup_vs_baseline.png")

    print(f"Found runs: {len(runs)}")
    print(f"Wrote: {out_dir / 'runs.csv'}")
    print(f"Wrote: {out_dir / 'per_run_metrics.csv'}")
    print(f"Wrote: {out_dir / 'aggregate_metrics.csv'}")
    print(f"Wrote: {out_dir / 'speedup_vs_baseline.csv'}")
    if not any([p1, p1_alias, p2, p3, p4, p4_alias, p5, p6, p7, p8, p9]):
        print("Skipped plotting (matplotlib unavailable).")
    else:
        if p1:
            print(f"Wrote: {out_dir / 'best_score_vs_cost.png'}")
        if p1_alias:
            print(f"Wrote: {out_dir / 'best_vs_cost.png'}")
        if p2:
            print(f"Wrote: {out_dir / 'success_vs_budget.png'}")
        if p3:
            print(f"Wrote: {out_dir / 'cost_to_target.png'}")
        if p4:
            print(f"Wrote: {out_dir / 'budget_adherence.png'}")
        if p4_alias:
            print(f"Wrote: {out_dir / 'oob_vs_budget.png'}")
        if p8:
            print(f"Wrote: {out_dir / 'overshoot_ratio_vs_budget.png'}")
        if p9:
            print(f"Wrote: {out_dir / 'avg_cost_vs_budget.png'}")
        if p5:
            print(f"Wrote: {out_dir / 'tier_usage.png'}")
        if p6:
            print(f"Wrote: {out_dir / 'meta_trigger_rate.png'}")
        if p7:
            print(f"Wrote: {out_dir / 'speedup_vs_baseline.png'}")


if __name__ == "__main__":
    main()
