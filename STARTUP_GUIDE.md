# SkyDiscover Startup and "Where to Change Code" Guide

This guide explains:

1. **What file starts the app**
2. **How startup flows through the code**
3. **What files you should edit for common changes**
4. **A quick map of important folders/files**

---

## 1) First line of the main function (startup entrypoint)

If you run SkyDiscover from the command line (`skydiscover-run ...`), the startup entrypoint is:

- `skydiscover/cli.py`
- Function: `main()`

The `main()` function calls `asyncio.run(main_async())`, and `main_async()` parses args, loads config, and launches the `Runner`.

How this entrypoint is wired:

- `pyproject.toml` defines the CLI script:
  - `skydiscover-run = "skydiscover.cli:main"`

So the most important “first file” is **`skydiscover/cli.py`**.

---

## 2) Startup sequence (high-level)

When you run a command like:

```bash
skydiscover-run <initial_program> <evaluation_file> -c configs/default.yaml
```

Flow is:

1. **CLI parser** in `skydiscover/cli.py`
   - Reads command-line args (`parse_args()`).
2. **Config loading/overrides** in `skydiscover/config.py`
   - `load_config(...)`
   - `apply_overrides(...)`
3. **Runner initialization** in `skydiscover/runner.py`
   - Builds output folder
   - Loads initial program
   - Creates search database/controller
4. **Discovery loop** in `Runner.run(...)`
   - Evaluates programs
   - Iterates search algorithm
   - Saves checkpoints and best result

---

## 3) Where to edit code (by task)

### A) Change CLI arguments / command behavior
Edit:

- `skydiscover/cli.py`

Typical edits:

- Add new flag in `parse_args()`
- Change startup validation in `main_async()`
- Change top-level run logic

---

### B) Change configuration schema/defaults
Edit:

- `skydiscover/config.py`
- `configs/*.yaml` (template/default config files)

Typical edits:

- Add a new config field (dataclass/model)
- Parse and validate config values
- Add default config values

---

### C) Change run lifecycle/checkpointing/output
Edit:

- `skydiscover/runner.py`

Typical edits:

- Pre/post discovery behavior
- Checkpoint load/save behavior
- Best-program save logic

---

### D) Change search algorithm behavior
Edit:

- `skydiscover/search/route.py` (routing to implementation)
- `skydiscover/search/registry.py` (factory registration)
- `skydiscover/search/default_discovery_controller.py`
- Algorithm-specific modules under `skydiscover/search/...`

Typical edits:

- Add a new search strategy
- Modify candidate generation, ranking, or selection

---

### E) Change evaluation behavior
Edit:

- `skydiscover/evaluation/evaluator.py`
- `skydiscover/evaluation/container_evaluator.py`
- `skydiscover/evaluation/harbor_evaluator.py`
- `skydiscover/evaluation/llm_judge.py`

Typical edits:

- Scoring logic
- Program execution environment
- External evaluator interfaces

---

### F) Change model/provider behavior (LLM)
Edit:

- `skydiscover/llm/base.py`
- `skydiscover/llm/openai.py`
- `skydiscover/llm/llm_pool.py`
- `skydiscover/llm/agentic_generator.py`

Typical edits:

- Add provider support
- Modify generation calls/retries
- Model selection and weighting

---

### G) Change prompts/context building
Edit:

- `skydiscover/context_builder/base.py`
- `skydiscover/context_builder/utils.py`
- `skydiscover/context_builder/human_feedback.py`
- `skydiscover/prompt/*`

Typical edits:

- Prompt format
- Context assembly
- Human feedback ingestion

---

### H) Change user-facing docs/examples
Edit:

- `README.md`
- `configs/README.md`
- `examples/*`
- `docs/*`

---

## 4) Important files and what they do

- `pyproject.toml`
  - Package metadata, dependencies, CLI entrypoints.
- `skydiscover/cli.py`
  - Command entrypoint (`main`) and argument parsing.
- `skydiscover/runner.py`
  - Main orchestration for a discovery run.
- `skydiscover/config.py`
  - Config objects, loading, overrides, output-dir helpers.
- `skydiscover/search/*`
  - Search strategies + controllers + persistence/database logic.
- `skydiscover/evaluation/*`
  - Program evaluation and metric computation.
- `skydiscover/llm/*`
  - LLM provider adapters and generation components.

---

## 5) Practical “I want to update code, where do I start?” checklist

1. Decide your change type (CLI/config/search/evaluator/LLM).
2. Start from the section above and open the listed file.
3. Run tests for the area you changed.
4. Run a smoke command with `skydiscover-run` to verify startup.

If you are unsure, start in this order:

1. `skydiscover/cli.py`
2. `skydiscover/runner.py`
3. `skydiscover/config.py`
4. Specific subsystem folder (`search`, `evaluation`, `llm`)

This order matches the startup path and helps you quickly find where behavior is coming from.

---

## 6) Python API guide: run many benchmarks automatically (no one-by-one manual runs)

If you want to use the Python API and avoid running benchmarks manually one at a time, use a small batch runner script.

### Minimal batch pattern

```python
from pathlib import Path
from skydiscover import run_discovery

# Define each benchmark once
BENCHMARKS = [
    {
        "name": "circle_packing",
        "initial_program": "benchmarks/math/circle_packing/initial_program.py",
        "evaluator": "benchmarks/math/circle_packing/evaluator.py",
        "config": "benchmarks/math/circle_packing/config.yaml",
    },
    {
        "name": "erdos_distance",
        "initial_program": "benchmarks/math/erdos_distance/initial_program.py",
        "evaluator": "benchmarks/math/erdos_distance/evaluator.py",
        "config": "benchmarks/math/erdos_distance/config.yaml",
    },
]

results = []
base_out = Path("runs/batch_math")
base_out.mkdir(parents=True, exist_ok=True)

for task in BENCHMARKS:
    out_dir = base_out / task["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_discovery(
        initial_program=task["initial_program"],
        evaluator=task["evaluator"],
        config=task["config"],
        # Common overrides for all tasks:
        search="evox",
        model="gpt-5",
        iterations=50,
        # Keep outputs so you can inspect checkpoints and best program later:
        output_dir=str(out_dir),
        cleanup=False,
    )

    results.append(
        {
            "task": task["name"],
            "best_score": result.best_score,
            "output_dir": result.output_dir,
        }
    )

# Print simple summary table
for r in results:
    print(f"{r['task']}: score={r['best_score']:.4f}, output={r['output_dir']}")
```

### Notes for reliable batch runs

- Keep **one benchmark spec per dict** (`initial_program`, `evaluator`, `config`).
- Set `cleanup=False` and a per-task `output_dir` so outputs do not overwrite each other.
- Start with small `iterations` (e.g., 10–30) for quick validation.
- If one benchmark needs a different algorithm/model, add optional keys per task and pass them in the loop.

### Optional: per-task override example

```python
search = task.get("search", "evox")
model = task.get("model", "gpt-5")
iterations = task.get("iterations", 50)
```

### Optional: continue when one task fails

Wrap each run in `try/except`, record the error, and continue:

```python
try:
    result = run_discovery(...)
except Exception as exc:
    results.append({"task": task["name"], "error": str(exc)})
    continue
```

This is the easiest way to run a full suite from Python without manually launching every benchmark command.

---

## 7) What is Docker’s role in SkyDiscover?

Docker is used to make benchmark evaluation **reproducible and isolated**.

### Why Docker is used

- Keeps evaluator dependencies isolated from your host machine.
- Makes scores more comparable across different environments.
- Lets benchmarks include system packages, binaries, datasets, and scripts safely.

### When Docker is used

Docker is used when `evaluation_file` points to an evaluator directory containing:

- `Dockerfile`
- `evaluate.sh`

In this mode, SkyDiscover runs candidate solutions inside that containerized evaluator.

### When Docker is NOT required

If you pass a plain Python evaluator file (for example, `evaluator.py` with `evaluate(program_path)`), evaluation runs on your host Python environment and Docker is not required.

### Practical advice

- Use Docker evaluator format for serious benchmarking and reproducibility.
- Use plain Python evaluator format for fast local prototyping.
- If host and Docker results differ, trust the containerized benchmark as the authoritative environment.
