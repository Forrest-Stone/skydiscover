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

---

## 8) Q&A（你问的两个点）

### Q1: `/skydiscover/skydiscover/prompt` 文件夹是干啥的？

`skydiscover/prompt` 不是“单独跑逻辑”的模块，它更像是**提示词/上下文构建能力的统一导出入口**：

- 通过 `__init__.py` 把不同算法对应的 ContextBuilder（如 `Default` / `Evox` / `AdaEvolve` / `GEPA`）和 `TemplateManager` 暴露出来。
- 这些组件决定 LLM 在每轮看到什么上下文、用什么模板组织提示词。

所以，如果你要改“提示词策略、上下文拼接、人工反馈注入”，通常从这里关联到的 `context_builder/*` 去改。

### Q2: `skydiscover/evaluation` 下面的文件是不是都要保持一致？

**不是“内容要一致”**，而是“接口风格要一致”。

它们分工不同：

- `evaluator.py`：本地 Python evaluator（`evaluate(program_path)`）
- `container_evaluator.py`：Docker evaluator（`Dockerfile + evaluate.sh`）
- `harbor_evaluator.py`：Harbor 协议任务（`tests/test.sh + reward`）
- `llm_judge.py`：可选 LLM 打分补充器
- `wrapper.py`：把老式 evaluator 包装成容器 JSON 协议

### Q3: 如果我要改进，是否可以不改这些 evaluation 文件？

可以，取决于你改进的目标：

- **只改搜索算法/提示词**：通常不用改 `evaluation/*`。
- **只换 benchmark 题目或评分细节**：通常只改 benchmark 自己的 `evaluator.py` 或 `evaluator/` 目录，不用改框架层 `skydiscover/evaluation/*`。
- **想扩展新的评测协议/执行后端**（例如新容器协议、新 remote sandbox）：才需要改 `skydiscover/evaluation/*`。

一句话：`evaluation/*` 是框架层评测“引擎”，不是每次实验都要改。

---

## 9) Q&A（评测样例、速度、算力、搜索策略怎么改）

### Q1: 评测有没有“测试集样例”？

有，但来源取决于 benchmark：

- 如果是 Harbor/容器 benchmark，测试通常在 benchmark 目录里的 `tests/`（或 evaluator 自带数据）里。
- 如果是 Python evaluator，样例/数据通常由该 `evaluator.py` 自己读取。

框架层 `skydiscover` 不强制统一“样例文件名”，而是通过 evaluator 接口统一执行。

### Q2: 运行是不是非常快？要不要很多计算资源？

不一定，**差异很大**：

- 轻量 Python evaluator 可能很快。
- Docker/Harbor 任务可能慢很多（容器内测试、数据加载、编译/运行开销）。
- 搜索迭代会重复调用评测，迭代数越大、并发越高，总成本越高。

所以它不只是“单纯跑一下代码”，而是“多轮生成 + 多轮评测”的优化循环。

### Q3: 搜索策略要在哪里修改？

按“改动范围”分三层看：

1. **接入层（路由/注册）**
   - `skydiscover/search/route.py`：把 search type 映射到 controller/database。
   - `skydiscover/search/registry.py`：注册与工厂创建逻辑。

2. **通用循环层（默认策略框架）**
   - `skydiscover/search/default_discovery_controller.py`：默认迭代流程（采样→生成→评测→入库）。

3. **算法实现层（你真正要改逻辑的地方）**
   - `skydiscover/search/adaevolve/*`
   - `skydiscover/search/evox/*`
   - `skydiscover/search/gepa_native/*`
   - 或 `topk/beam_search/best_of_n` 的 database 实现

### Q4: 如果我要改“搜索策略逻辑”，通常改哪些文件？

最常见两类：

- **改已有算法（推荐）**
  - 改对应算法目录下的 `controller.py` / `database.py` / 辅助模块（例如 adaevolve 的 adaptation、archive；evox 的 utils / strategy db）。
  - 一般不需要动 `route.py`（除非你改了 search type 名字）。

- **新增一个搜索算法**
  1. 新建 `skydiscover/search/<your_algo>/`。
  2. 实现数据库类（继承/兼容 `ProgramDatabase`）和可选 controller（继承默认 `DiscoveryController` 或复用默认）。
  3. 在 `search/route.py` 里 `register_database("<your_algo>", ...)`，必要时 `register_controller("<your_algo>", ...)`。
  4. 在 CLI 的 `_SEARCH_CHOICES` 增加新算法名，保证命令行可选。
  5. 在 config 里设置 `search.type: <your_algo>` 做验证。

### Q5: 你现在应该怎么下手（最小改动路径）

如果你只是想“调策略效果”而不是“发明新框架”，建议先：

1. 选定一个已有算法（如 `adaevolve`）。
2. 只改该算法目录下核心策略模块。
3. 用小迭代数在一个 benchmark 上快速验证。
4. 稳定后再扩到多 benchmark 批量跑。

---

## 10) 指定路径分析：`benchmarks/math/circle_packing/evaluator/evaluator.py`

你提到的这个路径是 **容器版 evaluator**（Docker 模式），不是根目录那个 `benchmarks/math/circle_packing/evaluator.py`（本地 Python 模式）。

### 这个文件的评测逻辑（容器版）

1. 要求候选程序实现 `run_packing()`，返回 `(centers, radii, sum_radii)`。
2. 用 `signal.alarm` 做单次运行超时保护（`TIMEOUT_SECONDS=300`）。
3. 强校验输出形状：`centers==(26,2)`、`radii==(26,)`。
4. 校验几何约束：无 NaN、半径非负、圆在单位正方形内、圆之间不重叠。
5. 评分使用 `sum_radii` 作为 `combined_score`（越大越好）。
6. 输出标准 JSON（包含 `status/combined_score/metrics/artifacts`）。

### 速度和算力判断（针对这个文件）

- 单次评测理论上是“程序执行 + 几何校验”，如果候选程序本身不重，通常不会太慢。
- 但它在 Docker 容器内运行（通过 `evaluate.sh` 调用），会有容器执行开销。
- 真正总耗时通常来自“多轮搜索反复评测”，不是一次评测本身。

### 这个 benchmark 里为何有两个 evaluator.py？

- `benchmarks/math/circle_packing/evaluator/evaluator.py`：容器协议版（给 `evaluate.sh` 调用，输出 JSON）。
- `benchmarks/math/circle_packing/evaluator.py`：本地 Python 接口版（返回 dict，含 stage1/stage2 兼容）。

你可以二选一使用（取决于你传入的是 `evaluator/` 目录还是 `evaluator.py` 文件）。

---

## 11) AdaEvolve 深入导读：从哪里开始看、执行顺序、改策略改哪些文件

你这个问题非常关键。下面给你“按函数级别”的阅读路径和改造路径。

### A. 先看执行入口（从框架到 AdaEvolve）

1. `skydiscover/cli.py:main()`
2. `skydiscover/cli.py:main_async()` 解析参数、加载 config
3. `skydiscover/runner.py:Runner.__init__()` 创建数据库（`create_database`）
4. `skydiscover/runner.py:Runner.run()` 构建 `DiscoveryControllerInput`
5. `skydiscover/search/route.py:get_discovery_controller()`
   - 根据 `search.type` 选择 controller
   - `adaevolve` 会走 `AdaEvolveController`
6. `AdaEvolveController.run_discovery()` 进入 AdaEvolve 主循环

这就是你读代码时最稳定的一条主线。

### B. AdaEvolve 单轮迭代的函数执行顺序（重点）

在 `AdaEvolveController.run_discovery()` 内，每轮大致是：

1. `AdaEvolveController._run_iteration(iteration, checkpoint_callback)`
2. （可选）`_generate_paradigms_if_needed()`：全局停滞时触发 paradigm
3. `_run_normal_step(iteration)`：失败可重试
4. `_generate_child(iteration, error_context=...)`：
   - 调 `self.database.sample(...)` 选 parent + context
   - 构建 prompt（`AdaEvolveContextBuilder.build_prompt`）
   - 调 `_execute_generation(...)`
5. `_execute_generation(...)`：
   - LLM 生成候选
   - 解析 diff/full rewrite
   - `self.evaluator.evaluate_program(...)` 评测
   - 组装 `Program` 返回
6. 回到 `_run_iteration(...)`：
   - 成功则 `_process_result(...)`（写库、日志、checkpoint）
7. 回到 `run_discovery(...)` 的 `finally`：
   - `self.database.end_iteration(iteration)`
   - 这里会做岛调度更新 + 迁移等关键动作

如果你要改“搜索策略”，最关键就是 `database.sample(...)` 与 `database.end_iteration(...)` 这两个节点。

### C. AdaEvolve 完整文件清单（代码）

`skydiscover/search/adaevolve/` 下完整文件如下：

- `README.md`
- `__init__.py`
- `controller.py`
- `database.py`
- `adaptation.py`
- `archive/__init__.py`
- `archive/diversity.py`
- `archive/unified_archive.py`
- `paradigm/__init__.py`
- `paradigm/generator.py`
- `paradigm/tracker.py`

### D. 如果你要“改进搜索策略”，通常改哪里

#### 1) 改“父代选择 / 探索-开发平衡”

优先改：

- `database.py:sample()`
- `database.py:_sample_from_archive()`
- `database.py:_sample_top()` / `_sample_weighted()` / `_sample_pareto_front()`

这是最直接影响“选谁当 parent”的地方。

#### 2) 改“岛调度/UCB/迁移”

优先改：

- `database.py:end_iteration()`
- `database.py:_migrate()` / `_migrate_archives()`
- `adaptation.py`（UCB / intensity 计算逻辑）

这是最直接影响“算力分配到哪个岛”和“信息怎么跨岛流动”的地方。

#### 3) 改“质量-多样性存档规则”

优先改：

- `archive/unified_archive.py`（收纳/淘汰/elite score）
- `archive/diversity.py`（distance/novelty 定义）

如果你想让搜索更“敢于跳出局部最优”，通常这里收益很大。

#### 4) 改“高层范式突破（paradigm）”

优先改：

- `paradigm/tracker.py`（何时判定停滞）
- `paradigm/generator.py`（生成什么样的突破方向）
- `controller.py:_generate_paradigms_if_needed()`（触发与注入时机）

### E. 改策略时，框架层常见联动文件

除了 adaevolve 目录本身，你通常还会触及：

- `configs/adaevolve.yaml`（参数暴露与默认值）
- `skydiscover/search/route.py`（若新增新算法名或替换 controller）
- `skydiscover/cli.py`（若新增 `--search` 候选）

### F. 推荐你第一轮改造的最小路径（一步一步）

1. 只改 `database.py:sample()` 的 parent 采样概率（不要一开始动太多模块）。
2. 固定其他开关（先不改 migration / paradigm），做 A/B 对比。
3. 若效果不稳定，再改 `archive/unified_archive.py` 的 elite score 权重。
4. 最后才动 `adaptation.py` 的 intensity/UCB 公式。

这样改可以最快定位“到底是哪一层带来的收益”。
