# SPHERICAL

HPC workflow orchestration framework for multi-GPU protein inference and engineering campaigns.

## Features

- **AsyncCampaignManager** — async-native orchestrator for concurrent multi-workflow campaigns with priority scheduling, resource pools, and dependency signalling
- **Adaptive Optimization Layers** — opt-in, config-driven: quality routing (Sharder), flow control (Backpressure), Thompson-sampling Bandits, surrogate-gated Triage (RUN/DISCARD/ADVANCE), and a BudgetController that keeps spend on plan; drift-driven Replanning
- **Structured Campaign Plans** — typed `CampaignPlan`/`StageSpec` schema (`src/campaign/plan/`) alongside the legacy flat config, resolved by a single `load_plan()`
- **Multi-GPU Inference** — worker pool per GPU with automatic load balancing; aiohttp HTTP server/client
- **ESM2 Inference Workflow** — standalone or campaign-embedded ESM2-650M embedding service
- **SGDES Workflow** — Structure-Guided Deep Evolution Solver for iterative protein sequence optimisation
- **Dragon/Asyncflow Integration** — HPC runtime for distributed multi-node execution via DragonHPC
- **Automatic Device Detection** — CUDA GPUs if available, CPU fallback
- **YAML Config with Env-Var Expansion** — `${VAR}` references in config files are resolved at load time
- **Telemetry & Visualization** — asyncflow native JSONL telemetry with workflow dashboard plots; campaign replica timeline and resource utilization charts from SLURM logs

---

## Repository Layout

```
spherical/
├── src/
│   ├── campaign/                    # AsyncCampaignManager + BaseWorkflow + ResourcePool
│   │   ├── campaign_manager.py      # core: scheduler/executor/monitor mixins
│   │   ├── sharder.py · backpressure.py · bandit.py     # quality routing, flow control, learning
│   │   ├── triage.py · surrogate.py · budget_controller.py  # surrogate-gated selective execution
│   │   ├── replanning.py · monitor.py · candidate_log.py    # drift handling + tracking
│   │   └── plan/                    # CampaignPlan/StageSpec schema + load_plan()
│   ├── inference/                   # InferenceService base, orchestrator, server
│   │   ├── esm2_service/            # ESM2InferenceService + ESM2Client
│   │   ├── inference_client.py
│   │   ├── orchestrator.py
│   │   ├── server.py
│   │   └── utils.py                 # load_config, ensure_dir, export_metrics
│   └── utils/
│       ├── logger.py                # structured Logger with colours
│       └── workflow.py              # _expand_env, load_config, find_gpus, make_policies
├── workflows/
│   ├── plot_telemetry.sh            # Plot asyncflow JSONL telemetry → workflow dashboard PNG
│   ├── esm2_inference/              # Standalone ESM2 inference runner
│   │   ├── run_esm2_infern.py
│   │   └── config.yaml
│   ├── run_campaign/                # Multi-workflow campaign (DDSim + Inference)
│   │   ├── run_campaing.py
│   │   ├── inference_workflow.py
│   │   ├── ddmd_workflow.py
│   │   ├── plot_cm_timeline.py         # Gantt timeline + resource chart from SLURM log
│   │   └── config.yaml
│   └── sgdes/                       # SGDES protein engineering
│       ├── run_workflow.py
│       ├── sgdes_workflow.py
│       └── config.yaml
└── tests/
    ├── test_campaign_manager.py
    ├── test_inference_service.py
    ├── test_client.py
    ├── test_server.py
    ├── test_sgdes_workflow.py
    ├── test_logger.py
    └── test_utils.py
```

---

## Installation

```bash
# Core
pip install -e .

# With ESM2 model support (torch + transformers)
pip install -e ".[esm2]"

# With Dragon/RADICAL HPC support
pip install -e ".[dragon]"

# Full (includes dev and plotting extras)
pip install -e ".[esm2,dragon,dev]"
```

---

## Quick Start

### ESM2 Inference (standalone)

```bash
# Local mode — model runs in-process, no HTTP server
python workflows/esm2_inference/run_esm2_infern.py \
    --config_file workflows/esm2_inference/config.yaml \
    --mode local

# Server mode (default) — model hosted via HTTP; use dragon for multi-node
dragon -m workflows/esm2_inference/run_esm2_infern.py \
    --config_file workflows/esm2_inference/config.yaml
```

Key config options (`workflows/esm2_inference/config.yaml`):

```yaml
model_path: "facebook/esm2_t33_650M_UR50D"
num_services: 1
num_workers_per_gpu: 4
server_port: 8000
num_batches: 32
max_batch_tokens: 20000
mode: local           # "local" or "server"
engine: concurrent    # "concurrent" or "dragon"
service_python: "${VE_HOME}/esm2/bin/python"   # resolved at load time
```

### Multi-workflow Campaign

```bash
python workflows/run_campaign/run_campaing.py --config workflows/run_campaign/config.yaml
```

Config structure:

```yaml
resources:
  total_cpus: 128
  total_gpus: 4

workflows:
  ddsim:
    replicas:      8
    concurrency_floor:  2
    concurrency_cap:  4
    priority:      5
    required_cpus: 20
    dependencies:  []

  inference:
    replicas:      16
    concurrency_floor:  1
    concurrency_cap:  4
    priority:      10
    required_cpus: 32
    required_gpus: 1
    dependencies:  [ddsim]
```

### SGDES Protein Engineering

```bash
sbatch workflows/sgdes/delta_gpu_sbatch.sh
```

See [workflows/sgdes/README.md](workflows/sgdes/README.md) for full setup, configuration, and scaling results.

---

## Campaign Manager

`AsyncCampaignManager` orchestrates heterogeneous workflow groups inside a single `asyncio` event loop.

### Authoring a workflow

```python
from src.campaign import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    workflow_id = "my_wf"

    async def run(self, replica_id: str) -> None:
        await do_work(self.asyncflow, self.config)
        await self._signal_ready()          # unblock dependent groups immediately

    async def on_replica_done(self, replica_id, cm, final_state):
        if final_state == "done":
            await cm.add_replicas("downstream", n=1)
```

### Runner pattern

```python
cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)
await cm.start()
await cm.wait()
await cm.close()
```

See [src/campaign/README.md](src/campaign/README.md) for full API reference, scheduler details, and a live run trace.

---

## Extending for New Model Types

Subclass `InferenceService` from `src.inference.inference_service`:

```python
from src.inference.inference_service import InferenceService

class MyModelService(InferenceService):
    def _load_models(self):
        for device in self.devices:
            self.models[device] = load_model().to(device)

    def process_batch_sync(self, batch_id: int, device: str):
        results = self.models[device](self.reply_store[batch_id])
        self.reply_store[batch_id] = results
        self.processed_queue.put_nowait(batch_id)

    async def generate_batch(self) -> tuple:
        seq = await self.input_queue.get()
        if seq is None:
            raise StopAsyncIteration
        batch = tokenize(seq)
        return len(batch), batch
```

---

## Environment Variables in YAML Configs

All config files support `${VAR}` and `$VAR` shell-style references.
They are expanded by `_expand_env` (in `src/utils/workflow.py`) at load time,
so paths like the following work without any Python-side substitution:

```yaml
outdir:         "${SPHERICAL_DIR}/workflows/sgdes/mayv_output"
service_python: "${VE_HOME}/esm2/bin/python"
```

If a variable is not set at runtime the literal string is preserved, producing a clear `FileNotFoundError` rather than an obscure downstream crash.

---

## Visualization

### Asyncflow workflow telemetry dashboard

`workflows/plot_telemetry.sh` plots the native JSONL telemetry produced by
`asyncflow.start_telemetry()` (enabled via `collect_telemetry: true` in the
workflow config).  It calls
`radical.asyncflow/workflows/telemetry/plot_workflow_dashboard.py` and saves a
PNG to `workflows/plots/<wf_name>/`.

```bash
bash workflows/plot_telemetry.sh <telemetry.jsonl> [--out-dir DIR]
```

The workflow name is inferred from the directory layout
(`<wf_name>/telemetry-output/<file>.jsonl`); pass `--out-dir` to override.
Output file: `workflow_dashboard_<YYYYMMDD_HHMMSS>.png`.

**Example** (SGDES run):
```bash
bash workflows/plot_telemetry.sh \
    workflows/sgdes/telemetry_output/out.jsonl \
    --out-dir plots/sgdes
```

### Campaign Manager replica timeline

`workflows/run_campaign/plot_cm_timeline.py` parses a SLURM output log and
produces a Gantt chart of replica execution spans with a resource utilization
panel (GPU/CPU in use over time) and a campaign config summary table.

```bash
python workflows/run_campaign/plot_cm_timeline.py slurm-<jobid>.out \
    [--config workflows/run_campaign/config.yaml] \
    [--out timeline.png]
```

| Output element | Description |
|----------------|-------------|
| Gantt chart | One bar per replica, coloured by workflow group; red border = error; dependency arrows show signal-ready flow |
| Resource panel | Step plot of GPU and CPU slots in use over elapsed time (from scheduler log lines) |
| Config table | Replicas, priority, CPU/GPU requirements, min/max, and dependency graph per group |

`config.yaml` is auto-detected when it sits next to the log file.  If found,
group metadata is taken from the config (authoritative); otherwise it is parsed
from the log lines.

**Example**:
```bash
python workflows/run_campaign/plot_cm_timeline.py \
    workflows/run_campaign/slurm-17715157.out \
    --out replica_timeline.png
```

### Dreamer campaign timeline (with simulation stats)

`workflows/run_campaign/dreamer_campaign/plot_dreamer_timeline.py` is a
Dreamer-specific superset of the timeline above: it produces the same Gantt +
resource-utilization rows **plus** a third row of emulation metrics (simulated
makespan per replica, task-ops box plots from the `dreamer-profiles/*.json`,
and a per-workflow stats table).

```bash
python workflows/run_campaign/dreamer_campaign/plot_dreamer_timeline.py <log> \
    [--profiles-dir dreamer-profiles/] \
    [--config workflows/run_campaign/dreamer_campaign/config.yaml] \
    [--out dreamer_timeline.png]
```

The profiles directory is auto-detected next to the log when `--profiles-dir`
is omitted. Use `plot_cm_timeline.py` for non-Dreamer campaigns.

### Benchmark optimization plots

`workflows/run_campaign/dreamer_campaign/plot_optimizations.py` reads the
`benchmark_results.json` produced by `benchmark.py` and writes 7 comparison
plots (wall time, pipeline Gantt, cascade funnel, GPU utilization, shard
dispatch, bandit convergence, time-to-target) — one per optimization axis.

```bash
# 1. produce the results (N runs per configuration)
python workflows/run_campaign/dreamer_campaign/benchmark.py \
    --config workflows/run_campaign/dreamer_campaign/config.yaml \
    --runs 5 --out benchmark_results.json

# 2. render the plots
python workflows/run_campaign/dreamer_campaign/plot_optimizations.py \
    [--results benchmark_results.json] \
    [--out-dir plots/optimizations]
```

Config display names are mapped via `CFG_DISPLAY` and workflow stage labels via
`DISPLAY` at the top of the script; both default to the antigen-cascade names.

### Budget-control illustration

`workflows/run_campaign/dreamer_campaign/plot_budget_control.py` renders the
score-cutoff adaptation and burn-ratio convergence for the `budget_control`
benchmark case (a 2-panel figure) from the same `benchmark_results.json`.

```bash
python workflows/run_campaign/dreamer_campaign/plot_budget_control.py \
    [--results benchmark_results.json] \
    [--out plots/diagrams/budget_control_illustration.png]
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Lint and format
ruff check .
ruff format .
```

---

## License

MIT License
