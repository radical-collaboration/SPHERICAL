# SPHERICAL

HPC workflow orchestration framework for multi-GPU protein inference and engineering campaigns.

## Features

- **AsyncCampaignManager** — async-native orchestrator for concurrent multi-workflow campaigns with priority scheduling, resource pools, and dependency signalling
- **Adaptive Optimization Layers** — opt-in, config-driven: quality routing (Sharder), flow control (Backpressure), surrogate-gated Triage (RUN/DISCARD/ADVANCE), and a BudgetController that keeps spend on plan; drift-driven Replanning. Cross-stage scheduling priority is driven by the **ADR agent layer** (rule / bandit / LLM policies), not an in-CM bandit
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
│   ├── run_campaign/                # Multi-workflow campaigns
│   │   ├── plot_cm_timeline.py         # Gantt timeline + resource chart from SLURM log
│   │   ├── esm2_ddsim_campaign/        # real HPC campaign: ESM2 inference + DeepDriveSim (Dragon/GPU)
│   │   │   ├── run_campaing.py · config.yaml · gpu_sbatch.sh
│   │   │   └── inference_workflow.py · ddmd_workflow.py · miniapps_workflow.py · dummy_workflow.py
│   │   └── dreamer_campaign/           # in-process emulation (radical.dreamer) for benchmarking
│   │       ├── run_campaign.py · config*.yaml
│   │       ├── benchmark.py · benchmark_adr.py    # feature-flag + ADR-policy benchmarks
│   │       └── plot_optimizations.py · plot_policy_comparison.py · plot_deadline_yield.py
│   └── sgdes/                       # SGDES protein engineering
│       ├── run_workflow.py
│       ├── sgdes_workflow.py
│       └── config.yaml
└── tests/
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

Two campaigns ship under `workflows/run_campaign/`:

```bash
# Real HPC campaign (ESM2 inference + DeepDriveSim) — Dragon backend, real GPUs:
cd workflows/run_campaign/esm2_ddsim_campaign
dragon run_campaing.py --config config.yaml
# local smoke test (no Dragon): python run_campaing.py --config config.yaml --engine concurrent

# Emulated campaign (radical.dreamer, in-process) — for benchmarking scheduling policies:
cd workflows/run_campaign/dreamer_campaign
python run_campaign.py --config config.yaml --policy rule    # none | rule | bandit | llm
```

Config structure:

```yaml
resources:
  total_cpus: 128
  total_gpus: 4

# Optional ADR agent layer — drives cross-stage scheduling priority each tick.
cm:
  adr:
    policy: rule        # none | rule | bandit | llm  (override with --policy)
    tick_s: 2.0

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
    [--config workflows/run_campaign/esm2_ddsim_campaign/config.yaml] \
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

### ADR scheduling-policy comparison

The dreamer runner can drive scheduling from a swappable `radical.adr` policy
(`--policy {none|rule|bandit|llm}`) and record each decision cycle to JSONL with
`--record`. `plot_policy_comparison.py` then plots the policies side by side —
assigned priority per workflow over cycles — so the rule/llm stable downstream-first
ladder contrasts visually with the bandit's still-exploring (reshuffling) priorities.

```bash
cd workflows/run_campaign/dreamer_campaign

# run the same campaign under each policy, recording decisions
python run_campaign.py --policy rule   --record
python run_campaign.py --policy bandit --record
python run_campaign.py --policy llm    --record    # needs OPENROUTER_API_KEY

# plot them together
python plot_policy_comparison.py \
    adr-decisions-rule.jsonl adr-decisions-bandit.jsonl adr-decisions-llm.jsonl \
    --out plots/policy_comparison.png
```

Requires `pip install -e ".[adr]"` (the LLM policy also needs `".[llm]"`). The
policy and recording can also be set in `config.yaml` under `cm.adr`.

**Batch benchmark (all policies in one job).** `benchmark_adr.py` runs every
policy N times (same metrics shape as `benchmark.py`), writing one results JSON
plus per-cycle decision logs under `adr-logs/`:

```bash
python workflows/run_campaign/dreamer_campaign/benchmark_adr.py \
    --runs 5 --out benchmark_adr_results.json
    # or restrict: --policies none rule bandit
```

Cross-stage scheduling priority is owned entirely by the ADR policy (the CM has
no in-loop scheduling bandit); `--policy bandit` runs the same Thompson-sampling
bandit wrapped as an ADR agent.

`benchmark_adr.py` also supports a **deadline-yield** objective (`--mode
deadline-yield --deadline 60`): instead of time-to-N-leads, it measures how many
terminal leads each policy produces within a fixed wall-clock window (higher is
better — the realistic HPC framing). `plot_deadline_yield.py` renders the
leads-per-policy figure with per-run spread. For the full analysis of when each
policy wins and why downstream-first is hard to beat, see
[docs/scheduling_policy_comparison.md](docs/scheduling_policy_comparison.md).

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
