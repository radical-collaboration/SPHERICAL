# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Installation

```bash
# Core installation
pip install -e .

# With ESM2 model support (torch + transformers)
pip install -e ".[esm2]"

# With Dragon/RADICAL HPC support
pip install -e ".[dragon]"

# Full dev setup (recommended)
pip install -e ".[esm2,dragon,dev]"
```

Requires **Python ≥ 3.10** (union type syntax, structural pattern matching).

### Common Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run a single test file or test
pytest tests/test_campaign_manager.py
pytest tests/test_campaign_manager.py::TestClass::test_method

# Lint and format check
ruff check .
ruff format --check .

# Auto-format code
ruff format .

# Run linting and format with auto-fixes
ruff check . --fix
ruff format .
```

### Key Entry Points

- **Multi-workflow campaign**: `python workflows/run_campaign/esm2_ddsim_campaign/run_campaing.py --config workflows/run_campaign/esm2_ddsim_campaign/config.yaml` *(note: `run_campaing.py` — intentional typo in filename)*
- **Dreamer campaign (single run)**: `python workflows/run_campaign/dreamer_campaign/run_campaign.py --config workflows/run_campaign/dreamer_campaign/config.yaml`
- **Dreamer benchmark (multi-config, N runs)**: `python workflows/run_campaign/dreamer_campaign/benchmark.py --config workflows/run_campaign/dreamer_campaign/config.yaml --runs 5 --out benchmark_results.json`
- **ESM2 inference (standalone)**: `python workflows/esm2_inference/run_esm2_infern.py --config workflows/esm2_inference/config.yaml --mode local`

---

## Architecture Overview

SPHERICAL is an **async-native HPC workflow orchestrator** built on `radical.asyncflow`. The core innovation is **AsyncCampaignManager**, which orchestrates multiple heterogeneous workflow groups (replicas) inside a single Python asyncio event loop with sophisticated multi-stage dependency signalling, adaptive resource scheduling, and optional adaptive batching.

### High-Level Design

```
AsyncCampaignManager (campaign_manager.py)
├── SchedulerMixin (scheduler.py)
│   └── Two-pass greedy scheduler: guarantee min_replicas, fill to max_replicas
├── ExecutorMixin (executor.py)
│   └── Replica lifecycle: launch, monitor, completion, GPU assignment
├── MonitorMixin (monitor_mixin.py)
│   └── Periodic health checks + drift detection
│
├── BaseWorkflow (base_workflow.py)
│   └── User-defined workflow classes subclass this; override run() or start()
│
└── Optional Features (feature flags in config)
    ├── BackpressureNegotiator (backpressure.py) — per-edge queue depth controller
    ├── Sharder (sharder.py) — batches upstream triggers before downstream dispatch
    ├── Monitor (monitor.py) — detects pass-through & budget burn drift
    ├── Bandit (bandit.py) — Thompson-sampling arm selection for scheduling & sharding
    └── CandidateLog (candidate_log.py) — tracks upstream results for sharder ranking
```

### Core Concepts

#### AsyncCampaignManager

Orchestrates workflow groups with dependencies and resource constraints:

- **Groups**: Named pools of replicas of the same workflow class. Each group has:
  - `replicas`: total count (0 = dependent, wait for trigger)
  - `min_replicas` / `max_replicas`: concurrent caps
  - `priority`: scheduling priority (higher = first)
  - `required_cpus` / `required_gpus`: per-replica resource reservation
  - `dependencies`: upstream groups that must signal before this group starts

- **Two signalling modes**:
  - `_signal_done()`: broadcast to all downstream groups (topology-driven)
  - `_trigger_dependent(name, replicas=N)`: explicit queue N replicas to a named group

- **Scheduler**: Runs on every state change (replica finish, signal received). Two-pass greedy:
  1. Pass 1: guarantee `min_replicas` for all eligible groups (highest priority first)
  2. Pass 2: fill remaining capacity up to `max_replicas` (highest priority first)

A group is **eligible** when its dependencies are **ready**:
  - Workflow-driven: dependency called `_signal_done()` (sets `group.ready = True`)
  - Count-based fallback: `dep.finished_replicas >= dep.dependency_threshold`

#### BaseWorkflow

All user workflows subclass `BaseWorkflow`. The CM injects six objects at construction:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `config` | dict | per-group config (CM scheduling keys stripped) |
| `_cm` | AsyncCampaignManager | reference to running CM (`None` in unit tests) |
| `_group_name` | str | name of this group (used by `_signal_done()`) |
| `asyncflow` | WorkflowEngine | shared radical.asyncflow engine |
| `policies` | list[Policy] | Dragon `Policy` per assigned GPU (empty on concurrent) |
| `engine_dragon` | object | Dragon backend handle (`None` on concurrent) |

When GPUs are assigned, two extra config keys are injected:
- `assigned_gpu_ids`: list of GPU IDs for this replica
- `group_gpu_ids`: all GPUs held by the group right now

**Workflow entry points**: Define **either** `async def run()` or `def start()`, not both. The CM detects which is overridden and raises `ValueError` if both or neither are defined. Async coroutines are awaited directly; sync functions run via `asyncio.to_thread`.

Optional hook: `on_replica_done(replica_id, cm, final_state)` — called after entry-point returns/raises; can be async or sync.

#### ResourcePool

Tracks available CPU cores and GPU slots:

```python
pool = ResourcePool(total_cpus=128, total_gpus=4)
pool.can_fit(cpus=4, gpus=1)  # → True/False
pool.allocate(cpus=4, gpus=1)
pool.release(cpus=4, gpus=1)
pool.usage_str()  # → "cpus=20/128  gpus=1/4"
```

Setting total to 0 disables tracking (unlimited).

---

## Campaign Configuration

All campaigns use YAML config files with two sections:

### Resources & Engine

```yaml
engine: concurrent  # or "dragon" for HPC
resources:
  total_cpus: 128
  total_gpus: 4
```

### Workflow Groups

```yaml
workflows:
  sim:
    replicas: 8              # independent: starts immediately
    min_replicas: 2
    max_replicas: 4
    priority: 10
    required_cpus: 4
    required_gpus: 1
    # other keys forwarded to workflow.config

  analysis:
    priority: 8
    min_replicas: 1
    max_replicas: 4
    required_cpus: 4
    required_gpus: 1
    dependencies: [sim]      # dependent: starts at 0 replicas
                             # sim's _signal_done() adds replicas at runtime
```

**Key insight**: Switch a group between independent/dependent modes purely through config, no workflow code changes needed.

### Optional Feature Flags

```yaml
cm:
  features:
    backpressure: true   # hysteresis queue depth controller per edge
    sharder: true        # adaptive batch dispatch from trigger buffer
    monitor: true        # periodic health checks + drift alerts
    bandit: true         # Thompson-sampling cross-stage optimization
  monitor_interval_s: 30   # tick interval for monitor
  telemetry:
    collect_telemetry: true
    telemetry_dir: telemetry-results
  workflow_registry:
    sim: my_module.SimWorkflow
    analysis: my_module.AnalysisWorkflow
```

---

## Optional Features Deep Dive

### Backpressure (backpressure.py)

Per-edge hysteresis state machine that throttles downstream queue depth:

```yaml
workflows:
  downstream:
    backpressure_high: 200   # queue ≥ 200 → THROTTLE (block new starts)
    backpressure_low: 100    # queue ≤ 100 → WIDEN (dispatch more)
```

Three states (HOLD → THROTTLE → WIDEN → HOLD):
- **HOLD**: normal, neither throttling nor widening
- **THROTTLE**: queue too deep, sharder dispatch returns 0
- **WIDEN**: queue drained, dispatch multiplier increases

### Sharder (sharder.py)

Buffers upstream trigger signals and batch-dispatches downstream, with optional priority ranking:

```yaml
workflows:
  downstream:
    sharding:
      target_size: 100         # nominal batch size
      min_size: 10
      max_size: 200
      stratify: soft           # soft | strict | off
      use_bandit: true         # Thompson-sampling BP multiplier selection
      dispatch_cap: 500        # max replicas to dispatch (drop low-priority candidates)
```

**Stratify modes**:
- `off`: dispatch exactly 1 trigger per cycle
- `soft`: adaptive sizing with tail dispatch (partial batches acceptable)
- `strict`: hold buffer until target_size or upstream done (chemical diversity, etc.)

**Candidate ranking** (via ProfileWeights):
- Score, surrogate prediction, uncertainty, age, diversity (scaffold novelty)

### Monitor (monitor.py, monitor_mixin.py)

Periodic health checks and drift detection:

```yaml
cm:
  features:
    monitor: true
  monitor_interval_s: 30
  replan:
    budget_burn_deviation_pct: 20    # alert if spend > expected + 20%
    pass_through_deviation_pct: 25   # alert if pass-through ratio deviates 25%
    surrogate_recall_floor: 0.90     # alert if surrogate recall < 90%
```

Two monitoring paths:
1. **Reactive** (per replica finish) — low-latency drift check
2. **Periodic** (background task) — full health table, stall detection

### Bandit (bandit.py)

Thompson-sampling multi-armed bandit for optimization. Two use cases:

**Shard optimizer**: arms = multiplier factors [0.5, 0.75, 1.0, 1.25, 1.5]; reward = throughput

**Scheduling bandit**: arms = cross-stage priority; reward = downstream BP state quality

---

## Key File Organization

### Campaign Manager Core

- **campaign_manager.py**: Main class; constructor, config loading, group registration
- **base_workflow.py**: User-defined workflow base class
- **types.py**: `_GroupInfo`, `ResourcePool`, `WorkflowStats` data structures
- **scheduler.py**: SchedulerMixin — two-pass scheduling logic
- **executor.py**: ExecutorMixin — replica launch/completion/GPU assignment
- **monitor_mixin.py**: MonitorMixin — periodic health checks
- **gpu.py**: `detect_gpus()` (CUDA/nvidia-smi probe) and `find_gpus()` (Dragon node enumeration) — both degrade gracefully without Dragon/CUDA
- **sync_wrapper.py**: `CampaignManager` — synchronous wrapper around AsyncCampaignManager (thin thread-based bridge)

### Optional Features

- **backpressure.py**: `BackpressureNegotiator` — hysteresis state machine
- **sharder.py**: `Sharder` — buffering and batch dispatch with priority ranking
- **candidate_log.py**: `CandidateLog`, `CandidateHistory` — tracks upstream results
- **monitor.py**: `Monitor`, `DriftEvent` — drift detection logic
- **bandit.py**: `Bandit`, `SchedulingBandit` — Thompson-sampling optimization
- **profiles.py**: `ProfileWeights`, `PROFILES` — candidate ranking profiles
- **metrics.py**: `CampaignMetrics` — in-process event recording (timing, BP transitions, etc.)

### Utilities

- **src/utils/logger.py**: Colored structured logging with metrics recording
- **src/utils/workflow.py**: `_expand_env()`, `load_config()`, `find_gpus()`, `make_policies()`
- **src/inference/**: Multi-GPU inference service framework (ESM2 embeddings, HTTP server/client)

### Examples & Workflows

- **workflows/run_campaign/esm2_ddsim_campaign/**: Real multi-workflow campaign (DDMd sim + ESM2 inference)
  - `run_campaing.py`: main entry point
  - `ddmd_workflow.py`: wraps DeepDriveSim DDMd pipeline
  - `inference_workflow.py`: ESM2 client workflow
  - `config.yaml`: multi-stage campaign config
  
- **workflows/run_campaign/dreamer_campaign/**: Emulation campaign using radical.dreamer
  - `dreamer_workflow.py`: simulates task execution in-process
  - `config.yaml`: plan-based campaign (cm-prototype schema)
  - `benchmark.py`: runner with profiling

- **workflows/esm2_inference/**: Standalone ESM2 service
  - `run_esm2_infern.py`: launches inference server with worker pools per GPU

---

## YAML Config Features

### Environment Variable Expansion

All config files support `${VAR}` and `$VAR` shell-style references, expanded at load time by `_expand_env()` in `src/utils/workflow.py`:

```yaml
service_python: "${VE_HOME}/esm2/bin/python"
outdir: "${SPHERICAL_DIR}/workflows/sgdes/output"
```

Unset variables are preserved as literal strings (fail fast with clear `FileNotFoundError`).

### Config File Merging

When a workflow entry has `config_file`, that YAML is loaded and merged (scheduling params in the main config take precedence):

```yaml
workflows:
  inference:
    config_file: inference_specific.yaml  # loaded and merged
    replicas: 4                           # overrides any value in the file
```

---

## Testing

### Test Organization

- **tests/test_campaign_manager.py**: Core CM logic (scheduling, execution, hooks)
- **tests/test_inference_service.py**: Multi-GPU inference orchestration
- **tests/test_server.py**: aiohttp server endpoints
- **tests/test_client.py**: HTTP client interface
- **tests/test_sgdes_workflow.py**: SGDES protein engineering workflow
- **tests/test_logger.py**: Structured logging utilities
- **tests/test_utils.py**: Config loading and GPU helpers

### Test Markers

```bash
# Run only fast tests (skip slow)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only GPU tests (if available)
pytest -m gpu
```

### Async Tests

All async tests use `anyio` (not `pytest-asyncio` directly). The standard pattern used throughout the test suite:

```python
import pytest
pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

async def test_something():
    ...
```

`asyncio_mode = "auto"` in `pyproject.toml` applies to `pytest-asyncio`; the tests themselves rely on `anyio` with the `anyio_backend` fixture pinning execution to asyncio.

---

## Key Design Patterns

### Workflow Authoring Pattern

```python
from src.campaign import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    workflow_id = "my_wf"

    async def run(self, replica_id: str) -> None:
        # Do work
        result = await compute(self.asyncflow, self.config)
        
        # Signal downstream groups
        await self._signal_done()  # broadcast to all dependents
        # OR
        await self._trigger_dependent("specific_group", replicas=1)  # explicit

    async def on_replica_done(self, replica_id, cm, final_state):
        if final_state == "done":
            # cleanup
            pass
```

### Campaign Runner Pattern

```python
from src.campaign import AsyncCampaignManager

WORKFLOW_REGISTRY = {
    "workflow1": Workflow1,
    "workflow2": Workflow2,
}

# Option 1: from config
cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)

# Option 2: manual registration
cm = AsyncCampaignManager(engine="concurrent", total_cpus=128, total_gpus=4)
cm.register_group("wf1", Workflow1, replicas=4, ...)
cm.register_group("wf2", Workflow2, dependencies=["wf1"], ...)

# Run
await cm.start()
await cm.wait()
await cm.close()

# Caller is responsible for asyncflow lifecycle
asyncflow = await WorkflowEngine.create(backend)
# ... start telemetry if needed
cm = AsyncCampaignManager.from_config(config, registry, asyncflow=asyncflow)
# ... run campaign
await telemetry.stop()
await asyncflow.shutdown()
```

### GPU Assignment

When `required_gpus > 0`:
1. CM pops GPU IDs from a global free list (FIFO)
2. Injects `assigned_gpu_ids` and `group_gpu_ids` into replica config
3. Builds Dragon `Policy(HOST_NAME, gpu_affinity=[...])` and injects as `self.policies[0]`
4. Returns IDs to free list when replica finishes

---

## Important Implementation Notes

### No-op Signals in Unit Tests

Both `_signal_done()` and `_trigger_dependent()` are no-ops when `_cm is None`, making workflows safe to unit test without a CM:

```python
# Unit test — no CM injected
wf = MyWorkflow(config={...})
await wf.run("test_0")  # signals are no-ops
```

### Two Signalling Methods Are Mutually Exclusive per Workflow

- If a workflow calls `_signal_done()`, downstream groups are determined entirely by config `dependencies`
- If a workflow calls `_trigger_dependent()`, it explicitly decides which group gets replicas
- Mixing both on the same workflow is allowed but unconventional — `_signal_done()` is simpler for data-driven fan-out

### Scheduler Re-runs on Every State Change

Every replica completion or signal call triggers `_schedule()`, which:
1. Acquires the lock
2. Runs `_schedule_locked()` (two-pass greedy with dependency + resource checks)
3. Fires resulting replica tasks outside the lock

This keeps scheduling immediate and fair across groups.

### Campaign Completion Logic

The CM completes when:
- All groups with `replicas > 0` are finished
- All sharder buffers are empty
- Groups that were never triggered (dependent groups with 0 finished replicas) are excluded from the check

This allows purely dependent groups to remain inactive without stalling the campaign.

---

## Performance Tuning

### Concurrency Caps

- `max_replicas`: sliding-window concurrency cap per group
- `min_replicas`: guaranteed concurrent slots (priority-ordered across groups in Pass 1)
- If a group has slots but cannot be satisfied by resources, a WARNING is logged

### Backpressure Tuning

High `backpressure_high` + low `backpressure_low` gap = frequent oscillation.  Recommend:
- `high_water ≈ 1.5 × downstream_total`
- `low_water ≈ 0.5 × downstream_total`

### Sharder Target Size

- Too small (e.g., 1): no batching benefit, frequent dispatch overhead
- Too large: causes queue buildup and backpressure throttling
- Recommend: 5–20% of downstream group's total replicas, tuned via A/B testing

### Monitor Interval

- Too small (< 10s): log spam, overhead
- Too large (> 120s): miss transient drifts
- Recommend: 20–60s for typical campaigns; shorter (5–15s) for debugging

---

## Deployment

### Local Testing (Concurrent Backend)

```bash
python run_campaing.py --config config.yaml
```

Uses `radical.asyncflow` ConcurrentExecutionBackend (pure asyncio, no MPI).

### HPC Deployment (Dragon Backend)

```bash
dragon -m workflows/run_campaign/esm2_ddsim_campaign/run_campaing.py \
  --config workflows/run_campaign/esm2_ddsim_campaign/config.yaml
```

Sets `engine: dragon` in config; CM detects and uses DragonExecutionBackendV3.

### Telemetry Visualization

```bash
bash workflows/plot_telemetry.sh \
  workflows/sgdes/telemetry_output/out.jsonl \
  --out-dir plots/sgdes
```

Plots asyncflow native JSONL telemetry to workflow dashboard PNG.

### Campaign Timeline Visualization

```bash
python workflows/run_campaign/plot_cm_timeline.py \
  slurm-17715157.out \
  --config workflows/run_campaign/config.yaml \
  --out replica_timeline.png
```

Parses SLURM log; produces Gantt chart (replicas + resource utilization) + config summary table.

