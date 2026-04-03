# Campaign Manager

Async-native orchestrator for multi-workflow HPC campaigns.  Runs concurrent
replicas of heterogeneous workflows inside a single `asyncio` event loop, with
priority-based scheduling, sliding-window concurrency caps, and workflow-driven
dependency signalling.

---

## Module layout

```
src/campaign/
├── campaign_manager.py       # sync CampaignManager + BaseWorkflow + WorkflowStats
├── async_campaign_manager.py # AsyncCampaignManager (asyncio Tasks, recommended)
└── __init__.py               # exports all four public names
```

---

## Core concepts

### BaseWorkflow

All user workflows subclass `BaseWorkflow`.

```python
class BaseWorkflow:
    workflow_id: str               # unique prefix for replica IDs

    def __init__(self, config, on_ready, asyncflow): ...

    async def run(self, replica_id: str): ...          # entry point (override)
    async def on_replica_done(self, replica_id, cm, final_state): ...  # hook (optional)
    async def _signal_ready(self): ...                 # call from run() to unblock dependents
```

The CM injects two objects at construction time:

| Injected attribute | Type | Purpose |
|--------------------|------|---------|
| `self.config` | `dict` | per-group config section from the YAML |
| `self._on_ready` | async callable | calls `cm.signal_ready(group_name)` |
| `self.asyncflow` | `WorkflowEngine` | shared `radical.asyncflow` engine |

### Workflow groups

A **group** is a named pool of replicas of the same workflow class.  Each group
has:

| Field | Meaning |
|-------|---------|
| `replicas` | total replicas to complete |
| `max_replicas` | sliding-window concurrency cap |
| `min_replicas` | minimum guaranteed concurrent slots (pass 1 of scheduler) |
| `priority` | higher → scheduled first |
| `required_cpus` | CPU cores reserved from the pool while one replica is running |
| `required_gpus` | GPU slots reserved from the pool while one replica is running |
| `dependencies` | list of group names that must signal ready first |
| `dep_threshold` | fallback: how many finished replicas in a dep group counts as "ready" |

### ResourcePool

The CM maintains a single `ResourcePool` that tracks CPU cores and GPU slots
across the entire campaign.  Both dimensions are optional: setting a total to
`0` disables tracking for that type (unlimited).

```
ResourcePool(total_cpus=128, total_gpus=4)
  available_cpus=108  available_gpus=3   ← after some allocations
```

| Method | Description |
|--------|-------------|
| `can_fit(cpus, gpus)` | `True` when the requested amounts are currently available |
| `allocate(cpus, gpus)` | Decrement available counts (called when replica starts) |
| `release(cpus, gpus)` | Increment available counts (called when replica finishes) |
| `usage_str()` | `"cpus=20/128  gpus=1/4"` (used/total for tracked types) |
| `available_str()` | `"cpus=108/128  gpus=3/4"` |
| `as_dict()` | Full snapshot included in `cm.status()["resources"]` |

### AsyncCampaignManager

The recommended implementation.  Each replica is an `asyncio.Task`; the
scheduler runs on every state change (new replica starts, replica finishes,
`signal_ready` fires, `add_replicas` called).

**Two-pass greedy scheduler:**

1. **Pass 1** — guarantee `min_replicas` concurrent slots for all eligible
   groups, highest priority first.
2. **Pass 2** — fill remaining capacity up to `max_replicas`, highest priority
   first.

Each scheduling pass also gates on `ResourcePool.can_fit()`: a group that has
slots available under `max_replicas` but cannot be satisfied by the current
resource pool is skipped (and a WARNING is emitted).  Resources are allocated
atomically with the counter increment and released when the replica finishes.

A group is *eligible* when every dependency group is considered **ready**:

- **Workflow-driven** (preferred): the dependency called `await
  self._signal_ready()` at an appropriate milestone during execution.  This
  fires immediately regardless of how many replicas have finished.
- **Count-based fallback**: `dep.finished_replicas >= dep_threshold` (default 1).

### Shared asyncflow engine

The CM creates one `radical.asyncflow.WorkflowEngine` at `start()` and injects
it into every replica.  This is critical: asyncflow manages asyncio subprocesses
under the hood, and creating/destroying multiple engines in the same event loop
causes task submissions to hang.  The shared engine is shut down by `cm.close()`
only after all groups have finished.

---

## Authoring a workflow

```python
from src.campaign import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    workflow_id = "my_wf"

    async def run(self, replica_id: str) -> None:
        # Use self.config for parameters, self.asyncflow for task dispatch.
        await do_simulation(self.asyncflow, self.config)

        # Signal CM that dependent groups may now start.
        await self._signal_ready()

        await do_training(self.asyncflow, self.config)

    async def on_replica_done(self, replica_id, cm, final_state):
        # Optional: called after run() returns or raises.
        if final_state == "done":
            await cm.add_replicas("downstream_group", n=1)
```

Rules:
- Define **either** `run()` or `start()` — not both.
- `_signal_ready()` is idempotent at the CM level; call it as many times as
  needed, only the first call has effect.
- `on_replica_done` may be `async def` or `def`; the CM handles both.
- Do **not** call `asyncflow.shutdown()` from within a replica — the shared
  engine is owned by the CM.

---

## Configuration

```yaml
# ── Cluster resource budget ─────────────────────────────────────────────────
# The CM uses these totals to gate scheduling: a new replica only starts when
# its required_cpus / required_gpus can be satisfied from the pool.
# Set either to 0 to disable tracking for that type (unlimited).
resources:
  total_cpus: 128
  total_gpus: 4

workflows:
  ddsim:
    replicas:      32       # total replicas
    min_replicas:  2        # guaranteed concurrent minimum
    max_replicas:  4        # sliding-window cap
    priority:      5
    required_cpus: 20       # cores reserved per running replica
    required_gpus: 0        # ddsim is CPU-only
    dependencies:  []
    ddsim_data_ready: 3     # workflow-specific: signal ready after N sims

  inference:
    replicas:      16
    min_replicas:  1
    max_replicas:  4
    priority:      10       # higher → scheduled before ddsim overflow slots
    required_cpus: 32       # num_cpus_per_service per replica
    required_gpus: 1        # one GPU per service instance
    dependencies:  [ddsim]  # waits for ddsim signal_ready
    dependency_threshold: 1
```

**How the resource budget interacts with `max_replicas`:**

With the config above, 4 ddsim replicas running simultaneously consume
`4 × 20 = 80` CPUs.  When inference also starts, each inference replica
takes 1 GPU and 32 CPUs.  The scheduler will not start a 5th concurrent
inference replica even if `max_replicas` would allow it, because
`128 − 80 − 4×32 = 80 − 128 = −` would require 208 CPUs total.  The pool
acts as a hard ceiling that sits *below* the `max_replicas` window.

Config keys consumed by the CM and stripped before forwarding to
`workflow.config`:

```
replicas  dependencies  dependency_threshold  priority
min_replicas  max_replicas  required_cpus  required_gpus
```

---

## Runner pattern

```python
cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)
await cm.start()   # schedules all groups with satisfied dependencies
await cm.wait()    # blocks until every group is done
await cm.close()   # shuts down the shared asyncflow engine
```

---

## Live run trace (slurm-37972870)

The following annotated excerpt is from an actual Bridges-2 run
(`examples/run_campaign/`, 2026-03-12, 4 GPUs):

```
config:
  ddsim:     8 replicas, priority=5, min=2, max=4
  inference: 16 replicas, priority=10, min=1, max=4, deps=[ddsim]
```

```
09:52:57  CM initialised; asyncflow engine created (concurrent backend)
09:52:57  Group 'ddsim' eligible → scheduling ddsim_0..3  [DDDD]
09:52:58  ddsim_0..3 start DummyWorkflow (shared_asyncflow=True)
          Sims launch, training data threshold hit → sim preempted for training
09:53:12  Prediction scores evaluated; low-scoring sims killed
09:53:18  ddsim_0: "3 sims completed — signaling ready for downstream workflows"
          ↳  _signal_ready() → cm.signal_ready('ddsim')
          ↳  Group 'inference' is now eligible — status → running
09:53:18  scheduling inference_0..3  [DDDDIIII]   ← ddsim still running!
09:53:20  InferenceService starts loading ESM2 on 4 GPUs
09:53:25-29  4 × ESM2 service initialized (~1.5 s/GPU), workers started
09:53:29  All 4 clients start: 32 batches each, ~20 k tok/s per GPU
09:53:40  ddsim_3 done (1/8) → ddsim_4 starts       sliding window refill
09:53:47  ddsim_2 done (2/8) → ddsim_5 starts
09:54:05  ddsim_0 done (3/8) → ddsim_6 starts
09:54:11  ddsim_1 done (4/8) → ddsim_7 starts
09:54:14  inference_2 done (1/16) → tries add_replicas('ddsim', 1)
          WARNING: 'ddsim' already at configured cap (8) — ignoring
09:54:14  inference_0..3 all done → inference_4..7 scheduled
09:54:44  ddsim_7 done (8/8) → 'ddsim' group complete  [IIII]
09:54:54  inference_5 done (5/16)  viz: ddsim fully drained
09:55:32 – 09:56:15  inference_8..15 finish sequentially
09:56:15  inference_15 done (16/16) → ESM2 services torn down
          ↳  4 workers stopped (128 batches/worker processed)
09:56:15  All campaign workflow groups finished

── Campaign complete ──
  ddsim:     status=done  replicas=8/8
  inference: status=done  replicas=16/16

── Replica counts per workflow ──
  ddsim:     replicas_started=8   replicas_finished=8
  inference: replicas_started=16  replicas_finished=16

Total wall time: ~3 min 18 s
```

**Key observations from this run:**

1. **Early unblocking via `on_ready`**: inference started at 09:53:18 — only
   21 seconds into the run and long before any ddsim replica finished.  The
   `dep_threshold` fallback (count-based) would have blocked inference until
   the first ddsim replica completed at 09:53:40, wasting 22 seconds of GPU
   time.

2. **Steady-state concurrency**: the scheduler kept `[DDDDIIII]` — exactly 4
   ddsim + 4 inference running simultaneously — throughout the overlap window,
   fully utilizing the `max_replicas=4` cap for both groups.

3. **Shared asyncflow**: all 8 ddsim replicas share one engine instance
   (`shared_asyncflow=True`).  No per-replica engine creation overhead.

4. **Priority ordering**: inference (priority 10) fills its slots before
   ddsim (priority 5) when both groups compete for the same scheduling pass.

5. **ESM2 service pool**: 4 services initialized once; all 16 inference
   replicas reuse them via per-service locking and queue reset between replicas.

---

## API reference

### `AsyncCampaignManager`

| Method | Description |
|--------|-------------|
| `from_config(config, registry)` | Build from YAML config dict + `{name: cls}` registry |
| `register_group(name, cls, ...)` | Register a workflow group |
| `start()` | Schedule all eligible groups; creates the shared asyncflow engine |
| `wait(timeout=None)` | Async-block until all groups complete |
| `close()` | Shut down shared asyncflow engine |
| `signal_ready(group_name)` | Mark a group ready; unblock its dependents |
| `add_replicas(group_name, n)` | Dynamically extend a group (up to `configured_replicas` cap) |
| `status()` | Snapshot dict of all group states; includes `"resources"` key |
| `stats()` | Per-group `WorkflowStats(replicas_started, replicas_finished)` |

`register_group` resource parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `required_cpus` | `0` | CPU cores consumed while this replica runs |
| `required_gpus` | `0` | GPU slots consumed while this replica runs |

### `CampaignManager` (sync)

Same API surface but replicas run in a `ThreadPoolExecutor`.  Use when
workflows are pure Python and do not need `asyncio` internally.  Does not
support `on_ready` injection or shared asyncflow.  Resource tracking is
fully supported — `required_cpus`/`required_gpus` gate the sliding-window
refill in `_on_replica_finished` the same way as in the async CM.

### `BaseWorkflow`

| Attribute / method | Description |
|--------------------|-------------|
| `workflow_id` | class-level string; used as replica ID prefix |
| `config` | dict forwarded from the group's config section |
| `asyncflow` | shared `WorkflowEngine` (async CM only) |
| `_on_ready` | injected async callable; invoke via `_signal_ready()` |
| `_signal_ready()` | fires the `on_ready` callback; no-op if not injected |
| `on_replica_done(replica_id, cm, state)` | post-replica hook; override as needed |
