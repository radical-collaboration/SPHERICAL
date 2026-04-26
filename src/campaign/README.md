# Campaign Manager

RADICAL asyncflow-native orchestrator for multi-workflow HPC campaigns.  Runs
concurrent replicas of heterogeneous workflows inside a single `asyncio` event
loop backed by `radical.asyncflow`, with priority-based scheduling,
sliding-window concurrency caps, resource-pool gating, and workflow-driven
dependency signalling.

---

## Module layout

```
src/campaign/
├── campaign_manager.py   # AsyncCampaignManager, CampaignManager, BaseWorkflow,
│                         # ResourcePool, WorkflowStats — all in one file
└── __init__.py           # re-exports all five public names
```

---

## Core concepts

### BaseWorkflow

All user workflows subclass `BaseWorkflow`.

```python
class BaseWorkflow:
    workflow_id: str = "base"   # unique prefix for replica IDs

    def __init__(self, config, on_ready, asyncflow, policies, engine_dragon): ...

    async def run(self, replica_id: str): ...           # entry point (override run OR start)
    async def on_replica_done(self, replica_id, cm, final_state): ...  # optional hook
    async def _signal_ready(self): ...                  # call from run() to unblock dependents
```

The CM injects five objects at construction time:

| Injected attribute | Type | Purpose |
|--------------------|------|---------|
| `self.config` | `dict` | per-group config section (CM scheduling keys stripped) |
| `self._on_ready` | async callable | calls `cm.signal_ready(group_name)` |
| `self.asyncflow` | `WorkflowEngine` | shared `radical.asyncflow` engine |
| `self.policies` | `list[Policy]` | one Dragon `Policy` per assigned GPU (empty on concurrent backend) |
| `self.engine_dragon` | backend handle | Dragon backend; `None` on concurrent |

When GPUs are assigned, the CM also injects two extra keys into `config`:

| Config key | Value |
|------------|-------|
| `assigned_gpu_ids` | list of GPU IDs assigned to this replica |
| `group_gpu_ids` | all GPU IDs held by the group right now (useful for multi-GPU service init) |

### Workflow entry point

Define **either** `run()` or `start()` — not both.  The CM detects which one
is overridden at `register_group` time and raises `ValueError` if both or
neither are defined.

### Workflow groups

A **group** is a named pool of replicas of the same workflow class.  Each group
has:

| Field | Meaning |
|-------|---------|
| `replicas` | total replicas to complete |
| `max_replicas` | sliding-window concurrency cap (defaults to `replicas` if 0) |
| `min_replicas` | minimum guaranteed concurrent slots (Pass 1 of scheduler) |
| `priority` | higher → scheduled first |
| `required_cpus` | CPU cores reserved from the pool while a replica runs |
| `required_gpus` | GPU slots reserved from the pool while a replica runs |
| `dependencies` | list of group names that must signal ready first |
| `dependency_threshold` | count-based fallback: N finished replicas in a dep group counts as "ready" (default 1) |

### ResourcePool

The CM maintains a single `ResourcePool` tracking CPU cores and GPU slots.
Setting a total to `0` disables tracking for that type (unlimited).

```
ResourcePool(total_cpus=128, total_gpus=4)
  available_cpus=108  available_gpus=3   ← after some allocations
```

| Method | Description |
|--------|-------------|
| `can_fit(cpus, gpus)` | `True` when the requested amounts are currently available |
| `allocate(cpus, gpus)` | Decrement available counts (on replica start) |
| `release(cpus, gpus)` | Increment available counts (on replica finish) |
| `usage_str()` | `"cpus=20/128  gpus=1/4"` (used/total, tracked types only) |
| `available_str()` | `"cpus=108/128  gpus=3/4"` |
| `as_dict()` | Full snapshot included in `cm.status()["resources"]` |

### GPU assignment

When `required_gpus > 0`, the CM pops GPU IDs from a global free list (FIFO)
and injects them as `assigned_gpu_ids` and `group_gpu_ids` into the replica's
`config`.  A Dragon `Policy(HOST_NAME, gpu_affinity=[...])` is also built and
injected as `self.policies[0]`.  IDs are returned to the free list when the
replica finishes.

---

## Scheduling model

The CM runs a **two-pass greedy scheduler** on every state change (replica
start, replica finish, `signal_ready`, `add_replicas`):

1. **Pass 1** — guarantee `min_replicas` concurrent slots for all eligible
   groups, highest priority first.
2. **Pass 2** — fill remaining capacity up to `max_replicas`, highest priority
   first.

Each pass also gates on `ResourcePool.can_fit()`: a group that has slots under
`max_replicas` but cannot be satisfied by the current resource pool is skipped
and a WARNING is emitted.

A group is **eligible** when every dependency group is **ready**:

- **Workflow-driven** (preferred): the dependency called
  `await self._signal_ready()` at any point during execution.  This fires
  immediately, regardless of how many replicas have finished.
- **Count-based fallback**: `dep.finished_replicas >= dep_threshold` (default 1).

---

## Authoring a workflow

```python
from src.campaign import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    workflow_id = "my_wf"

    async def run(self, replica_id: str) -> None:
        # self.config  — dict forwarded from YAML workflow section
        # self.asyncflow — shared WorkflowEngine
        # self.policies  — Dragon Policy list (empty on concurrent backend)
        await do_simulation(self.asyncflow, self.config)

        # Unblock dependent groups immediately (does not wait for run() to return).
        await self._signal_ready()

        await do_training(self.asyncflow, self.config)

    async def on_replica_done(self, replica_id, cm, final_state):
        # Optional: called after run() returns or raises.
        # final_state is "done" or "failed".
        if final_state == "done":
            await cm.add_replicas("downstream_group", n=1)
```

Rules:
- Define **either** `run()` or `start()` — not both.
- `_signal_ready()` is idempotent at the CM level; call it as many times as
  needed, only the first call has effect.
- `on_replica_done` may be `async def` or `def`; the CM handles both.
- Do **not** call `asyncflow.shutdown()` from within a replica — the engine is
  owned by the caller and shut down after `cm.close()`.

---

## Runner pattern

```python
from src.campaign import AsyncCampaignManager

WORKFLOW_REGISTRY = {"my_wf": MyWorkflow, "downstream": DownstreamWorkflow}

cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)
await cm.start()    # schedules all groups with satisfied dependencies
await cm.wait()     # blocks until every group is done
await cm.close()    # releases CM resources (does NOT shut down asyncflow)
# caller shuts down asyncflow separately, after telemetry is stopped
```

### Pre-built engine (recommended for telemetry)

When the caller builds the asyncflow engine itself (to start telemetry before
the CM runs), pass it at construction time:

```python
asyncflow = await WorkflowEngine.create(backend)
telemetry = await asyncflow.start_telemetry(...)

cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY, asyncflow=asyncflow)
await cm.start()
await cm.wait()
await cm.close()

await telemetry.stop()
await asyncflow.shutdown()
```

---

## Configuration

```yaml
# ── Cluster resource budget ──────────────────────────────────────────────────
# Set either to 0 to disable tracking (unlimited).
resources:
  total_cpus: 128
  total_gpus: 4

# ── Execution backend ────────────────────────────────────────────────────────
engine: dragon    # "dragon" or "concurrent" (falls back to concurrent if Dragon unavailable)

# ── Workflow groups ──────────────────────────────────────────────────────────
workflows:
  ddsim:
    replicas:             8
    min_replicas:         2       # guaranteed concurrent minimum
    max_replicas:         4       # sliding-window cap
    priority:             5
    required_cpus:        20      # cores held while one replica runs
    required_gpus:        0       # ddsim is CPU-only
    dependencies:         []
    dependency_threshold: 1       # unused when no deps

  inference:
    replicas:             16
    min_replicas:         1
    max_replicas:         4
    priority:             10      # higher → scheduled before ddsim overflow slots
    required_cpus:        32
    required_gpus:        1       # one GPU per inference replica
    dependencies:         [ddsim]
    dependency_threshold: 1
```

Config keys consumed by the CM and stripped before forwarding to `workflow.config`:

```
replicas  dependencies  dependency_threshold  priority
min_replicas  max_replicas  required_cpus  required_gpus
```

---

## API reference

### `AsyncCampaignManager`

| Method | Description |
|--------|-------------|
| `from_config(config, registry, asyncflow=None, engine_dragon=None)` | Build from YAML config dict + `{name: cls}` registry |
| `register_group(name, cls, ...)` | Register a workflow group |
| `start()` | Schedule all eligible groups; creates the shared asyncflow engine if not pre-built |
| `wait(timeout=None)` | Async-block until all groups complete; returns `True` on success |
| `close()` | Release CM resources (does NOT shut down asyncflow) |
| `signal_ready(group_name)` | Mark a group ready; unblock its dependents |
| `add_replicas(group_name, n)` | Dynamically extend a group (capped at `configured_replicas`) |
| `status()` | Snapshot dict of all group states + `"resources"` key |
| `stats()` | Per-group `WorkflowStats(replicas_started, replicas_finished)` |

`register_group` key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `replicas` | `1` | Total replicas |
| `min_replicas` | `0` | Guaranteed concurrent minimum |
| `max_replicas` | `0` | Sliding-window cap (0 → equals `replicas`) |
| `priority` | `0` | Scheduling priority (higher = first) |
| `required_cpus` | `0` | CPU cores reserved per running replica |
| `required_gpus` | `0` | GPU slots reserved per running replica |
| `dep_threshold` | `1` | Finished-replica count fallback for dependency readiness |

### `CampaignManager` (sync wrapper)

Thin synchronous wrapper around `AsyncCampaignManager`.  Runs a dedicated
event loop in a background thread so callers without an async context can use
plain blocking calls.  Same `from_config` / `register_group` / `start` /
`wait` / `close` / `status` / `stats` API.

### `BaseWorkflow`

| Attribute / method | Description |
|--------------------|-------------|
| `workflow_id` | class-level string; used as replica ID prefix |
| `config` | dict forwarded from the group's config section (CM keys stripped) |
| `asyncflow` | shared `WorkflowEngine` |
| `policies` | list of Dragon `Policy` objects for assigned GPUs (empty on concurrent) |
| `engine_dragon` | Dragon backend handle (`None` on concurrent) |
| `_on_ready` | injected async callable; invoke via `_signal_ready()` |
| `_signal_ready()` | fires the on-ready callback; no-op if not injected |
| `on_replica_done(replica_id, cm, state)` | post-replica hook; override as needed |
