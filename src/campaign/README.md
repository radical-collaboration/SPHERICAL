# Campaign Manager

RADICAL asyncflow-native orchestrator for multi-workflow HPC campaigns.  Runs
concurrent replicas of heterogeneous workflows inside a single `asyncio` event
loop backed by `radical.asyncflow`, with priority-based scheduling,
sliding-window concurrency caps, resource-pool gating, and adaptive cascading
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

    def __init__(self, config, _cm, _group_name, asyncflow, policies, engine_dragon): ...

    async def run(self, replica_id: str): ...                          # entry point (override run OR start)
    async def on_replica_done(self, replica_id, cm, final_state): ... # optional hook
    async def _signal_done(self): ...           # broadcast signal to all dependent groups
    async def _trigger_dependent(self, name, replicas=1): ...  # explicit activation of a named group
```

The CM injects six objects at construction time:

| Injected attribute | Type | Purpose |
|--------------------|------|---------|
| `self.config` | `dict` | per-group config section (CM scheduling keys stripped) |
| `self._cm` | `AsyncCampaignManager` | reference to the running CM (`None` in unit tests without a CM) |
| `self._group_name` | `str` | name of this replica's group (used by `_signal_done`) |
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
| `replicas` | total replicas to complete (omit / set to 0 for dependent groups) |
| `max_replicas` | sliding-window concurrency cap (defaults to `replicas` if 0) |
| `min_replicas` | minimum guaranteed concurrent slots (Pass 1 of scheduler) |
| `priority` | higher → scheduled first |
| `required_cpus` | CPU cores reserved from the pool while a replica runs |
| `required_gpus` | GPU slots reserved from the pool while a replica runs |
| `dependencies` | upstream groups; used to route `_signal_done()` and gate scheduling |
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

## Adaptive cascading dependency model

### Two group modes

A workflow group is **independent** or **dependent**, controlled entirely by
the config — no workflow code changes required to switch between modes.

**Independent** — `replicas: N` present; group starts immediately on `cm.start()`.

**Dependent** — `replicas` omitted (defaults to 0); group stays inactive until an
upstream replica signals the CM.  Each signal adds more replicas to the queue;
signals can repeat throughout the lifetime of the upstream run.

### Two signalling methods

#### `_signal_done()` — broadcast, topology-driven

```python
await self._signal_done()
```

Called from `run()` to indicate that this iteration has produced output.
The CM auto-routes the signal to **every group** that lists the caller's group
in its `dependencies` config field, adding +1 replica to each.  The caller
does not need to know downstream group names — the pipeline topology lives
entirely in the config.

Use this for **data-driven fan-out**: one upstream replica fires once per
result, and the CM decides which downstream groups get a new replica based on
the config graph.

```
md ──signal_done()──► CM routes ──► miniapps (+1 replica per signal)
```

#### `_trigger_dependent(name, replicas=N)` — explicit, named

```python
await self._trigger_dependent("downstream_group", replicas=1)
```

Called from `run()` or `on_replica_done()` when the upstream workflow
decides—based on its own logic—to start a specific number of downstream
replicas.  Each call is additive: calling it again queues more replicas.
If the group was already marked done, it is re-opened for scheduling.

Use this when the **calling workflow knows** the target name and controls
exactly how many replicas to spawn per event (e.g. one inference result
triggers exactly one downstream job).

```
inference ──_trigger_dependent("dummy", replicas=1)──► dummy (+1 per result)
```

### Scheduler re-runs on every signal

Every call to `_signal_done()` or `_trigger_dependent()` increments the
target group's `replicas` counter and immediately re-runs the two-pass
scheduler.  If resources are available the new replica starts at once;
otherwise it queues until resources free up.

### Campaign completion

Groups registered with `replicas=0` (dependent groups that were never
triggered) are **excluded** from the all-done check.  The campaign completes
when all groups that were actually triggered have finished, plus all
independent groups are done.

---

## Scheduling model

The CM runs a **two-pass greedy scheduler** on every state change (replica
start, replica finish, `signal_done`, `trigger_dependent`):

1. **Pass 1** — guarantee `min_replicas` concurrent slots for all eligible
   groups, highest priority first.
2. **Pass 2** — fill remaining capacity up to `max_replicas`, highest priority
   first.

Each pass also gates on `ResourcePool.can_fit()`: a group that has slots under
`max_replicas` but cannot be satisfied by the current resource pool is skipped
and a WARNING is emitted.

A group is **eligible** when every dependency group is **ready**:

- **Workflow-driven** (preferred): a dependency group called `_signal_done()`
  at any point during execution (`group.ready = True`).
- **Count-based fallback**: `dep.finished_replicas >= dep_threshold` (default 1).

---

## Authoring a workflow

### Independent workflow

```python
from src.campaign import BaseWorkflow

class SimWorkflow(BaseWorkflow):
    workflow_id = "sim"

    async def run(self, replica_id: str) -> None:
        # self.config  — dict forwarded from YAML workflow section
        # self.asyncflow — shared WorkflowEngine
        # self.policies  — Dragon Policy list (empty on concurrent backend)
        result = await do_simulation(self.asyncflow, self.config)

        # Signal the CM every time a result is ready.
        # CM auto-routes +1 replica to every group in config's dependencies.
        await self._signal_done()
```

### Dependent workflow (topology-driven via _signal_done)

No changes needed in the dependent workflow itself — it just runs normally.
The CM starts it when an upstream `_signal_done()` fires.

```yaml
# config.yaml
workflows:
  sim:
    replicas: 4        # independent: starts immediately
    ...

  analysis:
    dependencies: [sim] # dependent: starts at replicas=0; sim's _signal_done() adds replicas
    ...                  # no "replicas:" key — the count comes from signals at runtime
```

### Dependent workflow (explicit via _trigger_dependent)

Use when this workflow decides the count and the target name based on its
execution logic (e.g. a quality filter on results).

```python
class InferenceWorkflow(BaseWorkflow):
    workflow_id = "inference"

    async def run(self, replica_id: str) -> None:
        results = await run_inference(self.asyncflow, self.config)
        for r in results:
            if r.quality > THRESHOLD:
                # Explicitly queue 1 more replica of the downstream group.
                await self._trigger_dependent("downstream", replicas=1)

    async def on_replica_done(self, replica_id, cm, final_state):
        # on_replica_done fires after run() returns; useful for teardown
        # that should happen once per replica (e.g. releasing shared services).
        ...
```

Rules:
- Define **either** `run()` or `start()` — not both.
- Both `_signal_done()` and `_trigger_dependent()` are no-ops when no CM was
  injected (safe to call in unit tests).
- `on_replica_done` may be `async def` or `def`; the CM handles both.
- Do **not** call `asyncflow.shutdown()` from within a replica — the engine is
  owned by the caller and shut down after `cm.close()`.

---

## Runner pattern

```python
from src.campaign import AsyncCampaignManager

WORKFLOW_REGISTRY = {"sim": SimWorkflow, "analysis": AnalysisWorkflow}

cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)
await cm.start()    # schedules all groups with replicas > 0
await cm.wait()     # blocks until every triggered group is done
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
resources:
  total_cpus: 128
  total_gpus: 4

# ── Execution backend ────────────────────────────────────────────────────────
engine: dragon    # "dragon" or "concurrent" (falls back to concurrent if Dragon unavailable)

# ── Workflow groups ──────────────────────────────────────────────────────────
#
# Two modes — controlled by whether 'replicas' is present:
#
#   Independent (replicas: N):
#     Group starts immediately on cm.start().
#
#   Dependent (no replicas / replicas: 0):
#     Group starts at 0 replicas; stays inactive until an upstream replica
#     calls _signal_done() or _trigger_dependent().  Each call is additive —
#     the upstream workflow decides when and how many replicas to add based on
#     its own execution logic.  Calls can repeat across the lifetime of one
#     upstream replica (e.g. once per iteration, once per result).
#
# To switch a dependent group to independent: add 'replicas: N' and remove
# 'dependencies'.  No workflow code needs to change.

workflows:
  md:
    replicas:      2          # independent: starts immediately
    min_replicas:  1
    max_replicas:  2
    priority:      10
    required_cpus: 4
    required_gpus: 1
    # Each iteration calls _signal_done() → CM routes +1 replica to miniapps.

  miniapps:
    priority:      8
    min_replicas:  1
    max_replicas:  2
    required_cpus: 4
    required_gpus: 1
    dependencies:  [md]       # dependent: no replicas key → starts at 0
                              # md's _signal_done() adds replicas at runtime

  inference:
    replicas:      8          # independent
    min_replicas:  1
    max_replicas:  4
    priority:      6
    required_cpus: 4
    required_gpus: 1
    # on_replica_done calls _trigger_dependent("dummy", replicas=1) per result.

  dummy:
    priority:      5
    min_replicas:  2
    max_replicas:  4
    required_cpus: 4
    required_gpus: 0
    dependencies:  [inference] # dependent: inference triggers via _trigger_dependent
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
| `start()` | Schedule all groups with `replicas > 0`; creates the shared asyncflow engine if not pre-built |
| `wait(timeout=None)` | Async-block until all triggered groups complete; returns `True` on success |
| `close()` | Release CM resources (does NOT shut down asyncflow) |
| `signal_done(group_name)` | Called by `_signal_done()`; adds +1 replica to every group that lists `group_name` in `dependencies` |
| `trigger_dependent(name, replicas, config=None)` | Called by `_trigger_dependent()`; adds `replicas` to the named group and re-opens it if done |
| `add_replicas(group_name, n)` | Dynamically extend a group up to its `configured_replicas` cap |
| `status()` | Snapshot dict of all group states + `"resources"` key |
| `stats()` | Per-group `WorkflowStats(replicas_started, replicas_finished)` |

`register_group` key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `replicas` | `1` | Total replicas (0 for dependent groups) |
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
| `_cm` | reference to the running `AsyncCampaignManager` (`None` if no CM) |
| `_group_name` | name of this group in the CM (used by `_signal_done`) |
| `asyncflow` | shared `WorkflowEngine` |
| `policies` | list of Dragon `Policy` objects for assigned GPUs (empty on concurrent) |
| `engine_dragon` | Dragon backend handle (`None` on concurrent) |
| `_signal_done()` | broadcast signal to CM; adds +1 replica to all downstream groups; no-op without a CM |
| `_trigger_dependent(name, replicas)` | explicitly queue N replicas of a named group; no-op without a CM |
| `on_replica_done(replica_id, cm, state)` | post-replica hook; override as needed |
