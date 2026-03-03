# ResourceManager — Architecture Reference

`src/campaign/resource_manager.py`

---

## Overview

`ResourceManager` is a **thread-safe, priority-aware resource allocator with preemptive scheduling**. It tracks a pool of CPU cores and (fractional) GPUs shared across multiple concurrent workflows and task types. When a high-priority task needs resources that are fully allocated, the manager automatically reclaims them from lower-priority running tasks, rather than waiting for a slot to become free organically.

The manager is intentionally **callback-driven and lock-free at call sites**: callers request resources asynchronously, receive a callback when granted, and receive a separate callback if the slot is later preempted. This decouples resource accounting (synchronous, thread-safe) from task execution (async, event-loop-driven).

---

## Core Concepts

### Resource Pool

The pool has two independent dimensions:

| Dimension | Type    | Meaning                                 |
|-----------|---------|------------------------------------------|
| `cpus`    | `int`   | CPU cores (whole numbers only)           |
| `gpus`    | `float` | GPU count (fractional, e.g. `0.5`, `0.00125`) |

Both dimensions must be satisfied simultaneously for a request to be granted.

**Floating-point note**: repeated subtraction of fractional GPU values (e.g. 800 × 0.00125) accumulates IEEE-754 rounding error. `_GPU_EPS = 1e-9` guards grant comparisons, and `_grant_locked` snaps the stored value to `0.0` when the result would be a tiny negative within that epsilon.

### Task States

A task exists in exactly one state at any time:

```
 request()                _try_dispatch_locked          release() / on_preempted
    │                            │                              │
    ▼                            ▼                              ▼
[PENDING] ─── resources available? ──► [RUNNING] ─────────────► [DONE]
              │   or victims found?              (normal finish)
              │
              └─► still blocked ──► stays in PENDING heap
```

- **PENDING**: `_Request` object in the `_pending` min-heap. Waiting for resources.
- **RUNNING**: `_RunningTask` object in `_running` dict. Holding reserved resources.
- **DONE**: removed from all tracking. Resources returned to pool.

### Priority Model

Priority is an integer where **higher = more important**. Typical values in SPHERICAL:

| Task type        | Priority | Notes                                     |
|------------------|----------|-------------------------------------------|
| `train_model`    | 10       | Highest — needs GPU, blocks all sims      |
| `inference`      | 10       | Model inference, CPU-only                 |
| `post_process`   | 10       | Post-processing step, CPU-only            |
| `simulation`     | 8        | Bulk of concurrent work, uses 0.5 GPU each |
| `client_req`     | 0        | Inference batch slots, fractional GPU     |

Preemption only targets tasks with **strictly lower** priority than the requester. Equal-priority tasks never preempt each other.

---

## Internal Data Structures

### `_Request` — pending heap entry

```python
@dataclass(order=True)
class _Request:
    neg_priority: int    # -priority (min-heap → highest priority pops first)
    seq:          int    # monotonically increasing insertion counter (FIFO tiebreak)
    task_id:      str    # unique identifier
    workflow_id:  str    # owning workflow (for bulk cancel/release)
    task_type:    str    # for logging and config lookup
    cpus:         int
    gpus:         float
    on_granted:   Callable[[], None]   # fire when slot is reserved
    on_preempted: Callable[[], None]   # fire if slot is later reclaimed
```

The heap is a **min-heap ordered by `(neg_priority, seq)`**, so `heapq.heappop()` always returns the highest-priority, oldest request. Only `neg_priority` and `seq` participate in ordering (`field(compare=False)` on the rest).

### `_RunningTask` — active task record

```python
@dataclass
class _RunningTask:
    task_id:      str
    workflow_id:  str
    task_type:    str
    priority:     int    # stored as positive (not negated)
    cpus:         int
    gpus:         float
    on_preempted: Callable[[], None]   # called if the RM reclaims this slot
```

Keyed by `task_id` in `_running: Dict[str, _RunningTask]`.

### `_logged_pending: set`

Tracks which pending `task_id`s have already had their "queued" debug message emitted. `_try_dispatch_locked` runs on every `request()` and `release()` call; without this guard the same task would log "queued" dozens of times while waiting. Cleared on grant or cancel.

---

## Public API

### `request(task_id, workflow_id, task_type, priority, cpus, gpus, on_granted, on_preempted)`

Submits a resource request. The method always returns immediately; work happens in callbacks.

**Flow inside `request()`**:
1. Wraps arguments into a `_Request` and pushes it onto the pending heap (`_enqueue_locked`).
2. Calls `_try_dispatch_locked` (under lock) to attempt immediate dispatch.
3. Releases the lock, then fires collected callbacks via `_fire()`.

`on_granted` — called once when the slot is successfully reserved. The caller should start execution here.

`on_preempted` — called once if the slot is reclaimed after being granted. Resources are already freed; the caller must **not** call `release()`. The caller is responsible for stopping the underlying work (e.g. cancelling the asyncio Task).

### `release(task_id)`

Called by a task when it finishes normally. Removes the task from `_running`, returns its resources to the pool, and triggers a dispatch attempt for waiting requests. No-op (with a warning log) if the task was already preempted.

### `release_by_workflow(workflow_id) → int`

Bulk-releases all running tasks belonging to `workflow_id`. Used for cleanup when a workflow exits. Returns the count of tasks released.

### `cancel(task_id) → bool`

Removes a **pending** (not yet granted) request from the heap. Returns `True` if found. Has no effect on running tasks.

### `cancel_by_workflow(workflow_id) → int`

Removes all pending requests for a workflow. Returns count removed.

### `close()`

Graceful shutdown:
- Fires `on_preempted` for all pending requests (unblocks any coroutines suspended in `_wait_for_resource`).
- Fires `on_preempted` for all running tasks.
- Resets resource counters to totals.

After `close()`, the manager is in a clean idle state and must not be used.

---

## Dispatch Algorithm: `_try_dispatch_locked`

Called every time state changes (new request, task release, bulk release). Runs a **greedy loop** from highest-priority pending request downward:

```
while pending:
    req = peek at heap head (highest priority)

    if avail_cpus >= req.cpus AND avail_gpus >= req.gpus - ε:
        pop from heap
        _grant_locked(req)          ← reserve resources, move to _running
        add to granted list

    else:
        victims = _find_victims_locked(req.cpus, req.gpus, req.priority)

        if victims found:
            pop from heap
            for each victim:
                return resources to pool
                remove from _running
                add to preempted list
            _grant_locked(req)
            add to granted list

        else:
            log "queued" (once, via _logged_pending)
            BREAK            ← head can't run even with preemption;
                               lower-priority items won't either
```

The early-exit `break` is a key correctness property: if the highest-priority pending request cannot be satisfied, dispatching lower-priority requests would violate priority ordering, so the loop stops entirely.

All collected `(granted, preempted)` lists are returned to the caller and fired **outside the lock** via `_fire()`.

### `_find_victims_locked`

Greedy victim selection:

1. Collect all running tasks with `priority < requester.priority` (candidates).
2. Sort by `(priority ascending, -(cpus + gpus) descending)` — lowest-priority first, ties broken by most resources freed per task (minimises victim count).
3. Accumulate victims until `avail + freed >= need` for both CPUs and GPUs.
4. Return the minimal victim list, or `[]` if even all candidates are insufficient.

---

## Thread Safety

The RM is designed for use from **multiple threads simultaneously** (e.g. a worker thread pool alongside an asyncio event loop).

Key guarantees:

- `_lock` is a `threading.Lock` held during all reads/writes to `_running`, `_pending`, `_logged_pending`, and resource counters.
- **Callbacks are always fired outside the lock.** `_enqueue_locked`, `_release_locked`, and `release_by_workflow` all collect callbacks under the lock, release it, then call `_fire()`. This means it is safe to call `request()` or `release()` from within `on_granted` or `on_preempted` without deadlock.
- **No CM↔RM lock nesting.** Any caller that holds its own lock must release it before calling RM methods.

---

## Integration Pattern: `_wait_for_resource` (asyncio bridge)

`DDSimManager` and `InferenceClient` both use an async helper that bridges the thread-safe RM callbacks into the asyncio event loop:

```python
async def _wait_for_resource(self, task_id, task_type, cancel_handle=None):
    loop = asyncio.get_running_loop()
    fut  = loop.create_future()

    def on_granted():
        loop.call_soon_threadsafe(fut.set_result, None)

    def on_preempted():
        def _handle():                      # runs in event loop thread
            if not fut.done():
                # PENDING phase — cancel the waiting future
                fut.set_exception(CancelledError(...))
            elif cancel_handle:
                # RUNNING phase — RM reclaimed slot while task is live
                self._rm_preempted_ids.add(task_id)
                cancel_handle[0].cancel()   # cancel the asyncio Task
        loop.call_soon_threadsafe(_handle)  # atomic with on_granted

    self._rm.request(..., on_granted=on_granted, on_preempted=on_preempted)
    await fut
```

### The `_handle()` race fix

`on_granted` and `on_preempted` can be called from different threads at nearly the same time. If `on_preempted` checked `fut.done()` directly in the RM thread, it could observe `False` (grant not yet applied), schedule `set_exception`, and then both `set_result` (from `on_granted`) and `set_exception` would fire in the event loop — the second raises `InvalidStateError`. Wrapping the check in `call_soon_threadsafe(_handle)` serialises both callbacks through the event loop queue, making the `fut.done()` check atomic with the subsequent action.

### Running-phase preemption via `cancel_handle`

`_wait_for_resource` only suspends until the resource is granted — after `await fut` returns, the future is already resolved (`fut.done() == True`), so `on_preempted`'s `fut.done()` branch becomes a no-op. To enable the RM to cancel a **running** asyncio Task:

```python
cancel_handle = []                               # mutable one-element list
await self._wait_for_resource(sim_idx, 'simulation', cancel_handle=cancel_handle)
simul = self.simulation(sim_inputs=sim_inputs)
cancel_handle.append(simul)                      # wire: RM → Task.cancel()
```

If the RM later preempts `sim_idx` (e.g. to grant a higher-priority `train_model`), `on_preempted._handle()` sees `cancel_handle` is populated, adds `task_id` to `_rm_preempted_ids`, and calls `cancel_handle[0].cancel()` on the live asyncio Task.

### `_rm_preempted_ids` — avoiding spurious `release()`

When a running sim is cancelled via `cancel_handle`, its `_on_sim_done` callback fires. Without tracking, `_on_sim_done` would call `self._rm.release(sim_idx)` — but the RM already reclaimed those resources during preemption. The result: a spurious warning log and an unnecessary `_try_dispatch_locked` call.

`_rm_preempted_ids` breaks the cycle:

```
on_preempted fires
  → _rm_preempted_ids.add(task_id)
  → cancel_handle[0].cancel()
  → _on_sim_done fires
      → sim_idx in _rm_preempted_ids?
          YES → discard, skip release()   (RM already handled it)
          NO  → call release()            (normal completion or cancel_sims kill)
```

---

## Callback Contracts Summary

| Scenario                                  | `on_granted` | `on_preempted` | Caller calls `release()`? |
|-------------------------------------------|:------------:|:--------------:|:-------------------------:|
| Resources available immediately           | ✓            | —              | Yes, on completion         |
| Preemption of lower-priority running tasks| ✓            | victims get ✓  | Requester: yes. Victims: **NO** |
| Request queued, later resources free up   | ✓ (deferred) | —              | Yes, on completion         |
| Request preempted while still pending     | —            | ✓              | No (never granted)         |
| Running task preempted by higher priority | ✓ (new task) | ✓ (victim)     | Victim: **NO**             |

---

## Known Design Decisions

**No same-priority preemption.** Tasks can only preempt tasks with *strictly lower* priority. Two simulations (both priority 8) will never preempt each other; they queue and wait for natural completion.

**Greedy victim selection, not optimal.** `_find_victims_locked` picks the fewest victims by iterating sorted candidates. It does not solve the bin-packing optimally — in practice the victim sets are small enough that this is not a concern.

**No starvation protection.** A steady stream of high-priority tasks could starve lower-priority ones indefinitely. In the SPHERICAL use case this is acceptable: `client_req` (priority 0) is designed to yield to simulations and training, and the bounded simulation count means GPU is always eventually freed.

**Fractional GPU accounting.** The RM tracks GPUs as floating-point fractions. This allows the inference client to register many small batch slots (e.g. 0.00125 GPU each = 800 batches across 4 GPUs) without each one claiming a full GPU. The `_GPU_EPS` guard and the snap-to-zero in `_grant_locked` prevent drift from accumulating into persistent negative available-GPU counts.
