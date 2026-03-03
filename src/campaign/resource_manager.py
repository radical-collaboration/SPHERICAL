#!/usr/bin/env python3
"""
ResourceManager — thread-safe, priority-aware resource allocator with preemption.

When a request cannot be satisfied from available resources the manager checks
whether preempting lower-priority *running* tasks would free enough.  The
minimum set of lowest-priority tasks is selected, their ``on_preempted``
callbacks are fired, their resources are immediately reclaimed, and the
requesting task's ``on_granted`` callback is fired.

All callbacks are always fired **outside** the internal lock, so it is safe
to call ``request`` or ``release`` from within a callback.

Quick-start
-----------
    rm = ResourceManager(total_cpus=64, total_gpus=8)

    rm.request(
        task_id="t1", workflow_id="wf", task_type="sim",
        priority=5, cpus=4, gpus=1.0,
        on_granted=lambda: start(),
        on_preempted=lambda: cancel(),
    )

    rm.release("t1")   # call when the task finishes normally

Preemption contract
-------------------
- ``on_preempted`` is called once if and only if the task was previously
  granted resources.
- After ``on_preempted`` fires, ``release`` must NOT be called for that
  task — the manager has already reclaimed the resources.
- The underlying thread/process may still be running; it is the caller's
  responsibility to stop it.
"""

import heapq
import itertools
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
from ..utils.logger import Logger

# Guard against floating-point rounding when comparing GPU amounts.
# Many small additions/subtractions (e.g. 800 × 0.00125) accumulate error
# that can make an "essentially equal" value compare as strictly less-than.
_GPU_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _RunningTask:
    task_id:      str
    workflow_id:  str
    task_type:    str
    priority:     int
    cpus:         int
    gpus:         float
    on_preempted: Callable[[], None]


@dataclass(order=True)
class _Request:
    """Min-heap entry: ordered by (neg_priority, seq) — highest priority runs first."""
    neg_priority:  int
    seq:           int
    task_id:       str      = field(compare=False)
    workflow_id:   str      = field(compare=False)
    task_type:     str      = field(compare=False)
    cpus:          int      = field(compare=False)
    gpus:          float    = field(compare=False)
    on_granted:    Callable = field(compare=False)
    on_preempted:  Callable = field(compare=False)


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class ResourceManager:
    """
    Thread-safe priority resource allocator with preemptive scheduling.

    Parameters
    ----------
    total_cpus : int
        Total CPU cores available.
    total_gpus : float
        Total GPUs available.
    """

    def __init__(self, total_cpus: int, total_gpus: float) -> None:
        self._total_cpus = total_cpus
        self._total_gpus = float(total_gpus)
        self._avail_cpus = total_cpus
        self._avail_gpus = float(total_gpus)

        self._running: Dict[str, _RunningTask] = {}
        self._pending: List[_Request]          = []   # min-heap
        self._logged_pending: set              = set()  # task_ids already logged as queued

        self._lock = threading.Lock()
        self._seq  = itertools.count()

        # ---- logger --------------------------------------------------
        self._log = Logger(name="ResourceManager", use_colors=True)
        self._log.info(
            f"ResourceManager initialised: total_cpus={total_cpus} total_gpus={total_gpus}"
        )
        self.debug = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cpus(self) -> int:
        return self._total_cpus

    @property
    def total_gpus(self) -> float:
        return self._total_gpus

    @property
    def available_cpus(self) -> int:
        with self._lock:
            return self._avail_cpus

    @property
    def available_gpus(self) -> float:
        with self._lock:
            return self._avail_gpus

    def running_tasks(self, workflow_id: str = None) -> List[dict]:
        """Snapshot of running tasks, optionally filtered by *workflow_id*."""
        with self._lock:
            return [
                {
                    "task_id":    t.task_id,
                    "workflow_id": t.workflow_id,
                    "task_type":  t.task_type,
                    "priority":   t.priority,
                    "cpus":       t.cpus,
                    "gpus":       t.gpus,
                }
                for t in self._running.values()
                if workflow_id is None or t.workflow_id == workflow_id
            ]
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(
        self,
        task_id:      str,
        workflow_id:  str,
        task_type:    str,
        priority:     int,
        cpus:         int,
        gpus:         float,
        on_granted:   Callable[[], None],
        on_preempted: Callable[[], None],
    ) -> None:
        """
        Request resources for a task.

        The manager grants immediately if resources are available, preempts
        lower-priority running tasks if necessary, or queues the request
        until resources free up.

        Parameters
        ----------
        task_id
            Unique identifier — used to release or cancel later.
        workflow_id / task_type
            Metadata used for preemption selection and notifications.
        priority
            Higher value = higher priority.  Requests preempt only tasks
            with **strictly lower** priority.
        cpus, gpus
            Resources needed.
        on_granted
            Called when resources are reserved.  Start execution here.
        on_preempted
            Called if this task is preempted after being granted.
            Resources are freed automatically; do **not** call ``release``.
        """

        if self.debug:
            self._log.debug(
                f"request: task_id={task_id} wf={workflow_id} type={task_type} "
                f"priority={priority} cpus={cpus} gpus={gpus} "
                f"[avail cpus={self.available_cpus} gpus={self.available_gpus}]"
            )
        req = _Request(
            neg_priority = -priority,
            seq          = next(self._seq),
            task_id      = task_id,
            workflow_id  = workflow_id,
            task_type    = task_type,
            cpus         = cpus,
            gpus         = gpus,
            on_granted   = on_granted,
            on_preempted = on_preempted,
        )
        granted, preempted = self._enqueue_locked(req)
        _fire(granted, preempted)

    def release(self, task_id: str) -> None:
        """
        Release resources held by a finished task.

        No-op if the task was already preempted or is unknown.
        """
        granted, preempted = self._release_locked(task_id)
        _fire(granted, preempted)

    def release_by_workflow(self, workflow_id: str) -> int:
        """
        Release all *running* tasks for ``workflow_id``.

        Use this when a workflow exits without calling ``release`` for each
        task (e.g. the workflow completed normally but never cleaned up its
        RM slots).  Returns the number of tasks released.
        """
        with self._lock:
            to_release = [t for t in list(self._running.values())
                          if t.workflow_id == workflow_id]
            for task in to_release:
                self._avail_cpus += task.cpus
                self._avail_gpus += task.gpus
                del self._running[task.task_id]
                if self.debug:
                    self._log.debug(
                        f"release_by_workflow: released task_id={task.task_id} "
                        f"cpus={task.cpus} gpus={task.gpus} "
                        f"→ avail cpus={self._avail_cpus} gpus={self._avail_gpus}"
                    )
            granted, preempted = self._try_dispatch_locked()
        _fire(granted, preempted)
        if self.debug:
            self._log.debug(
                f"release_by_workflow: workflow_id={workflow_id} released={len(to_release)}"
            )
        return len(to_release)

    def cancel(self, task_id: str) -> bool:
        """
        Remove a *pending* (not-yet-granted) request.

        Returns True if the request was found and removed.
        """
        with self._lock:
            before = len(self._pending)
            self._pending = [r for r in self._pending if r.task_id != task_id]
            removed = len(self._pending) < before
            if removed:
                heapq.heapify(self._pending)
                self._logged_pending.discard(task_id)
        if removed:
            self._log.debug(f"cancel: removed pending task_id={task_id}")
        else:
            self._log.debug(f"cancel: task_id={task_id} not found in pending")
        return removed

    def cancel_by_workflow(self, workflow_id: str) -> int:
        """
        Remove all *pending* requests for ``workflow_id``.

        Returns the number of requests removed.
        """
        with self._lock:
            before = len(self._pending)
            removed_ids = {r.task_id for r in self._pending if r.workflow_id == workflow_id}
            self._pending = [r for r in self._pending if r.workflow_id != workflow_id]
            removed = before - len(self._pending)
            if removed:
                heapq.heapify(self._pending)
                self._logged_pending -= removed_ids
        self._log.info(
            f"cancel_by_workflow: workflow_id={workflow_id} removed={removed}"
        )
        return removed

    def close(self) -> None:
        """
        Shut down the ResourceManager.

        - Cancels all *pending* requests: their ``on_preempted`` callbacks
          are fired so that any coroutines blocked in ``_wait_for_resource``
          are unblocked and can clean up.
        - Preempts all *running* tasks: their ``on_preempted`` callbacks are
          fired so the callers know the slot has been reclaimed.
        - Resets available resources back to totals.

        After ``close`` the manager is in a clean idle state and must not
        be used for new requests.
        """
        self._log.info(
            f"close: cancelling {len(self._pending)} pending, "
            f"preempting {len(self._running)} running"
        )

        with self._lock:
            # Collect pending requests — fire on_preempted to unblock waiters.
            pending_to_notify = list(self._pending)
            self._pending.clear()
            self._logged_pending.clear()

            # Collect running tasks — fire on_preempted to notify callers.
            running_to_notify = list(self._running.values())
            self._running.clear()

            # Reset resource counters.
            self._avail_cpus = self._total_cpus
            self._avail_gpus = self._total_gpus

        # Fire callbacks outside the lock.
        for req in pending_to_notify:
            try:
                req.on_preempted()
            except Exception as exc:
                self._log.warning(f"close: on_preempted error for pending {req.task_id}: {exc}")

        for task in running_to_notify:
            try:
                task.on_preempted()
            except Exception as exc:
                self._log.warning(f"close: on_preempted error for running {task.task_id}: {exc}")

        self._log.info("close: done")

    # ------------------------------------------------------------------
    # Internal — lock-acquiring wrappers
    # ------------------------------------------------------------------

    def _enqueue_locked(
        self, req: _Request
    ) -> "Tuple[List[_Request], List[_RunningTask]]":
        with self._lock:
            heapq.heappush(self._pending, req)
            return self._try_dispatch_locked()

    def _release_locked(
        self, task_id: str
    ) -> "Tuple[List[_Request], List[_RunningTask]]":
        with self._lock:
            task = self._running.pop(task_id, None)
            if task is not None:
                self._avail_cpus += task.cpus
                self._avail_gpus += task.gpus
                if self.debug:
                    self._log.debug(
                        f"released: task_id={task_id} cpus={task.cpus} gpus={task.gpus} → avail cpus={self._avail_cpus} gpus={self._avail_gpus}",
                    )
            else:
                if self.debug:
                    self._log.warning(f"release: task_id={task_id} not in running (already preempted?)")

            return self._try_dispatch_locked()

    # ------------------------------------------------------------------
    # Internal — scheduling (lock MUST be held)
    # ------------------------------------------------------------------

    def _try_dispatch_locked(
        self,
    ) -> "Tuple[List[_Request], List[_RunningTask]]":
        """
        Greedy dispatch loop: highest-priority first with preemption fallback.
        Returns (granted_requests, preempted_tasks) for the caller to fire
        outside the lock.
        """
        granted:   List[_Request]     = []
        preempted: List[_RunningTask] = []

        while self._pending:
            req = self._pending[0]

            if self._avail_cpus >= req.cpus and self._avail_gpus >= req.gpus - _GPU_EPS:
                heapq.heappop(self._pending)
                self._grant_locked(req)
                granted.append(req)

            else:
                victims = self._find_victims_locked(
                    req.cpus, req.gpus, -req.neg_priority
                )
                if victims:
                    heapq.heappop(self._pending)
                    self._log.info(
                        f"preempting {len(victims)} task(s) for task_id={req.task_id} wf={req.workflow_id} type={req.task_type} priority={-req.neg_priority}",
                    )
                    for v in victims:
                        self._avail_cpus += v.cpus
                        self._avail_gpus += v.gpus
                        del self._running[v.task_id]
                        self._log.info(
                            f"  preempted: task_id={v.task_id} wf={v.workflow_id} type={v.task_type} priority={v.priority} cpus={v.cpus} gpus={v.gpus}",
                        )
                        preempted.append(v)
                    self._grant_locked(req)
                    granted.append(req)
                else:
                    if req.task_id not in self._logged_pending:
                        self._logged_pending.add(req.task_id)
                        if self.debug:
                            self._log.debug(
                                f"queued: task_id={req.task_id} wf={req.workflow_id} type={req.task_type} priority={-req.neg_priority} "
                                f"(need cpus={req.cpus} gpus={req.gpus}, avail cpus={self._avail_cpus} gpus={self._avail_gpus}, no eligible victims)",
                            )
                    break   # head can't run; lower-priority items won't either

        return granted, preempted

    def _grant_locked(self, req: _Request) -> None:
        """Reserve resources and record as running.  Lock MUST be held."""
        self._logged_pending.discard(req.task_id)
        self._avail_cpus -= req.cpus
        self._avail_gpus -= req.gpus
        # Snap tiny negative values to 0 to prevent floating-point drift
        # accumulation (e.g. 800 × 0.00125 = 1.0 but introduces ~1e-16 error).
        if self._avail_gpus < 0 and abs(self._avail_gpus) < _GPU_EPS:
            self._avail_gpus = 0.0
        if self.debug:
            self._log.debug(
                f"granted: task_id={req.task_id} wf={req.workflow_id} type={req.task_type} priority={-req.neg_priority} cpus={req.cpus} gpus={req.gpus} "
                f"→ avail cpus={self._avail_cpus} gpus={self._avail_gpus}",
            )
        self._running[req.task_id] = _RunningTask(
            task_id      = req.task_id,
            workflow_id  = req.workflow_id,
            task_type    = req.task_type,
            priority     = -req.neg_priority,
            cpus         = req.cpus,
            gpus         = req.gpus,
            on_preempted = req.on_preempted,
        )

    def _find_victims_locked(
        self, cpus_need: int, gpus_need: float, min_priority: int
    ) -> "List[_RunningTask]":
        """
        Find the minimum set of lowest-priority running tasks to preempt.

        Only tasks with ``priority < min_priority`` are eligible.
        Sorted lowest-priority first; ties broken by most resources freed
        per task (fewer victims needed).
        Returns ``[]`` if no feasible preemption set exists.
        Lock MUST be held.
        """
        candidates = [t for t in self._running.values() if t.priority < min_priority]
        if not candidates:
            return []

        candidates.sort(key=lambda t: (t.priority, -(t.cpus + t.gpus)))

        freed_cpus, freed_gpus = 0, 0.0
        victims: List[_RunningTask] = []

        for task in candidates:
            victims.append(task)
            freed_cpus += task.cpus
            freed_gpus += task.gpus
            if (self._avail_cpus + freed_cpus >= cpus_need and
                    self._avail_gpus + freed_gpus >= gpus_need - _GPU_EPS):
                return victims

        return []   # cannot satisfy even by preempting all candidates


# ---------------------------------------------------------------------------
# Module-level callback helper
# ---------------------------------------------------------------------------


def _fire(
    granted:   "List[_Request]",
    preempted: "List[_RunningTask]",
) -> None:
    """Fire callbacks outside any lock — preempted tasks first, then grants."""
    for task in preempted:
        task.on_preempted()
    for req in granted:
        req.on_granted()
