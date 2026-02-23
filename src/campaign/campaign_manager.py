#!/usr/bin/env python3
"""
CampaignManager — class-based, chain-aware workflow scheduler.

Each workflow is a subclass of BaseWorkflow that defines its tasks,
resources, executors, and chaining logic as methods on the class.

Execution model
---------------
When a task group is dispatched the registered executor callable is invoked
in a thread-pool worker:

    executor(task_desc: dict) -> Any

where task_desc contains:

    {
        "ranks":          int,
        "cores_per_rank": int,
        "gpus_per_rank":  float,
        "pre_exec":       list[str],
        "shell":          bool,
        "workflow_id":    str,
        "task_type":      str,
    }

Task chaining
-------------
When every task in a group finishes, the campaign manager resolves
on_completion in this priority order:

  1. Callable passed directly to submit()
     Signature: fn(final_state: str, cm: CampaignManager, workflow_id: str)

  2. Callable returned by BaseWorkflow.on_completion_for(task_type)
     (default: looks for method after_<task_type> on the workflow instance)

  3. String from TaskSpec.on_completion field
     → automatically submits that task type in the same workflow.

Config schema
-------------
{
    "run_description": {"nodes": 1},
    "node_description": {"cores": 64, "gpus": 8},
    "settings":         {"batch_size": 500, "dry_run": false}
}
"""

import concurrent.futures
import heapq
import itertools
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..utils.logger import Logger
from .base_workflow import BaseWorkflow


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass
class _TaskSpec:
    """Internal task-type definition derived from a BaseWorkflow."""
    task_type:      str
    workflow_id:    str
    priority:       int
    ranks:          int
    cores_per_rank: int
    gpus_per_rank:  float
    pre_exec:       List[str]
    shell:          bool
    on_completion:  Optional[str]       # next task_type name (string chain)

    def build_task_desc(self) -> dict:
        """Return the standard task description dict passed to executors."""
        return {
            "ranks":          self.ranks,
            "cores_per_rank": self.cores_per_rank,
            "gpus_per_rank":  self.gpus_per_rank,
            "pre_exec":       self.pre_exec,
            "shell":          self.shell,
            "workflow_id":    self.workflow_id,
            "task_type":      self.task_type,
        }


@dataclass
class _TaskGroup:
    """An actively running group of task instances."""
    gid:            str
    workflow_id:    str
    task_type:      str
    priority:       int
    spec:           _TaskSpec
    cpus:           int
    gpus:           float
    futures:        List[concurrent.futures.Future]
    remaining:      int
    on_completion:  Optional[Union[str, Callable]]
    done_cb:        Optional[Callable]


@dataclass(order=True)
class _Pending:
    """Entry in the priority min-heap (higher priority = lower neg_priority)."""
    neg_priority:  int
    seq:           int
    spec:          _TaskSpec                        = field(compare=False)
    count:         int                              = field(compare=False, default=1)
    execute:       Optional[Callable]               = field(compare=False, default=None)
    on_completion: Optional[Union[str, Callable]]   = field(compare=False, default=None)
    done_cb:       Optional[Callable]               = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _noop() -> None:
    """Placeholder executor for dry_run / missing executors."""


# ---------------------------------------------------------------------------
# CampaignManager
# ---------------------------------------------------------------------------


class CampaignManager:
    """
    Priority-aware scheduler for chained multi-workflow campaigns.

    Parameters
    ----------
    config : dict | str | Path
        Resource configuration.  Only ``run_description``,
        ``node_description``, and ``settings`` are used; workflow
        definitions are now provided via :meth:`run_workflow`.
    """

    # ------------------------------------------------------------------
    def __init__(self, config: "dict | str | Path"):
        cfg: dict
        if isinstance(config, (str, Path)):
            with open(config) as fh:
                cfg = json.load(fh)
        else:
            cfg = config

        self._config = cfg

        # ---- resource budget -----------------------------------------
        run_desc        = cfg.get("run_description", {})
        node_desc       = cfg.get("node_description", {})
        n_nodes         = int(run_desc.get("nodes", 1))
        self._total_cpus: int   = n_nodes * int(node_desc.get("cores", 1))
        self._total_gpus: float = n_nodes * float(node_desc.get("gpus", 0))
        self._resources         = [self._total_cpus, self._total_gpus]

        # ---- settings ------------------------------------------------
        settings         = cfg.get("settings", {})
        self._batch_size = int(settings.get("batch_size", 1))
        self._dry_run    = bool(settings.get("dry_run", False))

        # ---- workflow/task registry ----------------------------------
        self._workflows: Dict[str, dict] = {}

        # ---- per-workflow runtime hooks ------------------------------
        # workflow_id -> {task_type: callable}
        self._executors:            Dict[str, Dict[str, Callable]] = {}
        self._completion_overrides: Dict[str, Dict[str, Callable]] = {}

        # ---- scheduling state ----------------------------------------
        self._lock    = threading.Lock()
        self._seq     = itertools.count()
        self._pending: List[_Pending]        = []
        self._active:  Dict[str, _TaskGroup] = {}
        self._inflight = 0
        self._all_done = threading.Event()
        self._all_done.set()

        # ---- thread pool ---------------------------------------------
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(self._total_cpus, 4),
            thread_name_prefix="campaign",
        )

        # ---- logger --------------------------------------------------
        self._log = Logger(name="CampaignManager", use_colors=True)
        self._log.info(
            f"Initialized: {n_nodes} node(s), {self._total_cpus} CPUs, "
            f"{self._total_gpus} GPUs"
            + (" [dry-run]" if self._dry_run else "")
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_cpus(self) -> int:
        return self._resources[0]

    @property
    def available_gpus(self) -> float:
        return self._resources[1]

    @property
    def total_cpus(self) -> int:
        return self._total_cpus

    @property
    def total_gpus(self) -> float:
        return self._total_gpus

    @property
    def workflows(self) -> Dict[str, dict]:
        """Registered workflows: {workflow_id: {init_tasks, tasks}}."""
        return {
            wf_id: {
                "init_tasks": wf["init_tasks"],
                "tasks":      list(wf["tasks"]),
            }
            for wf_id, wf in self._workflows.items()
        }

    # ------------------------------------------------------------------
    # Workflow entry point
    # ------------------------------------------------------------------

    def run_workflow(self, workflow: BaseWorkflow) -> None:
        """
        Register and start a workflow.

        The workflow's ``init_tasks`` are submitted immediately.
        Subsequent tasks are chained via the workflow's
        ``on_completion_for`` / ``after_<task_type>`` methods or
        ``TaskSpec.on_completion`` strings.

        Parameters
        ----------
        workflow
            Instance of a :class:`~src.campaign.base_workflow.BaseWorkflow`
            subclass.
        """
        self._register_workflow(workflow)
        wf = self._workflows[workflow.workflow_id]
        self._log.info(
            f"[{workflow.workflow_id}] Starting workflow — "
            f"init_tasks={wf['init_tasks']}"
        )
        for task_type in wf["init_tasks"]:
            self._submit_task(workflow.workflow_id, task_type)

    # ------------------------------------------------------------------
    # Direct submission
    # ------------------------------------------------------------------

    def submit(
        self,
        task_type:     str,
        workflow_id:   str,
        count:         int = 1,
        priority:      Optional[int] = None,
        done_cb:       Optional[Callable] = None,
        on_completion: Optional[Union[str, Callable]] = None,
    ) -> None:
        """
        Submit ``count`` parallel instances of a task type.

        ``on_completion`` fires once when ALL instances in the group finish.

        Parameters
        ----------
        task_type
            Must be registered for the workflow.
        workflow_id
            Owning workflow (must have been started with run_workflow).
        count
            Number of parallel task instances in the group.
        priority
            Explicit priority (overrides TaskSpec value).
        done_cb
            ``callable(final_state)`` — one-time notification for this group.
        on_completion
            Overrides chain for this submission only.
            ``str``  → submit that task_type when the group finishes.
            ``fn``   → ``fn(final_state, campaign_manager, workflow_id)``.
        """
        self._submit_task(
            workflow_id=workflow_id,
            task_type=task_type,
            count=count,
            priority=priority,
            done_cb=done_cb,
            on_completion=on_completion,
        )

    # ------------------------------------------------------------------
    # Cancellation / lifecycle
    # ------------------------------------------------------------------

    def cancel_workflow(self, workflow_id: str) -> None:
        """
        Drop all *pending* (not yet running) groups for a workflow.
        Running groups finish normally; their done_cb is NOT suppressed.
        """
        with self._lock:
            kept    = [e for e in self._pending if e.spec.workflow_id != workflow_id]
            removed = len(self._pending) - len(kept)
            heapq.heapify(kept)
            self._pending   = kept
            self._inflight -= removed
            if self._inflight <= 0 and not self._active:
                self._inflight = 0
                self._all_done.set()
        self._log.warning(
            f"[{workflow_id}] Cancelled {removed} pending group(s)"
        )

    def wait_all(self) -> None:
        """Block until every submitted group has reached a final state."""
        self._all_done.wait()

    def close(self) -> None:
        """Shutdown the thread pool, waiting for running tasks to finish."""
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internal — workflow registration
    # ------------------------------------------------------------------

    def _register_workflow(self, workflow: BaseWorkflow) -> None:
        """
        Build internal _TaskSpec objects from a BaseWorkflow instance and
        register its executors and on_completion overrides.
        """
        wf_id = workflow.workflow_id
        specs = workflow.task_specs()

        tasks: Dict[str, _TaskSpec] = {}
        for task_type, ts in specs.items():
            tasks[task_type] = _TaskSpec(
                task_type      = task_type,
                workflow_id    = wf_id,
                priority       = ts.priority,
                ranks          = ts.ranks,
                cores_per_rank = ts.cores_per_rank,
                gpus_per_rank  = ts.gpus_per_rank,
                pre_exec       = list(ts.pre_exec),
                shell          = ts.shell,
                on_completion  = ts.on_completion,
            )

        self._workflows[wf_id] = {
            "init_tasks": list(workflow.init_tasks),
            "tasks":      tasks,
        }

        # Executors: method run_<task_type> on the workflow instance (or None)
        self._executors[wf_id] = {
            task_type: fn
            for task_type in specs
            if (fn := workflow.executor_for(task_type)) is not None
        }

        # on_completion overrides: method after_<task_type> (or None → use string chain)
        self._completion_overrides[wf_id] = {
            task_type: fn
            for task_type in specs
            if (fn := workflow.on_completion_for(task_type)) is not None
        }

    # ------------------------------------------------------------------
    # Internal — submission
    # ------------------------------------------------------------------

    def _submit_task(
        self,
        workflow_id:   str,
        task_type:     str,
        count:         int = 1,
        priority:      Optional[int] = None,
        done_cb:       Optional[Callable] = None,
        on_completion: Optional[Union[str, Callable]] = None,
    ) -> None:
        spec     = self._get_spec(workflow_id, task_type)
        eff_prio = priority if priority is not None else spec.priority
        execute  = self._executors.get(workflow_id, {}).get(task_type)

        to_start: List[Tuple[_TaskGroup, _Pending]] = []

        with self._lock:
            if self._inflight == 0:
                self._all_done.clear()
            self._inflight += 1

            entry = _Pending(
                neg_priority  = -eff_prio,
                seq           = next(self._seq),
                spec          = spec,
                count         = count,
                execute       = execute,
                on_completion = on_completion,
                done_cb       = done_cb,
            )
            heapq.heappush(self._pending, entry)
            to_start = self._try_schedule_locked()

        for group, entry in to_start:
            self._start_group(group, entry)

    def _get_spec(self, workflow_id: str, task_type: str) -> _TaskSpec:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            raise KeyError(f"Unknown workflow: {workflow_id!r}")
        spec = wf["tasks"].get(task_type)
        if spec is None:
            raise KeyError(
                f"Unknown task type {task_type!r} in workflow {workflow_id!r}"
            )
        return spec

    # ------------------------------------------------------------------
    # Internal — scheduling (lock must be held for _try_schedule_locked)
    # ------------------------------------------------------------------

    @staticmethod
    def _group_cpus(spec: _TaskSpec, count: int) -> int:
        return count * spec.ranks * spec.cores_per_rank

    @staticmethod
    def _group_gpus(spec: _TaskSpec, count: int) -> float:
        return count * spec.ranks * spec.gpus_per_rank

    def _try_schedule_locked(self) -> "List[Tuple[_TaskGroup, _Pending]]":
        """
        Allocate resources for all ready-to-run pending groups.

        Lock MUST be held.  Does NOT submit tasks to the pool — returns
        (group, entry) pairs so the caller can start them after releasing
        the lock (preventing add_done_callback from firing synchronously
        while the lock is still held, which would cause deadlock).
        """
        to_start: List[Tuple[_TaskGroup, _Pending]] = []

        while self._pending:
            top       = self._pending[0]
            cpus_need = self._group_cpus(top.spec, top.count)
            gpus_need = self._group_gpus(top.spec, top.count)
            priority  = -top.neg_priority

            if self._resources[0] >= cpus_need and self._resources[1] >= gpus_need:
                heapq.heappop(self._pending)
                group = self._allocate_locked(top)
                to_start.append((group, top))
            elif self._preempt_pending_locked(cpus_need, gpus_need, priority):
                continue     # queue reordered — retry
            else:
                break

        return to_start

    def _allocate_locked(self, entry: _Pending) -> _TaskGroup:
        """
        Reserve resources and create the _TaskGroup.  Lock MUST be held.
        Does NOT touch the pool.
        """
        spec = entry.spec
        cpus = self._group_cpus(spec, entry.count)
        gpus = self._group_gpus(spec, entry.count)

        self._resources[0] -= cpus
        self._resources[1] -= gpus

        gid = f"g{next(self._seq)}"

        # Resolve on_completion: per-submit → workflow-override → TaskSpec string
        wf_override = self._completion_overrides.get(spec.workflow_id, {})
        completion  = (entry.on_completion
                       or wf_override.get(spec.task_type)
                       or spec.on_completion)

        group = _TaskGroup(
            gid           = gid,
            workflow_id   = spec.workflow_id,
            task_type     = spec.task_type,
            priority      = -entry.neg_priority,
            spec          = spec,
            cpus          = cpus,
            gpus          = gpus,
            futures       = [],
            remaining     = entry.count,
            on_completion = completion,
            done_cb       = entry.done_cb,
        )
        self._active[gid] = group
        self._log.task_started(
            f"[{spec.workflow_id}/{spec.task_type}] gid={gid} "
            f"priority={-entry.neg_priority} count={entry.count} "
            f"cpus={cpus} gpus={gpus} "
            f"(avail: {self._resources[0]} CPUs, {self._resources[1]} GPUs)"
        )
        return group

    def _start_group(self, group: _TaskGroup, entry: _Pending) -> None:
        """
        Submit tasks to the thread pool and register done-callbacks.
        Must be called WITHOUT the lock so callbacks never fire while the
        lock is held (avoids deadlock from synchronous callback on fast tasks).
        """
        task_desc = group.spec.build_task_desc()
        execute   = entry.execute

        for _ in range(entry.count):
            fn  = _noop if (execute is None or self._dry_run) else (
                lambda td=task_desc: execute(td)
            )
            fut = self._pool.submit(fn)
            group.futures.append(fut)
            fut.add_done_callback(self._make_done_cb(group.gid))

    # ------------------------------------------------------------------
    # Internal — completion
    # ------------------------------------------------------------------

    def _make_done_cb(self, gid: str) -> Callable:
        """Return a future-done callback bound to gid."""
        def _cb(future: concurrent.futures.Future) -> None:
            del future          # required by add_done_callback; not used here
            self._on_task_done(gid)
        return _cb

    def _on_task_done(self, gid: str) -> None:
        """Called from a pool thread when one task in a group finishes."""
        done_cb     = None
        completion  = None
        final_state = None
        workflow_id = None
        task_type   = None
        to_start:   List[Tuple[_TaskGroup, _Pending]] = []

        with self._lock:
            group = self._active.get(gid)
            if group is None:
                return

            group.remaining -= 1
            if group.remaining > 0:
                return      # wait for remaining tasks in the group

            errors = [f for f in group.futures if f.exception() is not None]
            if len(errors) == len(group.futures):
                final_state = "failed"
            elif errors:
                final_state = "mixed_final"
            else:
                final_state = "done"

            done_cb     = group.done_cb
            completion  = group.on_completion
            workflow_id = group.workflow_id
            task_type   = group.task_type

            del self._active[gid]
            self._resources[0] += group.cpus
            self._resources[1] += group.gpus

            self._inflight -= 1

            # Reserve a slot so _all_done is not set before the on_completion
            # callback has had a chance to submit the next task.
            if completion is not None:
                self._inflight += 1

            to_start = self._try_schedule_locked()

            if self._inflight <= 0 and not self._pending and not self._active:
                self._inflight = 0
                self._all_done.set()

        # Start any newly scheduled groups (outside the lock)
        for grp, ent in to_start:
            self._start_group(grp, ent)

        log = self._log.task_completed if final_state == "done" else self._log.error
        log(f"[{workflow_id}/{task_type}] gid={gid} state={final_state!r}")

        # Invoke user callbacks outside the lock
        if done_cb:
            done_cb(final_state)

        if completion is not None:
            try:
                if callable(completion):
                    completion(final_state, self, workflow_id)
                else:
                    self._submit_task(workflow_id, completion)
            finally:
                # Release the reservation made above
                to_start_after: List[Tuple[_TaskGroup, _Pending]] = []
                with self._lock:
                    self._inflight -= 1
                    to_start_after = self._try_schedule_locked()
                    if self._inflight <= 0 and not self._pending and not self._active:
                        self._inflight = 0
                        self._all_done.set()
                for grp, ent in to_start_after:
                    self._start_group(grp, ent)

    # ------------------------------------------------------------------
    # Internal — pending-queue preemption
    # ------------------------------------------------------------------

    def _preempt_pending_locked(
        self, cpus_need: int, gpus_need: float, min_priority: int
    ) -> bool:
        """
        Requeue lower-priority *pending* entries with fresh seq numbers so a
        higher-priority group jumps to the front of the queue once resources
        free up.  Returns True if any reordering was done.
        """
        candidates = [e for e in self._pending if -e.neg_priority < min_priority]
        if not candidates:
            return False

        freed_cpus, freed_gpus = 0, 0.0
        to_requeue: List[_Pending] = []

        for entry in sorted(candidates, key=lambda e: (-e.neg_priority, e.seq)):
            to_requeue.append(entry)
            freed_cpus += self._group_cpus(entry.spec, entry.count)
            freed_gpus += self._group_gpus(entry.spec, entry.count)
            if (self._resources[0] + freed_cpus >= cpus_need and
                    self._resources[1] + freed_gpus >= gpus_need):
                break
        else:
            return False        # freeing all candidates still not enough

        drop_ids  = {id(e) for e in to_requeue}
        kept      = [e for e in self._pending if id(e) not in drop_ids]
        heapq.heapify(kept)
        self._pending = kept

        for entry in to_requeue:
            heapq.heappush(self._pending, _Pending(
                neg_priority  = entry.neg_priority,
                seq           = next(self._seq),    # new seq → behind equal-priority items
                spec          = entry.spec,
                count         = entry.count,
                execute       = entry.execute,
                on_completion = entry.on_completion,
                done_cb       = entry.done_cb,
            ))

        return True
