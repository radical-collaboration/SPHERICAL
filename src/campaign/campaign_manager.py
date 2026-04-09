"""
CampaignManager — orchestrates multiple replicas of one or more workflows.

Concurrency (sliding window)
-----------------------------
``max_replicas`` caps concurrent running replicas.  When one finishes, the CM
starts the next one from the queue up to the cap.

Config structure
----------------
    workflows:
      ddsim:
        priority:      5
        replicas:      8
        min_replicas:  2
        max_replicas:  4
        dependencies:  []

      inference:
        priority:      10
        replicas:      1
        dependencies:  [ddsim]

Workflow authoring
------------------
    class MyWorkflow(BaseWorkflow):
        workflow_id = "my_wf"

        def run(self, replica_id: str) -> None:
            do_simulation()
            do_training()

        def on_replica_done(self, replica_id, cm, final_state):
            if final_state == "done" and self.converged:
                print(f"{replica_id} converged")
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from ..utils.logger import Logger


# ---------------------------------------------------------------------------
# ResourcePool
# ---------------------------------------------------------------------------


@dataclass
class ResourcePool:
    """
    Tracks available CPU cores and GPU slots for the campaign.

    Both counters are optional: a value of 0 for ``total_cpus`` or
    ``total_gpus`` disables tracking for that resource type (unlimited).
    """

    total_cpus: int = 0
    total_gpus: int = 0
    available_cpus: int = field(default=0, init=False)
    available_gpus: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.available_cpus = self.total_cpus
        self.available_gpus = self.total_gpus

    def can_fit(self, cpus: int, gpus: int) -> bool:
        """True when the requested resources are currently available."""
        if self.total_cpus > 0 and cpus > self.available_cpus:
            return False
        if self.total_gpus > 0 and gpus > self.available_gpus:
            return False
        return True

    def allocate(self, cpus: int, gpus: int) -> None:
        self.available_cpus -= cpus
        self.available_gpus -= gpus

    def release(self, cpus: int, gpus: int) -> None:
        self.available_cpus += cpus
        self.available_gpus += gpus

    def usage_str(self) -> str:
        """'cpus=used/total  gpus=used/total' for tracked resource types only."""
        parts = []
        if self.total_cpus > 0:
            parts.append(f"cpus={self.total_cpus - self.available_cpus}/{self.total_cpus}")
        if self.total_gpus > 0:
            parts.append(f"gpus={self.total_gpus - self.available_gpus}/{self.total_gpus}")
        return "  ".join(parts) if parts else "—"

    def available_str(self) -> str:
        """'cpus=avail/total  gpus=avail/total' for tracked resource types only."""
        parts = []
        if self.total_cpus > 0:
            parts.append(f"cpus={self.available_cpus}/{self.total_cpus}")
        if self.total_gpus > 0:
            parts.append(f"gpus={self.available_gpus}/{self.total_gpus}")
        return "  ".join(parts) if parts else "—"

    def as_dict(self) -> dict:
        return {
            "total_cpus": self.total_cpus,
            "available_cpus": self.available_cpus,
            "total_gpus": self.total_gpus,
            "available_gpus": self.available_gpus,
        }


# ---------------------------------------------------------------------------
# BaseWorkflow
# ---------------------------------------------------------------------------


class BaseWorkflow:
    """
    Base class for campaign workflows.

    Subclass contract
    -----------------
    - ``workflow_id``  (class attr, str) : unique prefix for replica IDs.
    - ``run(replica_id)``  **or**
      ``start(replica_id)``             : execute the entire workflow.
                                          Exactly one must be defined.
                                          Runs in a thread-pool worker.
    - ``on_replica_done(replica_id, cm, final_state)`` : (optional) hook called
                                          by the CM after the entry-point
                                          returns or raises.
                                          ``final_state`` is ``"done"`` or
                                          ``"failed"``.
    """

    workflow_id: str = "base"

    def __init__(
        self,
        config: Optional[dict] = None,
        on_ready: Optional[object] = None,
        asyncflow: Optional[object] = None,
        policies: Optional[List] = None,
        engine_dragon: Optional[object] = None,
    ) -> None:
        self.config = config
        # Async callable injected by AsyncCampaignManager; invoke via _signal_ready().
        self._on_ready = on_ready
        # Shared WorkflowEngine injected by AsyncCampaignManager (optional).
        self.asyncflow = asyncflow
        # One Dragon Policy per assigned GPU, injected by AsyncCampaignManager.
        # Single-GPU workflows use policies[0]; multi-GPU workflows cycle through the list.
        self.policies: List = policies or []
        # Dragon backend handle for routing tasks to specific backends.
        self.engine_dragon: Optional[object] = engine_dragon

    async def _signal_ready(self) -> None:
        """
        Signal the CM that this workflow has produced enough data for its
        dependents to start.

        Calls the ``on_ready`` coroutine injected by the CM at construction.
        No-op if no callback was provided.
        """
        import asyncio

        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    def run(self, replica_id: str) -> None:
        """
        Execute the workflow for one replica.

        Override in subclasses (or define ``start`` instead).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.run() not implemented (replica_id={replica_id!r})"
        )

    def on_replica_done(
        self,
        replica_id: str,
        cm: "CampaignManager",
        final_state: str,
    ) -> None:
        """
        Hook called after this replica's entry-point finishes.

        Override to add workflow-specific post-replica logic.  The default
        implementation does nothing.
        """


# ---------------------------------------------------------------------------
# Public stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStats:
    """Cumulative statistics for one workflow group."""

    replicas_started: int = 0
    replicas_finished: int = 0


# ---------------------------------------------------------------------------
# Internal group descriptor
# ---------------------------------------------------------------------------


@dataclass
class _GroupInfo:
    name: str
    workflow_class: Type[BaseWorkflow]
    replicas: int
    dependencies: List[str]
    group_config: Optional[dict]
    priority: int = 0
    min_replicas: int = 0
    max_replicas: int = 0
    required_cpus: int = 0
    required_gpus: int = 0
    entry_point: str = "run"
    status: str = "pending"
    started_count: int = 0
    running_count: int = 0
    finished_replicas: int = 0


# ---------------------------------------------------------------------------
# CampaignManager
# ---------------------------------------------------------------------------


class CampaignManager:
    """
    Orchestrates multiple replicas of one or more :class:`BaseWorkflow`
    subclasses using a sliding-window concurrency model.

    Parameters
    ----------
    max_workers : int, optional
        Thread-pool size.  Defaults to 8.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        total_cpus: int = 0,
        total_gpus: int = 0,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers or 8)
        self._log = Logger(name="CampaignManager", use_colors=True)
        self._resources = ResourcePool(total_cpus=total_cpus, total_gpus=total_gpus)

        self._lock = threading.Lock()

        self._groups: Dict[str, _GroupInfo] = {}
        self._replica_meta: Dict[str, Tuple[str, int]] = {}
        self._active_futures: Dict[str, Future] = {}
        self._stats: Dict[str, WorkflowStats] = {}

        self._all_done = threading.Event()

        self._log.info("CampaignManager initialised")

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        workflow_registry: Dict[str, Type[BaseWorkflow]],
    ) -> "CampaignManager":
        """
        Build a CampaignManager from a config dict and a workflow registry.

        Parameters
        ----------
        config
            Parsed config dict.  ``max_workers`` is optional; ``workflows``
            maps group names to per-workflow settings.
        workflow_registry
            ``{name: BaseWorkflow subclass}``.
        """
        res_cfg = config.get("resources", {})
        cm = cls(
            max_workers=config.get("max_workers"),
            total_cpus=int(res_cfg.get("total_cpus", 0)),
            total_gpus=int(res_cfg.get("total_gpus", 0)),
        )

        _cm_keys = {
            "replicas",
            "dependencies",
            "priority",
            "min_replicas",
            "max_replicas",
            "required_cpus",
            "required_gpus",
        }

        for name, wf_cfg in config.get("workflows", {}).items():
            wf_class = workflow_registry.get(name)
            if wf_class is None:
                cm._log.warning(f"from_config: no class registered for {name!r} — skipping")
                continue

            cm.register_group(
                name=name,
                workflow_class=wf_class,
                replicas=int(wf_cfg.get("replicas", 1)),
                dependencies=list(wf_cfg.get("dependencies", [])),
                priority=int(wf_cfg.get("priority", 0)),
                min_replicas=int(wf_cfg.get("min_replicas", 0)),
                max_replicas=int(wf_cfg.get("max_replicas", 0)),
                required_cpus=int(wf_cfg.get("required_cpus", 0)),
                required_gpus=int(wf_cfg.get("required_gpus", 0)),
                config={k: v for k, v in wf_cfg.items() if k not in _cm_keys} or None,
            )

        return cm

    # ------------------------------------------------------------------
    # Group-based API
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_entry_point(workflow_class: Type[BaseWorkflow]) -> str:
        has_run = workflow_class.run is not BaseWorkflow.run
        has_start = hasattr(workflow_class, "start")

        if has_run and has_start:
            raise ValueError(
                f"{workflow_class.__name__} defines both 'run' and 'start' — "
                "choose exactly one as the workflow entry point"
            )
        if not has_run and not has_start:
            raise ValueError(f"{workflow_class.__name__} must define either 'run' or 'start'")
        return "run" if has_run else "start"

    def register_group(
        self,
        name: str,
        workflow_class: Type[BaseWorkflow],
        replicas: int = 1,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        min_replicas: int = 0,
        max_replicas: int = 0,
        required_cpus: int = 0,
        required_gpus: int = 0,
        config: Optional[dict] = None,
    ) -> None:
        """
        Register a workflow group without starting it yet.

        Parameters
        ----------
        name
            Unique group name (referenced in ``dependencies`` lists).
        workflow_class
            Subclass of :class:`BaseWorkflow`.
        replicas
            Total replicas to complete.
        dependencies
            Group names that must finish before this group starts.
        priority
            Workflow-level priority (informational; used for logging).
        min_replicas
            Minimum concurrent slots (reserved; logged but not enforced by CM).
        max_replicas
            Concurrent cap (sliding window).  ``0`` = all at once.
        config
            Optional dict forwarded to each workflow constructor.
        """
        entry_point = self._resolve_entry_point(workflow_class)
        effective_max = max_replicas if max_replicas > 0 else replicas

        with self._lock:
            self._groups[name] = _GroupInfo(
                name=name,
                workflow_class=workflow_class,
                replicas=replicas,
                dependencies=list(dependencies or []),
                group_config=config,
                priority=priority,
                min_replicas=min_replicas,
                max_replicas=effective_max,
                required_cpus=required_cpus,
                required_gpus=required_gpus,
                entry_point=entry_point,
            )
            self._stats[name] = WorkflowStats()

        self._log.info(
            f"Registered group {name!r}: replicas={replicas} "
            f"priority={priority} min={min_replicas} max={effective_max} "
            f"deps={dependencies or []} "
            f"resources=(cpus={required_cpus}, gpus={required_gpus})"
        )

    def start(self) -> None:
        """
        Start all registered groups whose dependencies are satisfied.

        Dependent groups start automatically as their dependencies complete.
        """
        res = self._resources
        if res.total_cpus > 0 or res.total_gpus > 0:
            self._log.info(
                f"Resource pool: total_cpus={res.total_cpus}  total_gpus={res.total_gpus}"
            )

        with self._lock:
            ready = [
                g
                for g in self._groups.values()
                if g.status == "pending" and self._deps_satisfied_locked(g)
            ]
            for g in ready:
                g.status = "running"

        for g in ready:
            self._launch_group(g)

        with self._lock:
            if not self._groups:
                self._all_done.set()

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Snapshot of campaign progress."""
        with self._lock:
            return {
                "resources": self._resources.as_dict(),
                "groups": {
                    name: {
                        "status": g.status,
                        "priority": g.priority,
                        "replicas_total": g.replicas,
                        "replicas_started": g.started_count,
                        "replicas_running": g.running_count,
                        "replicas_finished": g.finished_replicas,
                        "min_replicas": g.min_replicas,
                        "max_replicas": g.max_replicas,
                        "required_cpus": g.required_cpus,
                        "required_gpus": g.required_gpus,
                        "dependencies": g.dependencies,
                    }
                    for name, g in self._groups.items()
                },
            }

    def stats(self) -> Dict[str, WorkflowStats]:
        """Per-workflow-group statistics (snapshot)."""
        with self._lock:
            return {name: WorkflowStats(**vars(s)) for name, s in self._stats.items()}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until all workflow groups have finished."""
        return self._all_done.wait(timeout=timeout)

    def close(self) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)
        self._log.info("CampaignManager closed")

    # ------------------------------------------------------------------
    # Internal — group management
    # ------------------------------------------------------------------

    def _deps_satisfied_locked(self, group: _GroupInfo) -> bool:
        for dep in group.dependencies:
            dep_group = self._groups.get(dep)
            if dep_group is None or dep_group.status != "done":
                return False
        return True

    def _launch_group(self, group: _GroupInfo) -> None:
        to_start = []
        with self._lock:
            while (
                group.started_count < group.replicas
                and group.running_count < group.max_replicas
                and self._resources.can_fit(group.required_cpus, group.required_gpus)
            ):
                idx = group.started_count
                group.started_count += 1
                group.running_count += 1
                self._resources.allocate(group.required_cpus, group.required_gpus)
                if group.name in self._stats:
                    self._stats[group.name].replicas_started = group.started_count
                to_start.append(idx)

        res_tag = ""
        if group.required_cpus > 0 or group.required_gpus > 0:
            res_tag = (
                f" [each: cpus={group.required_cpus} gpus={group.required_gpus}"
                f" | after alloc: {self._resources.usage_str()}]"
            )
        self._log.info(
            f"Launching group {group.name!r}: "
            f"queuing {len(to_start)}/{group.replicas} replica(s) "
            f"(priority={group.priority}){res_tag}"
        )
        for i in to_start:
            self._submit_replica(group, i)

    def _submit_replica(self, group: _GroupInfo, replica_idx: int) -> None:
        """Submit one replica directly to the thread pool."""
        replica_id = f"{group.name}_{replica_idx}"
        wf = group.workflow_class(config=group.group_config)

        with self._lock:
            self._replica_meta[replica_id] = (group.name, replica_idx)

        res_tag = ""
        if group.required_cpus > 0 or group.required_gpus > 0:
            res_tag = f" [cpus={group.required_cpus} gpus={group.required_gpus}]"
        self._log.info(f"  submitting replica {replica_id!r}{res_tag}")

        fut = self._executor.submit(
            self._run_replica,
            wf,
            replica_id,
            group.entry_point,
        )
        with self._lock:
            self._active_futures[replica_id] = fut

    def _try_start_dependents(self, finished_group_name: str) -> None:
        with self._lock:
            to_start = [
                g
                for g in self._groups.values()
                if g.status == "pending" and self._deps_satisfied_locked(g)
            ]
            for g in to_start:
                g.status = "running"

        for g in to_start:
            self._log.info(f"Group {g.name!r} unblocked by {finished_group_name!r}")
            self._launch_group(g)

    # ------------------------------------------------------------------
    # Internal — replica execution
    # ------------------------------------------------------------------

    def _run_replica(
        self,
        wf: BaseWorkflow,
        replica_id: str,
        entry_point: str,
    ) -> None:
        """Execute the workflow entry-point in a thread-pool worker."""
        try:
            getattr(wf, entry_point)(replica_id)
            final_state = "done"
        except Exception as exc:
            self._log.error(f"Replica {replica_id!r} raised: {exc}")
            final_state = "failed"
        finally:
            with self._lock:
                self._active_futures.pop(replica_id, None)

        self._handle_replica_done(wf, replica_id, final_state)

    def _handle_replica_done(self, wf: BaseWorkflow, replica_id: str, final_state: str) -> None:
        try:
            wf.on_replica_done(replica_id, self, final_state)
        except Exception as exc:
            self._log.error(f"Replica {replica_id!r} on_replica_done raised: {exc}")

        self._on_replica_finished(replica_id)

    def _on_replica_finished(self, replica_id: str) -> None:
        group_done = False
        next_replica_idx: Optional[int] = None
        group_name_local: Optional[str] = None
        release_tag = ""
        stalled = False

        with self._lock:
            meta = self._replica_meta.get(replica_id)
            if meta is None:
                return
            group_name_local, _ = meta

            g = self._groups.get(group_name_local)
            if g is None:
                return

            g.finished_replicas += 1
            g.running_count -= 1
            self._resources.release(g.required_cpus, g.required_gpus)

            if g.required_cpus > 0 or g.required_gpus > 0:
                release_tag = (
                    f" | released cpus={g.required_cpus} gpus={g.required_gpus}"
                    f" | available: {self._resources.available_str()}"
                )

            if group_name_local in self._stats:
                self._stats[group_name_local].replicas_finished = g.finished_replicas

            if (
                g.started_count < g.replicas
                and g.running_count < g.max_replicas
                and self._resources.can_fit(g.required_cpus, g.required_gpus)
            ):
                next_replica_idx = g.started_count
                g.started_count += 1
                g.running_count += 1
                self._resources.allocate(g.required_cpus, g.required_gpus)
                if group_name_local in self._stats:
                    self._stats[group_name_local].replicas_started = g.started_count
            elif (
                g.started_count < g.replicas
                and g.running_count < g.max_replicas
                and not self._resources.can_fit(g.required_cpus, g.required_gpus)
            ):
                stalled = True

            if g.finished_replicas >= g.replicas:
                g.status = "done"
                group_done = True

        self._log.info(f"Replica {replica_id!r} finished{release_tag}")
        if stalled and group_name_local:
            g = self._groups[group_name_local]
            self._log.warning(
                f"Group {group_name_local!r} stalled — waiting for resources "
                f"(needs cpus={g.required_cpus} gpus={g.required_gpus}  "
                f"available: {self._resources.available_str()})"
            )

        if next_replica_idx is not None and group_name_local:
            self._submit_replica(self._groups[group_name_local], next_replica_idx)

        if group_done and group_name_local:
            self._log.info(f"Workflow group {group_name_local!r} completed")
            self._try_start_dependents(group_name_local)

        with self._lock:
            all_done = bool(self._groups) and all(g.status == "done" for g in self._groups.values())

        if all_done:
            self._all_done.set()
            self._log.info("All campaign workflow groups finished")
