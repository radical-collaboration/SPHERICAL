"""
AsyncCampaignManager — async-native campaign orchestrator.

Each workflow replica is an asyncio Task (not a thread).  Supports both
``async def run(replica_id)`` and sync ``def run(replica_id)`` workflow
entry points (sync ones run via ``asyncio.to_thread``).

Scheduling model
----------------
The CM runs a two-pass greedy scheduler on every state change:

  Pass 1 — guarantee ``min_replicas`` for all eligible groups (highest
            priority first).
  Pass 2 — fill remaining capacity up to ``max_replicas`` (highest priority
            first).

A group becomes *eligible* either when a parent workflow calls
``_trigger_dependent()`` (explicit activation) or when each dependency group
has ``dep_threshold`` or more finished replicas / has called ``_signal_done()``
(count-based fallback, default 1).

Usage
-----
    cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)
    await cm.start()   # launch groups with no unmet dependencies
    await cm.wait()    # block until all groups finish
    await cm.close()   # shut down shared asyncflow engine

Workflow authoring
------------------
    class MyWorkflow(BaseWorkflow):
        workflow_id = "my_wf"

        async def run(self, replica_id: str) -> None:
            ...
"""

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Optional

# from .campaign_manager import BaseWorkflow, ResourcePool, WorkflowStats
from ..utils.logger import Logger

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _find_gpus() -> list[tuple[str, int]]:
    """Return [(hostname, gpu_id), ...] for every GPU visible to Dragon.

    Falls back to an empty list when Dragon is not active (e.g. concurrent
    backend during local testing).
    """
    try:
        from dragon.native.machine import Node, System

        gpus = []
        for huid in System().nodes:
            node = Node(huid)
            for gpu_id in node.gpus:
                gpus.append((node.hostname, gpu_id))
        return gpus
    except Exception:
        return []


def _make_policies(gpu_pool: list[tuple[str, int]], gpu_ids: list[int]) -> list:
    """Build a single Dragon Policy that covers all assigned GPU IDs.

    Returns a list with exactly one Policy whose gpu_affinity lists every
    assigned GPU.  Workflows that need per-GPU round-robin (e.g. DDMdWorkflow)
    are responsible for expanding this into individual per-GPU policies
    internally; single-GPU workflows (DummyWorkflow, MiniAppsWorkflow) use
    the compound policy directly and Dragon picks one GPU from the affinity list.

    Returns an empty list when *gpu_ids* is empty or Dragon is not available.
    """
    if not gpu_ids or not gpu_pool:
        return []
    try:
        from dragon.infrastructure.policy import Policy

        hostname = gpu_pool[0][0]  # single-node: all GPUs share the same host
        return [
            Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=hostname,
                gpu_affinity=list(gpu_ids),
            )
        ]
    except Exception:
        return []


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
        _cm: Optional[object] = None,
        _group_name: Optional[str] = None,
        asyncflow: Optional[object] = None,
        policies: Optional[list] = None,
        engine_dragon: Optional[object] = None,
    ) -> None:
        self.config = config
        # AsyncCampaignManager reference injected at construction.
        self._cm = _cm
        self._group_name = _group_name
        # Shared WorkflowEngine injected by AsyncCampaignManager (optional).
        self.asyncflow = asyncflow
        # One Dragon Policy per assigned GPU, injected by AsyncCampaignManager.
        # Single-GPU workflows use policies[0]; multi-GPU workflows cycle through the list.
        self.policies: list = policies or []
        # Dragon backend handle for routing tasks to specific backends.
        self.engine_dragon: Optional[object] = engine_dragon

    async def _trigger_dependent(
        self,
        name: str,
        replicas: int = 1,
        **kwargs,
    ) -> None:
        """
        Tell the CM to activate a dependent workflow group with *replicas* replicas.

        The group must already be registered (via config or register_group) with
        ``replicas=0``.  Calling this mid-run is the canonical way to start a
        workflow that depends on data produced by the current workflow.
        No-op when no CM was injected.
        """
        if self._cm is not None:
            await self._cm.trigger_dependent(name, replicas=replicas, **kwargs)

    async def _signal_done(self) -> None:
        """
        Signal the CM that this workflow has finished producing data for
        its dependents (count-based dependency fallback).

        Marks this group ``ready=True`` in the CM, which unblocks any groups
        that list this one as a dependency with ``dep_threshold > finished_replicas``.
        No-op when no CM was injected.
        """
        if self._cm is not None and self._group_name is not None:
            await self._cm.signal_done(self._group_name)

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
        cm: "AsyncCampaignManager",
        final_state: str,
    ) -> None:
        """
        Hook called after this replica's entry-point finishes.

        Override to add workflow-specific post-replica logic.  The default
        implementation does nothing.
        """


# ---------------------------------------------------------------------------
# Internal group descriptor
# ---------------------------------------------------------------------------


@dataclass
class _GroupInfo:
    name: str
    workflow_class: type[BaseWorkflow]
    replicas: int
    dependencies: list[str]
    group_config: Optional[dict]
    configured_replicas: int = 0
    priority: int = 0
    min_replicas: int = 0
    max_replicas: int = 0
    required_cpus: int = 0
    required_gpus: int = 0
    dep_threshold: int = 1
    entry_point: str = "run"
    status: str = "pending"
    started_count: int = 0
    running_count: int = 0
    finished_replicas: int = 0
    # Set to True when the workflow explicitly signals it has produced enough
    # data (via cm.signal_ready).  Takes precedence over dep_threshold check.
    ready: bool = False
    # GPU IDs currently held by all running replicas of this group.
    # Populated by _allocate_locked; cleared by _on_replica_finished.
    # Injected into each replica's config so shared services (e.g. inference)
    # can initialise on all group-level GPUs rather than just the first one.
    running_gpu_ids: list[int] = field(default_factory=list)


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
# Public stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStats:
    """Cumulative statistics for one workflow group."""

    replicas_started: int = 0
    replicas_finished: int = 0


# ---------------------------------------------------------------------------
# CampaignManager
# ---------------------------------------------------------------------------


class AsyncCampaignManager:
    """
    Async campaign manager that orchestrates multiple replicas of one
    or more :class:`~src.campaign.campaign_manager.BaseWorkflow` subclasses.

    Parameters
    ----------
    max_workers : int, optional
        Unused (kept for API parity with the sync CM).
    engine : str
        Execution backend — ``"dragon"`` (default) or ``"concurrent"``.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        engine: str = "concurrent",
        total_cpus: int = 0,
        total_gpus: int = 0,
        num_workers: Optional[int] = None,
        debug: bool = False,
        asyncflow=None,
        engine_dragon=None,
    ) -> None:
        self._log = Logger(name="AsyncCampaignManager", use_colors=True)
        self._seq = itertools.count()
        self._lock = asyncio.Lock()
        self._engine_type = engine
        self._num_workers = num_workers
        self._debug = debug
        self._asyncflow = asyncflow  # may be pre-built by caller
        self._engine_dragon = engine_dragon  # DragonExecutionBackendV3
        self._gpu_pool: list[tuple[str, int]] = []  # populated after Dragon init
        self._free_gpu_ids: list[int] = []  # available GPU IDs
        self._replica_gpu_assignments: dict[str, list[int]] = {}  # replica_id → [gpu_ids]
        self._resources = ResourcePool(total_cpus=total_cpus, total_gpus=total_gpus)

        self._groups: dict[str, _GroupInfo] = {}
        self._stats: dict[str, WorkflowStats] = {}
        self._all_done = asyncio.Event()

        self._log.info(f"AsyncCampaignManager initialised (engine={engine})")

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        workflow_registry: dict[str, type[BaseWorkflow]],
        asyncflow=None,
        engine_dragon=None,
    ) -> "AsyncCampaignManager":
        """
        Build an AsyncCampaignManager from a config dict + workflow registry.

        Pass *asyncflow* and *engine_dragon* when the caller has already
        created the backend (recommended — lets the caller manage telemetry
        and shutdown ordering).
        """
        res_cfg = config.get("resources", {})
        num_workers = config.get("num_workers")
        cm = cls(
            max_workers=config.get("max_workers"),
            engine=config.get("engine", "concurrent"),
            total_cpus=int(res_cfg.get("total_cpus", 0)),
            total_gpus=int(res_cfg.get("total_gpus", 0)),
            num_workers=int(num_workers) if num_workers is not None else None,
            debug=bool(config.get("debug", False)),
            asyncflow=asyncflow,
            engine_dragon=engine_dragon,
        )

        _cm_keys = {
            "replicas",
            "dependencies",
            "dependency_threshold",
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

            # Groups with dependencies default to replicas=0 — they remain
            # inactive until a parent workflow calls _trigger_dependent().
            has_deps = bool(wf_cfg.get("dependencies", []))
            default_replicas = 0 if has_deps else 1
            cm.register_group(
                name=name,
                workflow_class=wf_class,
                replicas=int(wf_cfg.get("replicas", default_replicas)),
                dependencies=list(wf_cfg.get("dependencies", [])),
                dep_threshold=int(wf_cfg.get("dependency_threshold", 1)),
                priority=int(wf_cfg.get("priority", 0)),
                min_replicas=int(wf_cfg.get("min_replicas", 0)),
                max_replicas=int(wf_cfg.get("max_replicas", 0)),
                required_cpus=int(wf_cfg.get("required_cpus", 0)),
                required_gpus=int(wf_cfg.get("required_gpus", 0)),
                config={k: v for k, v in wf_cfg.items() if k not in _cm_keys} or None,
            )

        return cm

    # ------------------------------------------------------------------
    # Group registration
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_entry_point(workflow_class: type[BaseWorkflow]) -> str:
        has_run = workflow_class.run is not BaseWorkflow.run
        has_start = "start" in workflow_class.__dict__ or (
            hasattr(workflow_class, "start")
            and workflow_class.start is not getattr(BaseWorkflow, "start", None)
        )

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
        workflow_class: type[BaseWorkflow],
        replicas: int = 1,
        dependencies: Optional[list[str]] = None,
        dep_threshold: int = 1,
        priority: int = 0,
        min_replicas: int = 0,
        max_replicas: int = 0,
        required_cpus: int = 0,
        required_gpus: int = 0,
        config: Optional[dict] = None,
    ) -> None:
        """Register a workflow group."""
        entry_point = self._resolve_entry_point(workflow_class)
        effective_max = max_replicas if max_replicas > 0 else replicas

        self._groups[name] = _GroupInfo(
            name=name,
            workflow_class=workflow_class,
            replicas=replicas,
            dependencies=list(dependencies or []),
            group_config=config,
            configured_replicas=replicas,
            priority=priority,
            min_replicas=min_replicas,
            max_replicas=effective_max,
            required_cpus=required_cpus,
            required_gpus=required_gpus,
            dep_threshold=dep_threshold,
            entry_point=entry_point,
        )
        self._stats[name] = WorkflowStats()

        self._log.info(
            f"Registered group {name!r}: replicas={replicas} "
            f"priority={priority} min={min_replicas} max={effective_max} "
            f"deps={dependencies or []} dep_threshold={dep_threshold} "
            f"resources=(cpus={required_cpus}, gpus={required_gpus})"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _init_asyncflow(self) -> None:
        """Create the shared WorkflowEngine (called once from start()).

        When *asyncflow* was provided at construction time the engine is
        already live — skip creation and only sync the GPU pool.
        """
        if self._debug:
            try:
                from rhapsody import enable_logging

                enable_logging(level="DEBUG")
                import logging as _logging

                _logging.getLogger("radical.asyncflow").setLevel(_logging.WARNING)
                _logging.getLogger("asyncio").setLevel(_logging.WARNING)
                self._log.warning("rhapsody.enable_logging")
            except ImportError:
                self._log.warning("rhapsody.enable_logging not available — skipping debug logging")

        if self._asyncflow is not None:
            # Engine was pre-built by the caller (e.g. run_campaing.py).
            # Discover GPUs and sync the ResourcePool so scheduling works.
            if self._engine_type == "dragon":
                self._gpu_pool = _find_gpus()
                self._free_gpu_ids = [gid for _, gid in self._gpu_pool]
                actual_gpus = len(self._free_gpu_ids)
                if actual_gpus != self._resources.total_gpus:
                    self._log.warning(
                        f"config total_gpus={self._resources.total_gpus} "
                        f"!= discovered GPUs={actual_gpus} — "
                        f"capping ResourcePool to {actual_gpus}"
                    )
                    self._resources.total_gpus = actual_gpus
                    self._resources.available_gpus = actual_gpus
                self._log.info(
                    f"GPU pool: {len(self._gpu_pool)} GPU(s) — "
                    + (
                        ", ".join(f"{h}:{g}" for h, g in self._gpu_pool)
                        if self._gpu_pool
                        else "none found"
                    )
                )
            else:
                # Concurrent mode: auto-detect CUDA GPUs.
                if not self._free_gpu_ids:
                    try:
                        import torch

                        n = torch.cuda.device_count()
                    except Exception:
                        try:
                            import subprocess

                            out = subprocess.check_output(
                                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                text=True,
                            )
                            n = len(out.strip().splitlines())
                        except Exception:
                            n = 0
                    n = min(n, self._resources.total_gpus)
                    self._free_gpu_ids = list(range(n))
                    self._log.info(f"Concurrent mode: auto-detected {n} GPU(s) for assignment")
            self._log.info(f"Using pre-built asyncflow engine (engine={self._engine_type})")
            return

        # ── Build engine from scratch ─────────────────────────────────────
        from radical.asyncflow import WorkflowEngine

        kw = {} if self._num_workers is None else {"num_workers": self._num_workers}

        if self._engine_type == "dragon":
            try:
                from rhapsody.backends import DragonExecutionBackendV3

                self._engine_dragon = await DragonExecutionBackendV3(**kw)
                self._gpu_pool = _find_gpus()
                self._free_gpu_ids = [gid for _, gid in self._gpu_pool]
                actual_gpus = len(self._free_gpu_ids)
                if actual_gpus != self._resources.total_gpus:
                    self._log.warning(
                        f"config total_gpus={self._resources.total_gpus} "
                        f"!= discovered GPUs={actual_gpus} — "
                        f"capping ResourcePool to {actual_gpus}"
                    )
                    self._resources.total_gpus = actual_gpus
                    self._resources.available_gpus = actual_gpus
                self._log.info(
                    f"GPU pool: {len(self._gpu_pool)} GPU(s) — "
                    + (
                        ", ".join(f"{h}:{g}" for h, g in self._gpu_pool)
                        if self._gpu_pool
                        else "none found"
                    )
                )
                backend = self._engine_dragon
            except ImportError:
                self._log.warning(
                    "Dragon backend not available — falling back to ConcurrentExecutionBackend"
                )
                self._engine_type = "concurrent"

        if self._engine_type == "concurrent":
            from rhapsody.backends import ConcurrentExecutionBackend

            backend = await ConcurrentExecutionBackend()
            if not self._free_gpu_ids:
                try:
                    import torch

                    n = torch.cuda.device_count()
                except Exception:
                    try:
                        import subprocess

                        out = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
                        )
                        n = len(out.strip().splitlines())
                    except Exception:
                        n = 0
                n = min(n, self._resources.total_gpus)
                self._free_gpu_ids = list(range(n))
                self._log.info(f"Concurrent mode: auto-detected {n} GPU(s) for assignment")

        self._asyncflow = await WorkflowEngine.create(backend=backend)
        self._log.info(f"Shared asyncflow engine created (engine={self._engine_type})")

    async def start(self) -> None:
        """Kick off the campaign — schedule all eligible groups."""

        if not self._groups:
            self._all_done.set()
            return
        await self._init_asyncflow()
        res = self._resources
        if res.total_cpus > 0 or res.total_gpus > 0:
            self._log.info(
                f"Resource pool: total_cpus={res.total_cpus}  total_gpus={res.total_gpus}"
            )
        await self._schedule()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Block (async) until all workflow groups have finished."""
        if timeout is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._all_done.wait()), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False
        await self._all_done.wait()
        return True

    async def close(self) -> None:
        """Release CM resources.

        Asyncflow shutdown is intentionally skipped here — when the engine
        was created externally (run_campaing.py) the caller is responsible
        for calling ``asyncflow.shutdown()`` after telemetry has been stopped.
        """
        self._asyncflow = None
        self._log.info("AsyncCampaignManager closed")

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def signal_done(self, group_name: str) -> None:
        """
        Signal that *group_name* has produced output and downstream work should run.

        Called by a workflow via ``self._signal_done()`` — can be called multiple
        times per replica (e.g. once per iteration).  Each call queues **1 more
        replica** in every group that lists *group_name* in its ``dependencies``.
        The CM routes the signal to the right downstream groups automatically, so
        the calling workflow does not need to know their names.

        Also sets ``group.ready = True`` to satisfy count-based dep checks.
        """
        async with self._lock:
            group = self._groups.get(group_name)
            if group is None:
                return
            group.ready = True
            dependents = [g for g in self._groups.values() if group_name in g.dependencies]
            for dep in dependents:
                dep.replicas += 1
                dep.configured_replicas += 1
                if dep.status == "done":
                    dep.status = "pending"
            if dependents:
                self._log.info(
                    f"{group_name!r} signaled done → +1 replica for {[d.name for d in dependents]}"
                )
        await self._schedule()

    async def trigger_dependent(
        self,
        name: str,
        replicas: int = 1,
        config: Optional[dict] = None,
    ) -> None:
        """
        Queue *replicas* more runs of the dependent group *name*.

        Called by a parent workflow (via ``self._trigger_dependent()``) each
        time its execution logic decides to launch downstream work.  May be
        called multiple times — each call adds *replicas* to the group's
        total and re-opens the group for scheduling if it had already finished.

        The group must be pre-registered (via config or ``register_group``).
        """
        async with self._lock:
            group = self._groups.get(name)
            if group is None:
                self._log.warning(f"trigger_dependent: group {name!r} not registered — ignoring")
                return
            group.replicas += replicas
            group.configured_replicas += replicas
            if group.status == "done":
                group.status = "pending"
            if config:
                group.group_config = {**(group.group_config or {}), **config}
            self._log.info(
                f"trigger_dependent: {name!r} +{replicas} replicas (total={group.replicas})"
            )
        await self._schedule()

    async def add_replicas(self, group_name: str, n: int = 1) -> None:
        """
        Dynamically add *n* more replicas to an existing workflow group.

        If the group was already marked "done" it is re-opened to "running"
        so the scheduler will consider it again.
        """
        async with self._lock:
            group = self._groups.get(group_name)
            if group is None:
                self._log.warning(f"add_replicas: group {group_name!r} not found — ignoring")
                return
            cap = group.configured_replicas
            if group.replicas >= cap:
                self._log.warning(
                    f"add_replicas: {group_name!r} already at configured cap "
                    f"({cap}) — ignoring request for +{n}"
                )
                return
            allowed = min(n, cap - group.replicas)
            group.replicas += allowed
            if group.status == "done":
                group.status = "running"
            self._log.info(
                f"add_replicas: {group_name!r} +{allowed} → "
                f"total={group.replicas}/{cap} started={group.started_count}"
            )
        await self._schedule()

    def status(self) -> dict:
        """Snapshot of campaign progress."""
        return {
            "resources": self._resources.as_dict(),
            "groups": {
                name: {
                    "status": g.status,
                    "priority": g.priority,
                    "replicas_total": g.replicas,
                    "replicas_configured": g.configured_replicas,
                    "replicas_started": g.started_count,
                    "replicas_running": g.running_count,
                    "replicas_finished": g.finished_replicas,
                    "min_replicas": g.min_replicas,
                    "max_replicas": g.max_replicas,
                    "required_cpus": g.required_cpus,
                    "required_gpus": g.required_gpus,
                    "dep_threshold": g.dep_threshold,
                    "ready": g.ready,
                    "dependencies": g.dependencies,
                }
                for name, g in self._groups.items()
            },
        }

    def stats(self) -> dict[str, WorkflowStats]:
        """Per-workflow-group statistics."""
        return {name: WorkflowStats(**vars(s)) for name, s in self._stats.items()}

    # ------------------------------------------------------------------
    # Internal — scheduler
    # ------------------------------------------------------------------

    def _deps_satisfied_locked(self, group: _GroupInfo) -> bool:
        """
        True when every dependency group is considered "ready".

        A dependency group is ready when **either**:
        - it has explicitly called ``signal_ready()`` (workflow-driven signal), OR
        - it has ``dep_threshold`` or more finished replicas (count-based fallback).
        """
        for dep_name in group.dependencies:
            dep = self._groups.get(dep_name)
            if dep is None:
                return False
            if not dep.ready and dep.finished_replicas < group.dep_threshold:
                return False
        return True

    def _can_start_locked(self, group: _GroupInfo) -> bool:
        """True if one more replica of *group* can be started right now."""
        if group.status == "done":
            return False
        if group.started_count >= group.replicas:
            return False
        # max_replicas == 0 means "no explicit cap — use replicas count".
        # This handles dependent groups registered with replicas=0 that later
        # receive replicas via trigger_dependent() or signal_done().
        effective_max = group.max_replicas if group.max_replicas > 0 else group.replicas
        if group.running_count >= effective_max:
            return False
        if not self._deps_satisfied_locked(group):
            return False
        if not self._resources.can_fit(group.required_cpus, group.required_gpus):
            return False
        return True

    def _allocate_locked(self, group: _GroupInfo) -> int:
        """Record one replica start for *group* (update counters, return idx)."""
        idx = group.started_count
        group.started_count += 1
        group.running_count += 1
        self._resources.allocate(group.required_cpus, group.required_gpus)
        self._stats[group.name].replicas_started = group.started_count
        # Pop specific GPU IDs from the global free pool (FIFO).
        replica_id = f"{group.name}_{idx}"
        gpu_ids = [
            self._free_gpu_ids.pop(0) for _ in range(group.required_gpus) if self._free_gpu_ids
        ]
        self._replica_gpu_assignments[replica_id] = gpu_ids
        group.running_gpu_ids.extend(gpu_ids)
        if gpu_ids:
            self._log.info(
                f"  GPU assign: {replica_id!r} → GPU(s) {gpu_ids}"
                f"  | free: {sorted(self._free_gpu_ids)}"
            )
        return idx

    def _schedule_locked(self) -> list[tuple["_GroupInfo", int]]:
        """
        Two-pass greedy scheduler.  Must be called under ``self._lock``.

        Pass 1 — guarantee ``min_replicas`` for all eligible groups,
                  highest priority first.
        Pass 2 — fill remaining capacity up to ``max_replicas``,
                  highest priority first.

        Returns a list of (group, replica_idx) pairs to start.
        """
        to_start: list[tuple[_GroupInfo, int]] = []

        eligible = [
            g
            for g in self._groups.values()
            if g.status != "done"
            and g.started_count < g.replicas
            and self._deps_satisfied_locked(g)
        ]

        for g in eligible:
            if g.status == "pending":
                g.status = "running"
                self._log.info(f"Group {g.name!r} is now eligible — status → running")

        eligible.sort(key=lambda g: g.priority, reverse=True)

        # Pass 1: guarantee min_replicas.
        for g in eligible:
            deficit = g.min_replicas - g.running_count
            for _ in range(deficit):
                if not self._can_start_locked(g):
                    break
                idx = self._allocate_locked(g)
                to_start.append((g, idx))

        # Pass 2: fill remaining capacity up to max_replicas.
        for g in eligible:
            while self._can_start_locked(g):
                idx = self._allocate_locked(g)
                to_start.append((g, idx))

        # Warn about groups that are eligible (deps satisfied, slots available)
        # but stalled on resources.
        for g in eligible:
            if (
                g.started_count < g.replicas
                and g.running_count < (g.max_replicas if g.max_replicas > 0 else g.replicas)
                and self._deps_satisfied_locked(g)
                and not self._resources.can_fit(g.required_cpus, g.required_gpus)
            ):
                self._log.warning(
                    f"Group {g.name!r} stalled — waiting for resources "
                    f"(needs cpus={g.required_cpus} gpus={g.required_gpus}  "
                    f"available: {self._resources.available_str()})"
                )

        if to_start:

            def _gpu_tag(g, idx):
                ids = self._replica_gpu_assignments.get(f"{g.name}_{idx}", [])
                return f"gpu={ids}" if ids else ""

            summary = ", ".join(
                f"{g.name}_{idx}" + (f"[{_gpu_tag(g, idx)}]" if _gpu_tag(g, idx) else "")
                for g, idx in to_start
            )
            replica_status = "  ".join(
                f"{g.name}: {g.running_count} running/ {g.finished_replicas} done/ {g.replicas} total"
                for g in self._groups.values()
            )
            _used: set[str] = set()
            _abbrevs: dict[str, str] = {}
            for g in self._groups.values():
                ch = next(
                    (c.upper() for c in g.name if c.upper() not in _used),
                    chr(ord("A") + len(_abbrevs)),
                )
                _abbrevs[g.name] = ch
                _used.add(ch)
            viz = "".join(_abbrevs[g.name] * g.running_count for g in self._groups.values())
            self._log.info(
                f"Scheduling: [{summary}] | [{viz}] replicas: {replica_status}"
                f" | resources: {self._resources.usage_str()}"
            )

        return to_start

    async def _schedule(self) -> None:
        """Run the scheduler and launch new replicas."""
        async with self._lock:
            to_start = self._schedule_locked()

        for group, replica_idx in to_start:
            asyncio.get_running_loop().create_task(self._run_replica(group, replica_idx))

    # ------------------------------------------------------------------
    # Internal — replica execution
    # ------------------------------------------------------------------

    async def _run_replica(self, group: _GroupInfo, replica_idx: int) -> None:
        """Execute one replica of a workflow group."""
        replica_id = f"{group.name}_{replica_idx}"
        final_state = "done"

        gpu_ids = self._replica_gpu_assignments.get(replica_id, [])
        policies = _make_policies(self._gpu_pool, gpu_ids)

        res_tag = ""
        if group.required_cpus > 0 or group.required_gpus > 0:
            res_tag = f" [cpus={group.required_cpus} gpus={group.required_gpus}]"
        if gpu_ids:
            host = self._gpu_pool[0][0] if self._gpu_pool else "?"
            res_tag += f" [gpu_affinity={gpu_ids} host={host}]"
        self._log.info(f"  starting replica {replica_id!r} (priority={group.priority}){res_tag}")

        # Inject GPU IDs into config so workflows that initialize services
        # in-process (e.g. inference) can select the correct CUDA devices
        # rather than always defaulting to cuda:0.
        #   assigned_gpu_ids  — this replica's own GPU(s)
        #   group_gpu_ids     — all GPUs held by the group right now
        # The first replica to call _ensure_initialized uses group_gpu_ids to
        # set up N services on N distinct GPUs (e.g. num_services=2).
        replica_config = group.group_config
        if gpu_ids:
            replica_config = {
                **group.group_config,
                "assigned_gpu_ids": gpu_ids,
                "group_gpu_ids": list(group.running_gpu_ids),
            }
        wf = group.workflow_class(
            config=replica_config,
            _cm=self,
            _group_name=group.name,
            asyncflow=self._asyncflow,
            policies=policies,
            engine_dragon=self._engine_dragon,
        )

        entry = getattr(wf, group.entry_point)
        try:
            if asyncio.iscoroutinefunction(entry):
                await entry(replica_id)
            else:
                await asyncio.to_thread(entry, replica_id)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            # Catch BaseException (not just Exception) so that unusual raises
            # like StopAsyncIteration or KeyboardInterrupt still reach
            # _handle_replica_done — otherwise the campaign hangs forever.
            self._log.error(f"Replica {replica_id!r} raised: {type(exc).__name__}: {exc}")
            final_state = "failed"

        await self._handle_replica_done(wf, group, replica_id, replica_idx, final_state)

    async def _handle_replica_done(
        self,
        wf: BaseWorkflow,
        group: _GroupInfo,
        replica_id: str,
        replica_idx: int,
        final_state: str,
    ) -> None:
        """Call workflow hook, then update group state and re-schedule."""
        try:
            hook = wf.on_replica_done
            if asyncio.iscoroutinefunction(hook):
                await hook(replica_id, self, final_state)
            else:
                hook(replica_id, self, final_state)
        except Exception as exc:
            self._log.error(f"Replica {replica_id!r} on_replica_done raised: {exc}")

        await self._on_replica_finished(group, replica_id)

    async def _on_replica_finished(self, group: _GroupInfo, replica_id: str) -> None:
        """Update group counters and re-run scheduler."""
        group_done = False
        async with self._lock:
            group.finished_replicas += 1
            group.running_count -= 1
            self._resources.release(group.required_cpus, group.required_gpus)
            self._stats[group.name].replicas_finished = group.finished_replicas

            if group.finished_replicas >= group.replicas:
                group.status = "done"
                group_done = True

        freed_gpu_ids = self._replica_gpu_assignments.pop(replica_id, [])
        self._free_gpu_ids.extend(freed_gpu_ids)
        for gid in freed_gpu_ids:
            try:
                group.running_gpu_ids.remove(gid)
            except ValueError:
                pass

        if freed_gpu_ids:
            if self._replica_gpu_assignments:
                asgn_str = ", ".join(
                    f"{rid}→{gids}" for rid, gids in sorted(self._replica_gpu_assignments.items())
                )
                self._log.info(
                    f"  GPU freed: {replica_id!r} released {freed_gpu_ids}"
                    f"  | active: [{asgn_str}]"
                    f"  | free: {sorted(self._free_gpu_ids)}"
                )
            else:
                self._log.info(
                    f"  GPU freed: {replica_id!r} released {freed_gpu_ids}"
                    f"  | active: (none)"
                    f"  | free: {sorted(self._free_gpu_ids)}"
                )

        release_tag = ""
        if group.required_cpus > 0 or group.required_gpus > 0:
            release_tag = (
                f" | released cpus={group.required_cpus} gpus={group.required_gpus}"
                f" | available: {self._resources.available_str()}"
            )
        if freed_gpu_ids:
            release_tag += (
                f" [freed gpu_affinity={freed_gpu_ids} | free_gpus={sorted(self._free_gpu_ids)}]"
            )
        self._log.info(f"Replica {replica_id!r} finished{release_tag}")

        if group_done:
            self._log.info(f"Workflow group {group.name!r} completed")

        self._log.info(
            f"_on_replica_finished: {group.name!r} - "
            f"finished_replicas={group.finished_replicas}/{group.replicas}"
        )

        await self._schedule()

        async with self._lock:
            # Groups registered with replicas=0 (triggered dependents not yet activated)
            # are excluded from the completion check — they only count once triggered.
            all_done = (
                bool(self._groups)
                and all(g.status == "done" or g.replicas == 0 for g in self._groups.values())
                and any(g.replicas > 0 for g in self._groups.values())
            )

        if all_done:
            self._all_done.set()
            self._log.info("All campaign workflow groups finished")


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------


class CampaignManager:
    """
    Synchronous campaign manager — thin wrapper around AsyncCampaignManager.

    Runs a dedicated event loop in a background thread so that callers without
    an async context can orchestrate workflows with plain blocking calls.
    Sync workflow entry points (``def run()``) are executed in a thread pool
    via ``asyncio.to_thread`` inside the background loop.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        engine: str = "concurrent",
        total_cpus: int = 0,
        total_gpus: int = 0,
        num_workers: Optional[int] = None,
        debug: bool = False,
        asyncflow=None,
        engine_dragon=None,
    ) -> None:
        import threading

        self._acm = AsyncCampaignManager(
            max_workers=max_workers,
            engine=engine,
            total_cpus=total_cpus,
            total_gpus=total_gpus,
            num_workers=num_workers,
            debug=debug,
            asyncflow=asyncflow,
            engine_dragon=engine_dragon,
        )

        # Skip heavy backend init — sync CM runs workflows directly in threads.
        async def _noop_init() -> None:
            pass

        self._acm._init_asyncflow = _noop_init

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="CampaignManagerLoop"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (sync mirrors of AsyncCampaignManager)
    # ------------------------------------------------------------------

    def register_group(self, *args, **kwargs) -> None:
        self._acm.register_group(*args, **kwargs)

    def start(self) -> None:
        future = asyncio.run_coroutine_threadsafe(self._acm.start(), self._loop)
        future.result()

    def wait(self, timeout: Optional[float] = None) -> bool:
        future = asyncio.run_coroutine_threadsafe(self._acm.wait(timeout=timeout), self._loop)
        outer_timeout = (timeout + 2.0) if timeout is not None else None
        try:
            return bool(future.result(timeout=outer_timeout))
        except Exception:
            return False

    def close(self) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(self._acm.close(), self._loop)
            future.result(timeout=5.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def status(self) -> dict:
        return self._acm.status()

    def stats(self) -> dict[str, WorkflowStats]:
        return self._acm.stats()

    @classmethod
    def from_config(
        cls,
        config: dict,
        workflow_registry: dict[str, type[BaseWorkflow]],
        **kwargs,
    ) -> "CampaignManager":
        res_cfg = config.get("resources", {})
        num_workers = config.get("num_workers")
        cm = cls(
            max_workers=config.get("max_workers"),
            engine=config.get("engine", "concurrent"),
            total_cpus=int(res_cfg.get("total_cpus", 0)),
            total_gpus=int(res_cfg.get("total_gpus", 0)),
            num_workers=int(num_workers) if num_workers is not None else None,
            debug=bool(config.get("debug", False)),
            **kwargs,
        )

        _cm_keys = {
            "replicas",
            "dependencies",
            "dependency_threshold",
            "priority",
            "min_replicas",
            "max_replicas",
            "required_cpus",
            "required_gpus",
        }

        for name, wf_cfg in config.get("workflows", {}).items():
            wf_class = workflow_registry.get(name)
            if wf_class is None:
                continue
            has_deps = bool(wf_cfg.get("dependencies", []))
            default_replicas = 0 if has_deps else 1
            cm.register_group(
                name=name,
                workflow_class=wf_class,
                replicas=int(wf_cfg.get("replicas", default_replicas)),
                dependencies=list(wf_cfg.get("dependencies", [])),
                dep_threshold=int(wf_cfg.get("dependency_threshold", 1)),
                priority=int(wf_cfg.get("priority", 0)),
                min_replicas=int(wf_cfg.get("min_replicas", 0)),
                max_replicas=int(wf_cfg.get("max_replicas", 0)),
                required_cpus=int(wf_cfg.get("required_cpus", 0)),
                required_gpus=int(wf_cfg.get("required_gpus", 0)),
                config={k: v for k, v in wf_cfg.items() if k not in _cm_keys} or None,
            )

        return cm
