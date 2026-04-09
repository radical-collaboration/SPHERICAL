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

A group becomes *eligible* when each dependency group has either explicitly
called ``signal_ready()`` (workflow-driven) or has ``dep_threshold`` or more
finished replicas (count-based fallback, default 1).

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
from typing import Dict, List, Optional, Tuple, Type

from .campaign_manager import BaseWorkflow, ResourcePool, WorkflowStats
from ..utils.logger import Logger


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _find_gpus() -> List[Tuple[str, int]]:
    """Return [(hostname, gpu_id), ...] for every GPU visible to Dragon.

    Falls back to an empty list when Dragon is not active (e.g. concurrent
    backend during local testing).
    """
    try:
        from dragon.native.machine import System, Node

        gpus = []
        for huid in System().nodes:
            node = Node(huid)
            for gpu_id in node.gpus:
                gpus.append((node.hostname, gpu_id))
        return gpus
    except Exception:
        return []


def _make_policies(gpu_pool: List[Tuple[str, int]], gpu_ids: List[int]) -> List:
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
# Internal group descriptor
# ---------------------------------------------------------------------------


@dataclass
class _GroupInfo:
    name: str
    workflow_class: Type[BaseWorkflow]
    replicas: int
    dependencies: List[str]
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
    running_gpu_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AsyncCampaignManager
# ---------------------------------------------------------------------------


class AsyncCampaignManager:
    """
    Async-native campaign manager that orchestrates multiple replicas of one
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
    ) -> None:
        self._log = Logger(name="AsyncCampaignManager", use_colors=True)
        self._seq = itertools.count()
        self._lock = asyncio.Lock()
        self._engine_type = engine
        self._num_workers = num_workers
        self._debug = debug
        self._asyncflow = None  # created in start()
        self._engine_dragon = None  # DragonExecutionBackendV3
        self._gpu_pool: List[Tuple[str, int]] = []  # populated after Dragon init
        self._free_gpu_ids: List[int] = []  # available GPU IDs
        self._replica_gpu_assignments: Dict[str, List[int]] = {}  # replica_id → [gpu_ids]
        self._resources = ResourcePool(total_cpus=total_cpus, total_gpus=total_gpus)

        self._groups: Dict[str, _GroupInfo] = {}
        self._stats: Dict[str, WorkflowStats] = {}
        self._all_done = asyncio.Event()

        self._log.info(f"AsyncCampaignManager initialised (engine={engine})")

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        workflow_registry: Dict[str, Type[BaseWorkflow]],
    ) -> "AsyncCampaignManager":
        """
        Build an AsyncCampaignManager from a config dict + workflow registry.
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

            cm.register_group(
                name=name,
                workflow_class=wf_class,
                replicas=int(wf_cfg.get("replicas", 1)),
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
    def _resolve_entry_point(workflow_class: Type[BaseWorkflow]) -> str:
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
        workflow_class: Type[BaseWorkflow],
        replicas: int = 1,
        dependencies: Optional[List[str]] = None,
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
        """Create the shared WorkflowEngine (called once from start())."""
        from radical.asyncflow import WorkflowEngine

        if self._debug:
            try:
                from rhapsody import enable_logging

                enable_logging(level="DEBUG")
                self._log.warning("rhapsody.enable_logging")
            except ImportError:
                self._log.warning("rhapsody.enable_logging not available — skipping debug logging")

        kw = {} if self._num_workers is None else {"num_workers": self._num_workers}

        if self._engine_type == "dragon":
            try:
                from rhapsody.backends import DragonExecutionBackendV3

                self._engine_dragon = await DragonExecutionBackendV3(**kw)
                self._gpu_pool = _find_gpus()
                self._free_gpu_ids = [gid for _, gid in self._gpu_pool]
                # Sync ResourcePool to actual GPU count so can_fit() and
                # _free_gpu_ids stay consistent.  config total_gpus acts as
                # an upper bound; actual discovery takes precedence.
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
            # In concurrent mode Dragon is absent, so _free_gpu_ids is empty.
            # Auto-detect CUDA GPUs and populate it so assigned_gpu_ids is
            # injected into replica configs (needed for CUDA_VISIBLE_DEVICES).
            if not self._free_gpu_ids:
                try:
                    import torch

                    n = torch.cuda.device_count()
                except Exception:
                    try:
                        import subprocess, re

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
        """Shut down the shared asyncflow engine."""
        if self._asyncflow is not None:
            try:
                await self._asyncflow.shutdown()
            except Exception as exc:
                self._log.warning(f"asyncflow shutdown raised: {exc}")
            self._asyncflow = None
        self._log.info("AsyncCampaignManager closed")

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def signal_ready(self, group_name: str) -> None:
        """
        Mark *group_name* as having produced enough data for its dependents.

        Called by a workflow via ``self._signal_ready()`` (injected at construction).
        Once set, the flag overrides the ``dep_threshold`` replica-count check so
        dependent groups are unblocked immediately.

        Idempotent — subsequent calls for the same group are no-ops.
        """
        async with self._lock:
            group = self._groups.get(group_name)
            if group is None or group.ready:
                return
            group.ready = True
            self._log.info(f"Group {group_name!r} signaled ready — unblocking dependents")
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

    def stats(self) -> Dict[str, WorkflowStats]:
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
        if group.running_count >= group.max_replicas:
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

    def _schedule_locked(self) -> List[Tuple["_GroupInfo", int]]:
        """
        Two-pass greedy scheduler.  Must be called under ``self._lock``.

        Pass 1 — guarantee ``min_replicas`` for all eligible groups,
                  highest priority first.
        Pass 2 — fill remaining capacity up to ``max_replicas``,
                  highest priority first.

        Returns a list of (group, replica_idx) pairs to start.
        """
        to_start: List[Tuple[_GroupInfo, int]] = []

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
                and g.running_count < g.max_replicas
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
            on_ready=lambda: self.signal_ready(group.name),
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
            all_done = bool(self._groups) and all(g.status == "done" for g in self._groups.values())

        if all_done:
            self._all_done.set()
            self._log.info("All campaign workflow groups finished")
