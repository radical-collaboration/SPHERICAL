"""
AsyncCampaignManager — async-native campaign orchestrator.

Each workflow replica is an asyncio Task.  Supports both
``async def run(replica_id)`` and sync ``def run(replica_id)`` entry points
(sync ones run via ``asyncio.to_thread``).

Scheduling model
----------------
Two-pass greedy scheduler on every state change (see scheduler.py):
  Pass 1 — guarantee ``min_replicas`` for all eligible groups (highest priority).
  Pass 2 — fill remaining capacity up to ``max_replicas`` (highest priority).

A group becomes eligible either via ``trigger_dependent()`` (explicit) or when
each dependency has ``dep_threshold`` finished replicas (count-based fallback).

Usage
-----
    cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)
    await cm.start()
    await cm.wait()
    await cm.close()
"""

import asyncio
import itertools
from typing import Optional

from ..utils.logger import Logger
from .backpressure import BackpressureNegotiator, BPState  # noqa: F401 (re-exported)
from .metrics import CampaignMetrics
from .bandit import Bandit, BanditArm, shard_bandit, resource_bandit, SchedulingBandit, scheduling_bandit  # noqa: F401
from .base_workflow import BaseWorkflow
from .candidate_log import CandidateLog, CandidateHistory, StageResult  # noqa: F401
from .executor import ExecutorMixin
from .monitor import Monitor, DriftKind  # noqa: F401 (re-exported)
from .monitor_mixin import MonitorMixin
from .scheduler import SchedulerMixin
from .sharder import Sharder, ShardingSpec
from .types import _GroupInfo, ResourcePool, WorkflowStats


class AsyncCampaignManager(SchedulerMixin, ExecutorMixin, MonitorMixin):
    """
    Async campaign manager that orchestrates multiple replicas of one
    or more :class:`BaseWorkflow` subclasses.
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
        features: Optional[dict] = None,
    ) -> None:
        self._log = Logger(name="AsyncCampaignManager", use_colors=True)
        self._seq = itertools.count()
        self._lock = asyncio.Lock()
        self._engine_type = engine
        self._num_workers = num_workers
        self._debug = debug
        self._asyncflow = asyncflow
        self._engine_dragon = engine_dragon
        self._gpu_pool: list[tuple[str, int]] = []
        self._free_gpu_ids: list[int] = []
        self._replica_gpu_assignments: dict[str, list[int]] = {}
        self._resources = ResourcePool(total_cpus=total_cpus, total_gpus=total_gpus)

        self._groups: dict[str, _GroupInfo] = {}
        self._stats: dict[str, WorkflowStats] = {}
        self._all_done = asyncio.Event()

        self._features: dict[str, bool] = features or {}
        self._bp: dict[str, BackpressureNegotiator] = {}
        self._sharders: dict[str, Sharder] = {}
        self._monitor: Optional[Monitor] = None
        self._monitor_interval_s: float = 30.0   # overwritten by from_config
        self._monitor_task: Optional[asyncio.Task] = None
        self._scheduling_bandit: Optional[SchedulingBandit] = None
        self._candidate_log: Optional[CandidateLog] = None
        self._cand_seq: itertools.count = itertools.count()
        self._replica_candidate_assignments: dict[str, str] = {}
        self._metrics: CampaignMetrics = CampaignMetrics()

        feat_summary = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in self._features.items())
        self._log.info(
            f"AsyncCampaignManager initialised (engine={engine})"
            + (f"  features: [{feat_summary}]" if feat_summary else "")
        )

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
        """Build an AsyncCampaignManager from a config dict + workflow registry."""
        res_cfg     = config.get("resources", {})
        num_workers = config.get("num_workers")
        features    = config.get("features", {})

        cm = cls(
            max_workers=config.get("max_workers"),
            engine=config.get("engine", "concurrent"),
            total_cpus=int(res_cfg.get("total_cpus", 0)),
            total_gpus=int(res_cfg.get("total_gpus", 0)),
            num_workers=int(num_workers) if num_workers is not None else None,
            debug=bool(config.get("debug", False)),
            asyncflow=asyncflow,
            engine_dragon=engine_dragon,
            features=dict(features) if features else {},
        )

        _cm_keys = {
            "replicas", "dependencies", "dependency_threshold",
            "min_replicas", "max_replicas", "priority", "required_cpus", "required_gpus",
            "concurrency_cap",
            "sharding",
        }

        for name, wf_cfg in config.get("workflows", {}).items():
            wf_class = workflow_registry.get(name)
            if wf_class is None:
                cm._log.warning(f"from_config: no class registered for {name!r} — skipping")
                continue

            has_deps = bool(wf_cfg.get("dependencies", []))
            default_replicas = 0 if has_deps else 1
            max_replicas = int(wf_cfg.get("max_replicas") or
                               wf_cfg.get("concurrency_cap") or 0)
            cm.register_group(
                name=name,
                workflow_class=wf_class,
                replicas=int(wf_cfg.get("replicas", default_replicas)),
                dependencies=list(wf_cfg.get("dependencies", [])),
                dep_threshold=int(wf_cfg.get("dependency_threshold", 1)),
                min_replicas=int(wf_cfg.get("min_replicas", 0)),
                max_replicas=max_replicas,
                priority=int(wf_cfg.get("priority", 0)),
                required_cpus=int(wf_cfg.get("required_cpus", 0)),
                required_gpus=int(wf_cfg.get("required_gpus", 0)),
                config={k: v for k, v in wf_cfg.items() if k not in _cm_keys} or None,
            )

        # ── Feature: Backpressure ─────────────────────────────────────────────
        if features.get("backpressure"):
            for name, wf_cfg in config.get("workflows", {}).items():
                hi = int(wf_cfg.get("backpressure_high") or 0)
                lo = int(wf_cfg.get("backpressure_low")  or 0)
                if hi > 0 and lo > 0 and hi > lo:
                    cm._bp[name] = BackpressureNegotiator(
                        edge_name=f"*_to_{name}", high_water=hi, low_water=lo,
                    )
                    cm._log.info(f"Backpressure [{name}]: high_water={hi}  low_water={lo}")

        # ── Feature: Sharder ─────────────────────────────────────────────────
        if features.get("sharder"):
            for name, wf_cfg in config.get("workflows", {}).items():
                sh_raw = wf_cfg.get("sharding")
                if sh_raw and isinstance(sh_raw, dict):
                    spec    = ShardingSpec.from_dict(sh_raw)
                    sharder = Sharder(name=name, spec=spec)
                    sharder._log_fn = cm._log.info
                    sharder._metrics_fn = lambda sid, n, sc, pr, _name=name: \
                        cm._metrics.record_shard(_name, sid, n, sc, pr)
                    cm._sharders[name] = sharder
                    cm._log.info(
                        f"Sharder [{name}]: target={spec.target_size} "
                        f"[{spec.min_size}, {spec.max_size}] stratify={spec.stratify}"
                    )

        # ── Candidate log (always enabled when any sharder exists) ───────────
        if any(wf_cfg.get("sharding") for wf_cfg in config.get("workflows", {}).values()):
            cm._candidate_log = CandidateLog()
            cm._log.info("CandidateLog enabled")

        # ── Feature: Monitor ──────────────────────────────────────────────────
        if features.get("monitor"):
            replan = config.get("replan", {})
            cm._monitor = Monitor(
                burn_dev_pct=float(replan.get("budget_burn_deviation_pct",    20.0)),
                passthrough_dev_pct=float(replan.get("pass_through_deviation_pct", 25.0)),
                recall_floor=float(replan.get("surrogate_recall_floor",       0.90)),
                breaches_to_escalate=2,
            )
            cm._monitor_interval_s = float(
                config.get("cm", {}).get("monitor_interval_s", 30.0)
            )
            cm._log.info(
                f"Monitor enabled: pass_through_dev={cm._monitor.passthrough_dev_pct}%  "
                f"budget_dev={cm._monitor.burn_dev_pct}%  "
                f"escalate_after={cm._monitor.breaches_to_escalate} breaches  "
                f"interval={cm._monitor_interval_s}s"
            )

        # ── Feature: Scheduling bandit ────────────────────────────────────────
        stage_names = list(config.get("workflows", {}).keys())
        if features.get("bandit") and len(stage_names) > 1:
            bandit_seed = config.get("bandit", {}).get("seeds", {}).get("bandit")
            # Warm-start: downstream stages get higher initial priority so the bandit
            # minimises time-to-target (first N terminal-stage completions).
            # Without this, the default FIFO order (insertion = upstream-first) runs
            # s1 at full capacity before feeding s4/s5, delaying the first leads.
            #
            # Priority by pipeline depth (deepest = highest priority):
            #   depth-0 (source)   → Beta(1, 1)  mean≈0.50  (lowest — runs last when competing)
            #   depth-1            → Beta(2, 1)  mean≈0.67
            #   depth-2            → Beta(3, 1)  mean≈0.75
            #   depth-3            → Beta(4, 1)  mean≈0.80
            #   depth-4+ (terminal)→ Beta(5, 1)  mean≈0.83  (highest — reaches target fastest)
            # Priors are weak (~1-5 effective observations) and quickly overridden by
            # the utilisation-based reward signal (see executor.py).
            def _dep_depth(name: str, visited: frozenset = frozenset()) -> int:
                if name in visited:
                    return 0
                deps = [d for d in cm._groups[name].dependencies if d in cm._groups]
                return 0 if not deps else 1 + max(
                    _dep_depth(d, visited | {name}) for d in deps
                )

            depths = {n: _dep_depth(n) for n in stage_names if n in cm._groups}
            max_alpha = 5  # terminal stage gets Beta(5,1) mean=0.83

            stage_priors: dict[str, tuple[float, float]] = {}
            for name in stage_names:
                if name not in cm._groups:
                    continue
                d = depths.get(name, 0)
                alpha = float(min(max_alpha, d + 1))   # deeper = higher priority
                stage_priors[name] = (alpha, 1.0)

            cm._scheduling_bandit = scheduling_bandit(
                stage_names, seed=bandit_seed, stage_priors=stage_priors or None
            )
            warm_str = "  ".join(
                f"{n}=Beta({a:.0f},1)" for n, (a, _) in stage_priors.items()
            )
            cm._log.info(
                f"SchedulingBandit enabled: {len(stage_names)} stages  "
                f"[{' '.join(stage_names)}]"
                + (f"  warm-start: {warm_str}" if warm_str else "")
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
        min_replicas: int = 0,
        max_replicas: int = 0,
        priority: int = 0,
        required_cpus: int = 0,
        required_gpus: int = 0,
        config: Optional[dict] = None,
    ) -> None:
        entry_point = self._resolve_entry_point(workflow_class)
        effective_max = max_replicas if max_replicas > 0 else replicas

        self._groups[name] = _GroupInfo(
            name=name,
            workflow_class=workflow_class,
            replicas=replicas,
            dependencies=list(dependencies or []),
            group_config=config,
            configured_replicas=replicas,
            min_replicas=min_replicas,
            max_replicas=effective_max,
            priority=priority,
            required_cpus=required_cpus,
            required_gpus=required_gpus,
            dep_threshold=dep_threshold,
            entry_point=entry_point,
        )
        self._stats[name] = WorkflowStats()
        self._log.info(
            f"Registered group {name!r}: replicas={replicas} "
            f"min={min_replicas} max={effective_max} "
            f"deps={dependencies or []} dep_threshold={dep_threshold} "
            f"resources=(cpus={required_cpus}, gpus={required_gpus})"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _setup_resources(self) -> None:
        """Sync CM resource state against the pre-built asyncflow engine.

        Called once from start(). The caller (run_campaign.py) is responsible
        for creating the backend and WorkflowEngine before passing asyncflow=
        to from_config() / __init__. This method only does CM-side setup:
        debug logging, GPU pool discovery, and ResourcePool cap correction.

        Raises RuntimeError if asyncflow was not provided.
        """
        if self._asyncflow is None:
            raise RuntimeError(
                "asyncflow engine not provided — create the backend and "
                "WorkflowEngine in your run script and pass asyncflow= to from_config()"
            )

        if self._debug:
            try:
                from rhapsody import enable_logging
                enable_logging(level="DEBUG")
                import logging as _logging
                _logging.getLogger("radical.asyncflow").setLevel(_logging.WARNING)
                _logging.getLogger("asyncio").setLevel(_logging.WARNING)
                self._log.warning("rhapsody.enable_logging active")
            except ImportError:
                self._log.warning("rhapsody.enable_logging not available — skipping")

        from .gpu import find_gpus, detect_gpus

        if self._engine_type == "dragon":
            self._gpu_pool = find_gpus()
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
                + (", ".join(f"{h}:{g}" for h, g in self._gpu_pool) or "none found")
            )
        else:
            # Concurrent mode: auto-detect CUDA GPUs for assignment tracking.
            if not self._free_gpu_ids:
                n = detect_gpus()
                self._free_gpu_ids = list(range(n))
                if n:
                    self._log.info(f"Concurrent mode: auto-detected {n} GPU(s) for assignment")

        self._log.info(f"CM ready (engine={self._engine_type})")

    async def start(self) -> None:
        """Kick off the campaign — schedule all eligible groups."""
        if not self._groups:
            self._all_done.set()
            return
        await self._setup_resources()
        res = self._resources
        if res.total_cpus > 0 or res.total_gpus > 0:
            self._log.info(
                f"Resource pool: total_cpus={res.total_cpus}  total_gpus={res.total_gpus}"
            )
        if self._monitor is not None:
            self._monitor_task = self._start_monitor_loop(self._monitor_interval_s)
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
        """Release CM resources (asyncflow shutdown left to caller)."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._asyncflow = None
        self._metrics.finish()
        self._log.info("AsyncCampaignManager closed")

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    async def signal_done(self, group_name: str) -> None:
        """Signal that *group_name* has produced output; queue 1 replica in each dependent."""
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
        candidate_id: Optional[str] = None,
        score: float = 0.0,
        surrogate_pred: float = 0.0,
        surrogate_unc: float = 0.0,
        scaffold_class: str = "",
        source_stage: str = "",
    ) -> None:
        """Queue replicas of the dependent group *name*.

        When candidate signals are supplied (candidate_id, score, …) the call
        is treated as a single candidate trigger:
          1. The result is recorded in CandidateLog for *source_stage*.
          2. threshold_top_fraction of *source_stage* gates whether the candidate
             enters the sharder buffer.
          3. The sharder ranks the buffer by profile-weighted priority on dispatch.

        When candidate_id is None the call is a count-based trigger (legacy API):
        *replicas* anonymous entries are added to the buffer with score=0.
        Routes directly to group.replicas when no sharder is registered.
        """
        async with self._lock:
            group = self._groups.get(name)
            if group is None:
                self._log.warning(f"trigger_dependent: group {name!r} not registered — ignoring")
                return
            if config:
                group.group_config = {**(group.group_config or {}), **config}

            sharder = self._sharders.get(name)
            if sharder is not None:
                if candidate_id is not None:
                    # ── Candidate-aware single-trigger path ──────────────────
                    enqueue_time = None
                    if self._candidate_log and source_stage:
                        result = self._candidate_log.record(
                            candidate_id, source_stage, score,
                            surrogate_pred, surrogate_unc, scaffold_class,
                        )
                        enqueue_time = self._candidate_log.get(candidate_id).enqueue_time
                        src_group = self._groups.get(source_stage)
                        top_frac = float(
                            (src_group.group_config or {}).get("threshold_top_fraction", 1.0)
                        ) if src_group else 1.0
                        if not self._candidate_log.passes_threshold(
                            candidate_id, source_stage, top_frac
                        ):
                            cutoff = self._candidate_log.threshold_cutoff(source_stage, top_frac)
                            self._log.info(
                                f"  Filtered {candidate_id!r} at {source_stage!r}: "
                                f"score={score:.4f} < cutoff={cutoff:.4f} "
                                f"(top-{top_frac:.0%})"
                            )
                            result.decision = "filtered"
                            return
                        result.decision = "passed"
                    sharder.receive(
                        candidate_id=candidate_id,
                        score=score,
                        surrogate_pred=surrogate_pred,
                        surrogate_unc=surrogate_unc,
                        scaffold_class=scaffold_class,
                        enqueue_time=enqueue_time,
                    )
                    self._log.info(
                        f"trigger_dependent: {name!r} candidate={candidate_id!r} "
                        f"score={score:.4f} → shard buffer (buffered={sharder.buffered})"
                    )
                else:
                    # ── Anonymous count-based path (legacy) ──────────────────
                    for _ in range(replicas):
                        anon_id = f"cand_{name}_{next(self._cand_seq):06d}"
                        sharder.receive(
                            candidate_id=anon_id,
                            score=score,
                            surrogate_pred=surrogate_pred,
                            surrogate_unc=surrogate_unc,
                            scaffold_class=scaffold_class,
                        )
                    self._log.info(
                        f"trigger_dependent: {name!r} +{replicas} anonymous → shard buffer "
                        f"(buffered={sharder.buffered})"
                    )
            else:
                group.replicas += replicas
                group.configured_replicas += replicas
                if group.status == "done":
                    group.status = "pending"
                self._log.info(
                    f"trigger_dependent: {name!r} +{replicas} replicas (total={group.replicas})"
                )
        await self._schedule()

    async def trigger_candidate(
        self,
        name: str,
        candidate_id: str,
        score: float = 0.0,
        surrogate_pred: float = 0.0,
        surrogate_unc: float = 0.0,
        scaffold_class: str = "",
        source_stage: str = "",
        config: Optional[dict] = None,
    ) -> None:
        """Convenience wrapper for a single named-candidate trigger.

        Equivalent to trigger_dependent(name, replicas=1, candidate_id=...).
        Workflows prefer this over trigger_dependent when they have scored results.
        """
        await self.trigger_dependent(
            name=name,
            replicas=1,
            config=config,
            candidate_id=candidate_id,
            score=score,
            surrogate_pred=surrogate_pred,
            surrogate_unc=surrogate_unc,
            scaffold_class=scaffold_class,
            source_stage=source_stage,
        )

    async def trigger_batch(
        self,
        name: str,
        candidates: list[dict],
    ) -> None:
        """Add N candidates to the sharder buffer in one lock acquisition.

        Each dict in *candidates* must contain ``candidate_id`` and may include
        ``score``, ``scaffold_class``, ``surrogate_pred``, ``surrogate_unc``.

        Unlike N sequential trigger_dependent() calls (each of which calls
        _schedule() after releasing the lock), this method holds the lock through
        all sharder.receive() calls and calls _schedule() exactly once.  The
        sharder buffer therefore accumulates N candidates before the first
        dispatch() runs, enabling meaningful priority ranking across the batch.

        Routes directly to group.replicas when no sharder is registered.
        """
        async with self._lock:
            group = self._groups.get(name)
            if group is None:
                self._log.warning(f"trigger_batch: group {name!r} not registered — ignoring")
                return
            sharder = self._sharders.get(name)
            n_added = 0
            for cand in candidates:
                candidate_id   = str(cand["candidate_id"])
                score          = float(cand.get("score", 0.0))
                scaffold_class = str(cand.get("scaffold_class", ""))
                surrogate_pred = float(cand.get("surrogate_pred", 0.0))
                surrogate_unc  = float(cand.get("surrogate_unc", 0.0))
                if sharder is not None:
                    sharder.receive(
                        candidate_id=candidate_id,
                        score=score,
                        surrogate_pred=surrogate_pred,
                        surrogate_unc=surrogate_unc,
                        scaffold_class=scaffold_class,
                    )
                else:
                    group.replicas += 1
                    group.configured_replicas += 1
                    if group.status == "done":
                        group.status = "pending"
                n_added += 1
            self._log.info(
                f"trigger_batch: {name!r} +{n_added} candidates "
                f"(buffered={sharder.buffered if sharder else '—'})"
            )
        await self._schedule()

    # ------------------------------------------------------------------
    # Status / stats
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "resources": self._resources.as_dict(),
            "groups": {
                name: {
                    "status": g.status,
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
        return {name: WorkflowStats(**vars(s)) for name, s in self._stats.items()}

    def metrics(self) -> CampaignMetrics:
        """Return the live metrics recorder for this campaign run."""
        return self._metrics

    # ------------------------------------------------------------------
    # Internal — schedule dispatch (called outside the lock)
    # ------------------------------------------------------------------

    async def _schedule(self) -> None:
        async with self._lock:
            to_start = self._schedule_locked()
        for group, replica_idx in to_start:
            asyncio.get_running_loop().create_task(self._run_replica(group, replica_idx))
