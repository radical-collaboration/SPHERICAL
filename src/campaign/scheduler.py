"""
SchedulerMixin — two-pass greedy scheduling logic for AsyncCampaignManager.

Mixed into AsyncCampaignManager; all methods use ``self`` to access shared
state (``_groups``, ``_resources``, ``_bp``, ``_sharders``, ``_log``).

Scheduling model
----------------
Every state change (replica finished, trigger received) calls ``_schedule``,
which holds the lock, calls ``_schedule_locked``, then fires the resulting
tasks outside the lock.

Pass 1 — guarantee ``min_replicas`` for all eligible groups (highest priority).
Pass 2 — fill remaining capacity up to ``max_replicas`` (highest priority).

A group is eligible when its dependencies are satisfied and it has replicas
waiting to be started.
"""

from .backpressure import BPState
from .types import _GroupInfo


class SchedulerMixin:

    # ------------------------------------------------------------------
    # Dependency and resource checks (must be called under self._lock)
    # ------------------------------------------------------------------

    def _deps_satisfied_locked(self, group: _GroupInfo) -> bool:
        """True when every dependency group is considered ready.

        Ready means any of:
        - dep.status == "done"  (all replicas finished — always satisfies, regardless of threshold)
        - explicit ``signal_ready()`` call (workflow-driven), OR
        - ``dep_threshold`` or more finished replicas (count-based fallback).

        The status=="done" path enables true sequential waterfall when dep_threshold
        is set very high: the downstream stage is blocked until the upstream group
        fully completes every replica, not just the first dep_threshold ones.
        """
        for dep_name in group.dependencies:
            dep = self._groups.get(dep_name)
            if dep is None:
                return False
            if dep.status == "done":
                continue  # fully completed upstream always satisfies dependency
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
        effective_max = group.max_replicas if group.max_replicas > 0 else group.replicas
        if group.running_count >= effective_max:
            return False
        if not self._deps_satisfied_locked(group):
            return False
        if not self._resources.can_fit(group.required_cpus, group.required_gpus):
            return False
        return True

    def _allocate_locked(self, group: _GroupInfo) -> int:
        """Record one replica start for *group*; update counters; return replica idx."""
        idx = group.started_count
        group.started_count += 1
        group.running_count += 1
        self._resources.allocate(group.required_cpus, group.required_gpus)
        self._stats[group.name].replicas_started = group.started_count
        replica_id = f"{group.name}_{idx}"
        # Assign the next pending candidate ID to this replica (FIFO from shard dispatch).
        if group._pending_candidates:
            self._replica_candidate_assignments[replica_id] = group._pending_candidates.popleft()
        gpu_ids = [
            self._free_gpu_ids.pop(0)
            for _ in range(group.required_gpus)
            if self._free_gpu_ids
        ]
        self._replica_gpu_assignments[replica_id] = gpu_ids
        group.running_gpu_ids.extend(gpu_ids)
        if gpu_ids:
            self._log.info(
                f"  GPU assign: {replica_id!r} → GPU(s) {gpu_ids}"
                f"  | free: {sorted(self._free_gpu_ids)}"
            )
        return idx

    # ------------------------------------------------------------------
    # Sharder flush (must be called under self._lock)
    # ------------------------------------------------------------------

    def _flush_sharders_locked(self) -> None:
        """Drain shard buffers into their groups' runnable replica queues.

        dispatch() returns a priority-ordered list of candidate IDs.  Each ID
        is appended to group._pending_candidates; _allocate_locked pops them
        FIFO so replica_idx → candidate_id mapping is stable.
        """
        for name, sharder in self._sharders.items():
            if sharder.buffered <= 0:
                continue
            g = self._groups.get(name)
            if g is None:
                continue
            bp        = self._bp.get(name)
            cap       = g.max_replicas if g.max_replicas > 0 else max(g.replicas, 1)
            occupancy = min(1.0, g.running_count / cap)
            # Collect scaffold classes of currently-running replicas for diversity scoring.
            running_scaffolds: set[str] = set()
            if self._candidate_log:
                for rid, cid in self._replica_candidate_assignments.items():
                    if rid.startswith(f"{name}_"):
                        h = self._candidate_log.get(cid)
                        if h and h.scaffold_class:
                            running_scaffolds.add(h.scaffold_class)
            dispatched = sharder.dispatch(
                bp=bp, occupancy=occupancy, running_scaffolds=running_scaffolds or None
            )
            n = len(dispatched)
            if n > 0:
                g.replicas += n
                g.configured_replicas += n
                g._pending_candidates.extend(dispatched)
                if g.status == "done":
                    g.status = "pending"
                self._log.info(
                    f"Sharder [{name}]: dispatched {n} → runnable "
                    f"(buffered={sharder.buffered}  queue_depth={g.replicas - g.started_count})"
                )
            elif sharder.buffered > 0:
                self._log.info(
                    f"Sharder [{name}]: holding {sharder.buffered} "
                    f"(bp={bp.state.value if bp else 'none'}  occupancy={occupancy:.2f})"
                )

    # ------------------------------------------------------------------
    # Main scheduler (must be called under self._lock)
    # ------------------------------------------------------------------

    def _schedule_locked(self) -> list[tuple[_GroupInfo, int]]:
        """Two-pass greedy scheduler.  Must be called under ``self._lock``.

        Returns a list of (group, replica_idx) pairs to start.
        """
        to_start: list[tuple[_GroupInfo, int]] = []

        # Stop scheduling immediately after early termination or natural completion.
        if self._all_done.is_set():
            return to_start

        # ── Sharder: flush buffers into runnable queues ──────────────────────
        if self._sharders:
            self._flush_sharders_locked()

        # ── Backpressure: refresh state for all controlled groups ────────────
        if self._features.get("backpressure"):
            for bp_name, bp_ctrl in self._bp.items():
                if bp_name in self._groups:
                    g = self._groups[bp_name]
                    queue_depth = max(0, g.replicas - g.started_count)
                    old_state = bp_ctrl.state
                    bp_ctrl.step(queue_depth)
                    if bp_ctrl.state != old_state:
                        # Suppress the trivial HOLD→WIDEN at startup (empty queue
                        # always triggers this; it carries no actionable information).
                        startup_widen = (
                            old_state.value == "hold"
                            and bp_ctrl.state.value == "widen"
                            and queue_depth == 0
                        )
                        if not startup_widen:
                            level = (
                                self._log.warning
                                if bp_ctrl.state == BPState.THROTTLE
                                else self._log.info
                            )
                            level(
                                f"Backpressure [{bp_name}]: "
                                f"{old_state.value} → {bp_ctrl.state.value}"
                                f"  (queue_depth={queue_depth})"
                            )
                        self._metrics.record_bp_transition(
                            bp_name, old_state.value, bp_ctrl.state.value, queue_depth
                        )

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

        if self._scheduling_bandit is not None:
            eligible = self._scheduling_bandit.rank(eligible)
        else:
            # Sort by group priority (higher = scheduled first).
            # stable sort: equal-priority groups keep registration order (FIFO).
            eligible = sorted(eligible, key=lambda g: -g.priority)

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

        # Warn about groups stalled on resources.
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
            bandit_scores: dict = {}
            if self._scheduling_bandit is not None:
                bandit_scores = {
                    g.name: self._scheduling_bandit._arms[g.name].sample(
                        self._scheduling_bandit._rng
                    )
                    for g in eligible if g.name in self._scheduling_bandit._arms
                }
            self._metrics.record_scheduling(
                chosen_groups=[g.name for g, _ in to_start],
                eligible_groups=[g.name for g in eligible],
                bandit_scores=bandit_scores,
            )

            def _gpu_tag(g, idx):
                ids = self._replica_gpu_assignments.get(f"{g.name}_{idx}", [])
                return f"gpu={ids}" if ids else ""

            summary = ", ".join(
                f"{g.name}_{idx}" + (f"[{_gpu_tag(g, idx)}]" if _gpu_tag(g, idx) else "")
                for g, idx in to_start
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
            buf_str = {n: s.buffered for n, s in self._sharders.items() if s.buffered}
            col = max(len(g.name) for g in self._groups.values()) + 2
            group_lines = "\n".join(
                f"  {g.name:<{col}} run={g.running_count:<3} "
                f"done={g.finished_replicas}/{g.replicas}"
                + (f"  buf={buf_str[g.name]}" if g.name in buf_str else "")
                for g in self._groups.values()
            )
            res_line = f"  {self._resources.usage_str()}"
            bandit_line = ""
            if self._scheduling_bandit is not None:
                bsum = self._scheduling_bandit.summary()
                best = self._scheduling_bandit.best()
                bandit_line = (
                    "\n  sched_bandit_best=" + repr(best) + "  "
                    + " ".join(f"{n}:{v:.2f}" for n, v in bsum.items())
                )
            self._log.info(
                f"Scheduling: [{summary}]  viz=[{viz}]\n"
                + group_lines + "\n"
                + res_line
                + bandit_line
            )

        return to_start
