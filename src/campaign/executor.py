"""
ExecutorMixin — replica launch, completion, and monitor logic for AsyncCampaignManager.

Mixed into AsyncCampaignManager; all methods use ``self`` to access shared
state (``_groups``, ``_resources``, ``_sharders``, ``_monitor``, ``_log``).
"""

import asyncio
from typing import TYPE_CHECKING

from .backpressure import BPState
from .gpu import make_policies
from .types import _GroupInfo

if TYPE_CHECKING:
    from .base_workflow import BaseWorkflow


def _campaign_complete(groups: dict, sharders: dict) -> bool:
    """True when every group has finished all its replicas and all sharder buffers are empty.

    Deliberately does NOT use group.status so it works even when the
    deps_done status-propagation chain stalls (e.g. a downstream stage
    finishes all replicas before its upstream is marked 'done').
    """
    if not groups:
        return False
    if not any(g.replicas > 0 for g in groups.values()):
        return False   # nothing has started yet
    for g in groups.values():
        if g.replicas == 0:
            continue   # not yet activated
        if g.running_count > 0:
            return False
        if g.finished_replicas < g.replicas:
            return False
    if any(s.buffered > 0 for s in sharders.values()):
        return False
    return True


class ExecutorMixin:

    async def _run_replica(self, group: _GroupInfo, replica_idx: int) -> None:
        """Execute one replica of a workflow group."""
        replica_id = f"{group.name}_{replica_idx}"
        final_state = "done"

        gpu_ids = self._replica_gpu_assignments.get(replica_id, [])
        policies = make_policies(self._gpu_pool, gpu_ids)

        res_tag = ""
        if group.required_cpus > 0 or group.required_gpus > 0:
            res_tag = f" [cpus={group.required_cpus} gpus={group.required_gpus}]"
        if gpu_ids:
            host = self._gpu_pool[0][0] if self._gpu_pool else "?"
            res_tag += f" [gpu_affinity={gpu_ids} host={host}]"
        self._log.info(f"  starting replica {replica_id!r}{res_tag}")

        # Build per-replica config: start from group config, layer in GPU and candidate info.
        replica_config = group.group_config
        if gpu_ids:
            replica_config = {
                **(replica_config or {}),
                "assigned_gpu_ids": gpu_ids,
                "group_gpu_ids": list(group.running_gpu_ids),
            }
        candidate_id = self._replica_candidate_assignments.pop(replica_id, None)
        score = None
        if candidate_id and self._candidate_log:
            h = self._candidate_log.get(candidate_id)
            if h:
                score = h.latest_score
                replica_config = {
                    **(replica_config or {}),
                    "candidate_id":       candidate_id,
                    "candidate_score":    h.latest_score,       # upstream quality score
                    "candidate_surr":     h.latest_surrogate_pred,  # surrogate model prediction
                    "candidate_surr_unc": h.latest_surrogate_unc,   # surrogate uncertainty
                    "candidate_scaffold": h.scaffold_class,          # chemical scaffold class
                }
            else:
                replica_config = {**(replica_config or {}), "candidate_id": candidate_id}
        self._metrics.record_replica_start(group.name, replica_id, candidate_id=candidate_id, score=score)
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
            self._log.error(f"Replica {replica_id!r} raised: {type(exc).__name__}: {exc}")
            final_state = "failed"

        await self._handle_replica_done(wf, group, replica_id, replica_idx, final_state)

    async def _handle_replica_done(
        self,
        wf: "BaseWorkflow",
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

        self._metrics.record_replica_finish(group.name, replica_id, final_state)
        await self._on_replica_finished(group, replica_id)

    async def _on_replica_finished(self, group: _GroupInfo, replica_id: str) -> None:
        """Update group counters, notify sharders, run monitor, then re-schedule."""
        group_done = False
        async with self._lock:
            group.finished_replicas += 1
            group.running_count -= 1
            self._resources.release(group.required_cpus, group.required_gpus)
            self._stats[group.name].replicas_finished = group.finished_replicas

            if group.finished_replicas >= group.replicas:
                # Dependent groups receive triggers incrementally while their
                # upstream runs, so finished==replicas fires spuriously after
                # every single completion (e.g. 1/1 when only 1 trigger has
                # arrived and more are still coming).  Only mark truly done
                # when all upstream dependencies are also done — i.e., no
                # more triggers can arrive from them.
                deps_done = not group.dependencies or all(
                    self._groups.get(d) is not None
                    and self._groups[d].status == "done"
                    for d in group.dependencies
                )
                if deps_done:
                    group.status = "done"
                    group_done = True

            # Update scheduling bandit: reward for this group based on downstream BP.
            if self._scheduling_bandit is not None:
                downstream_name = (group.group_config or {}).get("trigger_downstream")
                bp_ctrl = self._bp.get(downstream_name) if downstream_name else None
                if bp_ctrl is not None and bp_ctrl.state in (BPState.THROTTLE, BPState.WIDEN):
                    # Use BP state only for the extreme cases where it carries a clear
                    # directional signal: THROTTLE means this stage is flooding its
                    # downstream (back off), WIDEN means downstream is starved (run more).
                    sched_reward = 0.8 if bp_ctrl.state == BPState.WIDEN else 0.2
                else:
                    # BP HOLD (healthy) or no BP at all: use downstream utilisation as a
                    # fine-grained reward signal.  This lets the bandit differentiate
                    # stages even when BP never fires (all high_waters are above peak queue).
                    # Terminal stage (no downstream) → max reward; every finish directly
                    # counts toward the campaign target.
                    if downstream_name is None:
                        sched_reward = 1.0
                    else:
                        downstream_grp = self._groups.get(downstream_name)
                        if downstream_grp is not None and downstream_grp.max_replicas > 0:
                            util = downstream_grp.running_count / downstream_grp.max_replicas
                            sched_reward = max(0.2, 1.0 - 0.5 * util)
                        else:
                            sched_reward = 0.5
                self._scheduling_bandit.update(group.name, sched_reward)

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
                    f"{rid}→{gids}"
                    for rid, gids in sorted(self._replica_gpu_assignments.items())
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
            # Notify downstream sharders: no more triggers from this group,
            # so strict-stratify partial tails are safe to flush.
            for sh_name, sharder in self._sharders.items():
                sh_group = self._groups.get(sh_name)
                if sh_group and group.name in sh_group.dependencies:
                    sharder.mark_upstream_done()
                    self._log.info(
                        f"Sharder [{sh_name}]: upstream {group.name!r} done "
                        f"— partial tail ({sharder.buffered}) will flush next cycle"
                    )

        self._log.info(
            f"_on_replica_finished: {group.name!r} - "
            f"finished_replicas={group.finished_replicas}/{group.replicas}"
        )

        # ── Monitor: pass-through and budget drift checks ─────────────────────
        if self._monitor and group.finished_replicas > 0:
            grp_cfg = group.group_config or {}
            trigger_name = grp_cfg.get("trigger_downstream")
            expected_frac = float(grp_cfg.get("trigger_fraction", 1.0))
            _MIN_PASSTHROUGH_SAMPLE = 10
            if (trigger_name and trigger_name in self._groups and expected_frac < 1.0
                    and group.finished_replicas >= _MIN_PASSTHROUGH_SAMPLE):
                downstream_total = self._groups[trigger_name].replicas
                observed_frac = downstream_total / group.finished_replicas
                ev = self._monitor.check_passthrough(group.name, observed_frac, expected_frac)
                if ev:
                    tag = " [ESCALATING]" if self._monitor.is_escalating(ev) else ""
                    self._log.warning(
                        f"Monitor [{group.name}] pass_through drift{tag}: "
                        f"observed={observed_frac:.3f}  expected={expected_frac:.3f}"
                        f"  dev={ev.deviation_pct:.1f}%  breach={ev.breach_count}"
                    )

            budget = float(grp_cfg.get("budget_node_hours") or 0)
            if budget > 0 and group.replicas > 0:
                pilot = grp_cfg.get("pilot", {})
                nodes = int(pilot.get("nodes", 1))
                walltime_h = float(pilot.get("walltime_h", 1))
                spent_actual    = nodes * walltime_h * group.finished_replicas / group.replicas
                expected_so_far = budget * group.finished_replicas / group.replicas
                ev = self._monitor.check_budget(group.name, spent_actual, expected_so_far)
                if ev:
                    self._log.warning(
                        f"Monitor [{group.name}] budget drift: "
                        f"spent={spent_actual:.1f}  expected={expected_so_far:.1f} node-hours"
                        f"  dev={ev.deviation_pct:.1f}%"
                    )

        # ── Early termination: downstream_input_target ────────────────────────
        # Check BEFORE scheduling so that when the target is hit, _schedule_locked
        # sees _all_done=True and returns [] immediately — no new replicas start.
        if not self._all_done.is_set():
            for gname, g in self._groups.items():
                target = int((g.group_config or {}).get("downstream_input_target") or 0)
                if target > 0 and g.finished_replicas >= target:
                    self._all_done.set()
                    self._log.info(
                        f"Campaign target reached: {gname!r} finished "
                        f"{g.finished_replicas}/{target} replicas — stopping early"
                    )
                    return   # _schedule_locked will be a no-op for all future calls

        await self._schedule()

        async with self._lock:
            all_done = _campaign_complete(self._groups, self._sharders)

        if all_done:
            self._all_done.set()
            self._log.info("All campaign workflow groups finished")
