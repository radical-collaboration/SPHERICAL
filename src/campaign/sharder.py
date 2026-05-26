"""
Per-stage sharder: buffers upstream triggers and batch-dispatches downstream.

The sharder sits between the upstream producer (trigger_dependent signals) and
the downstream execution queue (group.replicas counter).  Incoming candidates
accumulate in a buffer; each scheduling cycle the CM calls dispatch() to move
a priority-ordered batch from the buffer into the runnable queue.

Priority scoring
----------------
Each candidate entry carries (score, surrogate_pred, surrogate_unc,
scaffold_class, enqueue_time).  dispatch() ranks buffered candidates under the
active ProfileWeights before selecting the top-N to release.

Five signals, all percentile-ranked within the buffer:
  score       — upstream stage score
  surrogate   — surrogate model prediction
  uncertainty — surrogate uncertainty
  age         — time since enqueue (anti-starvation)
  diversity   — novelty: less-common scaffold class in buffer → higher score;
                scaffolds already running in the downstream group score 0

threshold_top_fraction
----------------------
Set top_fraction < 1.0 on a stage to gate candidates at trigger time.
Only candidates whose score is in the top fraction of all scores seen so far
at that stage are accepted into the buffer.  Gating is handled by the CM
(trigger_dependent) before receive() is called; the sharder itself does not
re-check the threshold.

Backpressure and stratify semantics are unchanged from the integer-buffer version.

Bandit mode (use_bandit: true)
-------------------------------
Thompson-sampling bandit selects the BP multiplier from
[0.50, 0.75, 1.00, 1.25, 1.50].  Feedback is the BP state of the PREVIOUS
dispatch cycle.  In strict mode the selected factor is clamped to ≥ 1.0 so
the bandit's reward is consistent with what the sharder actually dispatched
(strict mode never sends fewer than target_size).
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .backpressure import BackpressureNegotiator, BPState

if TYPE_CHECKING:  # pragma: no cover
    from .bandit import Bandit
    from .profiles import ProfileWeights


# ── Candidate entry ────────────────────────────────────────────────────────────

@dataclass
class _CandidateEntry:
    candidate_id:   str
    score:          float = 0.0
    surrogate_pred: float = 0.0
    surrogate_unc:  float = 0.0
    scaffold_class: str   = ""
    enqueue_time:   float = 0.0   # set by Sharder.receive() via its now_fn


# ── Percentile-rank helper ─────────────────────────────────────────────────────

def _percentile_rank(values: list[float]) -> list[float]:
    """Map values to percentile ranks in [0, 1]. Stable for ties.

    Special case: a single value returns [1.0] so a one-element buffer
    doesn't zero out the score/surrogate/uncertainty contributions to
    priority.  (np.linspace(0, 1, 1) is [0.0], which would otherwise
    suppress real signal magnitude on tail-of-campaign single dispatches.)
    """
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(arr))
    return ranks.tolist()


# ── ShardingSpec ───────────────────────────────────────────────────────────────

@dataclass
class ShardingSpec:
    """Plan-level sharding bounds and dispatch strategy for one stage."""
    target_size:  int            = 50
    min_size:     int            = 1
    max_size:     int            = 200
    stratify:     str            = "soft"   # off | soft | strict
    top_fraction: float          = 1.0      # threshold_top_fraction gate (1.0 = no gate)
    profile:      str            = "diverse_top"
    use_bandit:   bool           = False
    bandit_seed:  Optional[int]  = None

    @classmethod
    def from_dict(cls, d: dict) -> "ShardingSpec":
        return cls(
            target_size  = int(d.get("target_size",   50)),
            min_size     = int(d.get("min_size",       1)),
            max_size     = int(d.get("max_size",      200)),
            stratify     = str(d.get("stratify",    "soft")),
            top_fraction = float(d.get("top_fraction", 1.0)),
            profile      = str(d.get("profile", "diverse_top")),
            use_bandit   = bool(d.get("use_bandit",  False)),
            bandit_seed  = d.get("bandit_seed"),
        )


# ── Sharder ────────────────────────────────────────────────────────────────────

@dataclass
class Sharder:
    """Buffer + priority-dispatch controller for one downstream stage."""
    name: str
    spec: ShardingSpec

    _buf:            list                                            = field(default_factory=list, init=False)
    _upstream_done:  bool                                            = field(default=False,        init=False)
    _last_factor:    Optional[float]                                 = field(default=None,         init=False)
    _last_occupancy: float                                           = field(default=0.0,          init=False)
    _shard_seq:      int                                             = field(default=0,            init=False)
    _bandit:         Optional["Bandit"]                              = field(default=None,         init=False)
    _profile:        Optional["ProfileWeights"]                      = field(default=None,         init=False)
    _log_fn:         Optional[Callable[[str], None]]                 = field(default=None,         init=False)
    _metrics_fn:     Optional[Callable[[int, int, list, list], None]] = field(default=None,        init=False)
    _now_fn:         Callable[[], float]                             = field(default=time.time,    init=False)

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    def __post_init__(self) -> None:
        if self.spec.use_bandit:
            from .bandit import shard_bandit
            self._bandit = shard_bandit(seed=self.spec.bandit_seed)
        from .profiles import get_profile
        self._profile = get_profile(self.spec.profile)

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_profile(self, profile_name: str) -> None:
        """Switch the active priority profile (e.g., on surrogate redeploy
        or when the downstream input target nears completion).
        """
        from .profiles import get_profile
        self.spec.profile = profile_name
        self._profile = get_profile(profile_name)

    def set_now_fn(self, fn: Callable[[], float]) -> None:
        """Inject a clock (default time.time) so the age signal and enqueue
        timestamps work under simulated time (e.g., SimPy env.now)."""
        self._now_fn = fn

    # ── Producer side ─────────────────────────────────────────────────────────

    def receive(
        self,
        candidate_id: str,
        score: float = 0.0,
        surrogate_pred: float = 0.0,
        surrogate_unc: float = 0.0,
        scaffold_class: str = "",
        enqueue_time: Optional[float] = None,
    ) -> None:
        """Accept one candidate into the buffer."""
        self._buf.append(_CandidateEntry(
            candidate_id=candidate_id,
            score=score,
            surrogate_pred=surrogate_pred,
            surrogate_unc=surrogate_unc,
            scaffold_class=scaffold_class,
            enqueue_time=enqueue_time if enqueue_time is not None else self._now_fn(),
        ))

    def mark_upstream_done(self) -> None:
        """Signal that no more triggers will arrive.

        Behavior on the next dispatch():
          - strict mode: flush ALL remaining candidates (priority-ordered) so
            the stage doesn't deadlock waiting for a full target_size batch
            that will never arrive.
          - soft / off mode: no effect — the campaign terminates via natural
            completion or _all_done elsewhere in the CM; flushing here would
            release low-priority candidates that should never run.
        """
        self._upstream_done = True

    def clear(self) -> None:
        """Drop all buffered candidates. Used when a stage is cancelled or
        when the CM needs to reset state (e.g., after a drain → resume)."""
        n = len(self._buf)
        self._buf.clear()
        self._log(f"  Sharder [{self.name}] cleared: dropped {n} buffered")

    @property
    def buffered(self) -> int:
        return len(self._buf)

    # ── Priority scoring ───────────────────────────────────────────────────────

    def _score_entries(
        self,
        entries: list[_CandidateEntry],
        now: float,
        running_scaffolds: Optional[set] = None,
    ) -> list[float]:
        """Compute one priority score per entry under the active profile.

        Signals are percentile-ranked within the buffer so all profiles operate
        on a common [0, 1] scale regardless of raw value magnitudes.
        Diversity: scaffold classes already running downstream score 0;
        within the buffer, less common scaffold class → higher diversity.
        """
        if not entries:
            return []
        w = self._profile
        already = running_scaffolds or set()

        score_n = _percentile_rank([e.score for e in entries])
        surr_n  = _percentile_rank([e.surrogate_pred for e in entries])
        unc_n   = _percentile_rank([e.surrogate_unc  for e in entries])

        ages    = [max(0.0, now - e.enqueue_time) for e in entries]
        age_max = max(ages) if max(ages) > 0 else 1.0
        age_n   = [a / age_max for a in ages]

        counts: dict[str, int] = {}
        for e in entries:
            counts[e.scaffold_class] = counts.get(e.scaffold_class, 0) + 1
        max_count = max(counts.values()) if counts else 1
        diversity = [
            0.0 if e.scaffold_class in already
            else 1.0 - (counts[e.scaffold_class] - 1) / max_count
            for e in entries
        ]

        return [
            w.score       * score_n[i]
            + w.surrogate   * surr_n[i]
            + w.uncertainty * unc_n[i]
            + w.age         * age_n[i]
            + w.diversity   * diversity[i]
            for i in range(len(entries))
        ]

    # ── Bandit helpers ─────────────────────────────────────────────────────────

    def _update_bandit(self, bp_state: Optional[BPState]) -> None:
        """Feed reward for the previous dispatch factor.

        Reward formula:
          THROTTLE → 0.1 + 0.3 × occupancy
            (previous factor flooded the queue; partial credit for high
             occupancy so we don't converge to the smallest factor just
             to avoid any THROTTLE.)
          HOLD     → 0.8  (queue depth healthy, flat — no occupancy signal)
          WIDEN    → 0.2 + 0.7 × occupancy
            (high occupancy means the previous dispatch kept the stage
             well-fed despite the queue draining → good factor choice;
             low occupancy means the stage was starved.)

        Using occupancy rather than raw WIDEN/HOLD lets the bandit converge
        even when BP stays WIDEN permanently (queue drains between dispatches),
        which happens when makespan < dispatch interval.
        """
        if self._bandit is None or self._last_factor is None:
            return
        if bp_state == BPState.THROTTLE:
            reward = 0.1 + 0.3 * self._last_occupancy
        elif bp_state == BPState.HOLD:
            reward = 0.8
        else:  # WIDEN or no BP
            reward = 0.2 + 0.7 * self._last_occupancy
        prev_factor = self._last_factor
        self._bandit.update(prev_factor, reward)
        self._last_factor = None
        # Defer string formatting until we know logging is wired up — large
        # bandit summaries are expensive to stringify in hot dispatch loops.
        if self._log_fn is not None:
            bsum = self._bandit.summary()
            best = max(bsum, key=bsum.get)
            reward_src = (
                f"throttle+occ={self._last_occupancy:.2f}→{reward:.2f}" if bp_state == BPState.THROTTLE
                else f"hold→0.80"   if bp_state == BPState.HOLD
                else f"occ={self._last_occupancy:.2f}→{reward:.2f}"
            )
            self._log(
                f"  Sharder [{self.name}] shard_bandit update: "
                f"factor={prev_factor:.2f}  reward={reward_src}  "
                f"best={best:.2f}×  [{' '.join(f'{k:.2f}:{v:.2f}' for k, v in bsum.items())}]"
            )

    def _bp_factor(self, bp_state: Optional[BPState]) -> float:
        """Select dispatch multiplier from bandit or fixed BP→factor mapping.

        In strict mode the factor is floored at 1.0 so the bandit's selected
        arm matches what dispatch() actually emits (strict mode never sends
        fewer than target_size).  Without this clamp, bandit selections of
        0.5 or 0.75 would be silently promoted to 1.0 by the strict-mode
        floor in dispatch(), and the bandit's posterior would inflate the
        value of small arms that had no real effect.
        """
        if self._bandit is not None:
            factor = self._bandit.select()
            if self.spec.stratify == "strict":
                factor = max(factor, 1.0)
            self._last_factor = factor
            return factor
        # Fixed mapping
        if bp_state == BPState.THROTTLE:
            return 1.0 if self.spec.stratify == "strict" else 0.5
        if bp_state == BPState.WIDEN:
            return 1.5
        return 1.0

    def bandit_summary(self) -> Optional[dict]:
        if self._bandit is None:
            return None
        return self._bandit.summary()

    # ── Consumer side ──────────────────────────────────────────────────────────

    def adaptive_size(
        self,
        bp:        "BackpressureNegotiator | None",
        occupancy: float,
    ) -> int:
        """Compute dispatch batch size for this scheduling cycle.

        Note: this method has a side effect when use_bandit=True — it consumes
        one bandit arm selection via _bp_factor.  Call it only from dispatch()
        or from tests that intend to advance the bandit.
        """
        sh       = self.spec
        bp_state = bp.state if bp is not None else None
        buf_len  = len(self._buf)

        factor    = self._bp_factor(bp_state)
        size_bp   = max(sh.min_size, min(sh.max_size, int(sh.target_size * factor)))
        if self._bandit is not None:
            bp_tag = f"×{factor:.2f}(bandit) target={sh.target_size}→{size_bp}"
        else:
            bp_tag = (f"bp={bp_state.value if bp_state else 'none'} "
                      f"×{factor:.2f}(fixed) target={sh.target_size}→{size_bp}")

        if occupancy > 0.85:
            size_occ = max(sh.min_size, int(size_bp * 0.75))
            occ_tag  = f"occ={occupancy:.2f}(high ×0.75) {size_bp}→{size_occ}"
        elif occupancy < 0.40:
            size_occ = min(sh.max_size, int(size_bp * 1.25))
            occ_tag  = f"occ={occupancy:.2f}(low ×1.25) {size_bp}→{size_occ}"
        else:
            size_occ = size_bp
            occ_tag  = f"occ={occupancy:.2f}(ok)"

        if 0 < buf_len < size_occ and sh.stratify != "strict":
            size_final = max(sh.min_size, buf_len)
            tail_tag   = f"tail(buf={buf_len}<{size_occ})→{size_final}"
        else:
            size_final = size_occ
            tail_tag   = ""

        size_final = max(sh.min_size, min(sh.max_size, size_final))

        if self._log_fn is not None:
            parts = [bp_tag, occ_tag]
            if tail_tag:
                parts.append(tail_tag)
            self._log(
                f"  Sharder [{self.name}] adaptive: buf={buf_len} | "
                + " | ".join(parts)
                + f" | size={size_final}"
            )
        return size_final

    def dispatch(
        self,
        bp:               "BackpressureNegotiator | None",
        occupancy:        float,
        running_scaffolds: Optional[set] = None,
    ) -> list[str]:
        """Dispatch one priority-ordered shard from the buffer.

        Returns a list of candidate_ids (highest priority first).
        Returns [] when:
          - buffer is empty
          - BP state is THROTTLE
          - stratify=strict and buffer < target_size (unless upstream is done)

        Candidates are ranked by the active profile's weight vector before
        selection; the top-N by priority score are dispatched.
        """
        if not self._buf:
            return []

        bp_state = bp.state if bp is not None else None
        # Store occupancy so _update_bandit can use it as the reward signal
        # for the *previous* factor (occupancy at this dispatch = outcome of prev dispatch).
        self._last_occupancy = occupancy
        self._update_bandit(bp_state)

        # ── Upstream-done fast-path (strict only) ──────────────────────────────
        # STRICT mode: flush ALL remaining candidates in priority order so we
        # don't deadlock on a partial tail smaller than target_size.
        # SOFT/OFF: no effect — the campaign terminates via natural completion
        # or _all_done elsewhere; flushing here would release low-priority
        # candidates that should never run.
        if self._upstream_done and self.spec.stratify == "strict":
            now    = self._now_fn()
            scores = self._score_entries(self._buf, now, running_scaffolds)
            order  = sorted(range(len(self._buf)), key=lambda i: -scores[i])
            dispatched = [self._buf[i].candidate_id for i in order]
            self._buf.clear()
            self._shard_seq += 1
            self._log(
                f"  Sharder [{self.name}] [upstream-done strict-flush] "
                f"shard={self._shard_seq}: dispatched {len(dispatched)} "
                f"(priority-ordered)  buffered=0 remaining"
            )
            return dispatched

        if bp_state == BPState.THROTTLE:
            return []
        if (self.spec.stratify == "strict"
                and len(self._buf) < self.spec.target_size):
            return []

        if self.spec.stratify == "off":
            n = 1
        else:
            n = self.adaptive_size(bp, occupancy)
            # In strict mode floor at target_size so dispatch never sends
            # fewer than a full batch.  _bp_factor already clamps the bandit's
            # selection to ≥ 1.0 in strict mode; this is defense-in-depth for
            # the fixed-mapping path and any future factor changes.
            if self.spec.stratify == "strict":
                n = max(n, self.spec.target_size)
            n = min(n, len(self._buf))

        # Rank buffer by priority, take top-n
        now    = self._now_fn()
        scores = self._score_entries(self._buf, now, running_scaffolds)
        order  = sorted(range(len(self._buf)), key=lambda i: -scores[i])
        top_n  = order[:n]

        top_entries = [self._buf[i] for i in top_n]
        top_scores  = [scores[i]    for i in top_n]
        dispatched  = [e.candidate_id for e in top_entries]

        dispatched_set = set(top_n)
        self._buf = [e for i, e in enumerate(self._buf) if i not in dispatched_set]

        self._shard_seq += 1
        if self._metrics_fn is not None:
            self._metrics_fn(
                self._shard_seq,
                n,
                [e.score for e in top_entries],
                top_scores,
            )
        if self._log_fn is not None:
            cand_str = "  ".join(
                f"{e.candidate_id}(s={e.score:.3f} p={ps:.3f} sc={e.scaffold_class})"
                for e, ps in zip(top_entries, top_scores)
            )
            self._log(
                f"  Sharder [{self.name}] shard={self._shard_seq}: "
                f"dispatched {n}  profile={self.spec.profile}  "
                f"buffered={len(self._buf)} remaining\n    {cand_str}"
            )

        return dispatched
