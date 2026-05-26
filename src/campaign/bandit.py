"""
Thompson-sampling multi-armed bandit for campaign optimization.

Two planned use cases (integration wired later via feature flag):

  Shard optimizer
    Arms:    multiplier factors applied to ShardingSpec.target_size
             e.g. [0.50, 0.75, 1.00, 1.25, 1.50]
    Reward:  throughput — replicas dispatched per second after the shard;
             normalized to [0, 1] relative to a rolling max

  Resource optimizer
    Arms:    per-stage budget reallocation factors
             e.g. {"s1": 0.9, "s2": 1.1, ...} encoded as discrete options
    Reward:  pass-through efficiency — observed / expected trigger fraction;
             clamped to [0, 1]

Algorithm
---------
Beta-Bernoulli Thompson sampling with continuous reward:

  Prior:  Beta(α=1, β=1)  — uniform, no preference
  Update: α += reward      (reward ∈ [0, 1])
          β += 1 - reward
  Select: sample each arm from Beta(α, β);
          choose arm with the highest sample

The continuous update degrades gracefully: reward=1.0 is a pure success,
reward=0.0 is a pure failure, values in between are fractional credit.

Usage
-----
    # Create a bandit with discrete multiplier arms
    b = Bandit(arms=[0.5, 0.75, 1.0, 1.25, 1.5], seed=42)

    # At each decision point, select an arm
    factor = b.select()

    # After observing the outcome, update with a normalized reward
    b.update(factor, reward=0.8)

    # Inspect current estimates
    print(b.summary())   # {0.5: 0.52, 0.75: 0.61, 1.0: 0.78, ...}
    print(b.best())      # 1.0  (arm with highest mean so far)
"""

import random
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BanditArm:
    """One arm of a Beta-Bernoulli bandit."""

    label:  Any
    alpha:  float = 1.0   # successes + prior
    beta:   float = 1.0   # failures  + prior

    def sample(self, rng: random.Random) -> float:
        """Draw a Thompson sample from Beta(alpha, beta)."""
        return rng.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """Update with *reward* ∈ [0, 1].  Values outside are clamped."""
        reward = max(0.0, min(1.0, reward))
        self.alpha += reward
        self.beta  += 1.0 - reward

    def reset(self) -> None:
        """Return to uninformative uniform prior."""
        self.alpha = 1.0
        self.beta  = 1.0

    @property
    def mean(self) -> float:
        """Current posterior mean estimate."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def pulls(self) -> int:
        """Effective number of updates (alpha + beta - 2 initial prior units)."""
        return max(0, round(self.alpha + self.beta - 2))

    def __repr__(self) -> str:
        return (
            f"BanditArm({self.label!r}  "
            f"mean={self.mean:.3f}  pulls={self.pulls}  "
            f"α={self.alpha:.2f}  β={self.beta:.2f})"
        )


class Bandit:
    """Multi-armed bandit with Thompson sampling over a fixed discrete action set.

    Parameters
    ----------
    arms:
        Iterable of labels (any hashable value) representing the discrete actions.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, arms: list[Any], seed: Optional[int] = None) -> None:
        if not arms:
            raise ValueError("Bandit requires at least one arm")
        self._rng  = random.Random(seed)
        self._arms: dict[Any, BanditArm] = {
            label: BanditArm(label=label) for label in arms
        }

    # ── Decision ──────────────────────────────────────────────────────────────

    def select(self) -> Any:
        """Thompson sampling: return the label of the arm with the highest sample."""
        return max(self._arms.values(), key=lambda a: a.sample(self._rng)).label

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self, arm_label: Any, reward: float) -> None:
        """Update *arm_label* with *reward* ∈ [0, 1].

        Unknown labels are silently ignored so callers don't need to guard.
        """
        arm = self._arms.get(arm_label)
        if arm is not None:
            arm.update(reward)

    def reset(self, arm_label: Optional[Any] = None) -> None:
        """Reset one arm (or all arms if *arm_label* is None) to the uniform prior."""
        targets = [self._arms[arm_label]] if arm_label is not None else self._arms.values()
        for arm in targets:
            arm.reset()

    # ── Inspection ────────────────────────────────────────────────────────────

    def best(self) -> Any:
        """Return the label of the arm with the highest posterior mean."""
        return max(self._arms.values(), key=lambda a: a.mean).label

    def summary(self) -> dict[Any, float]:
        """Posterior mean estimate for each arm — useful for logging."""
        return {label: arm.mean for label, arm in self._arms.items()}

    def arms(self) -> list[BanditArm]:
        """All arms, sorted by label (for deterministic logging)."""
        try:
            return sorted(self._arms.values(), key=lambda a: a.label)
        except TypeError:
            return list(self._arms.values())

    def __repr__(self) -> str:
        arm_str = "  ".join(repr(a) for a in self.arms())
        return f"Bandit(best={self.best()!r}  [{arm_str}])"


# ── Preconfigured factory functions ───────────────────────────────────────────

def shard_bandit(seed: Optional[int] = None) -> Bandit:
    """Bandit for shard-size multiplier selection.

    Arms represent scale factors applied to ShardingSpec.target_size.
    Replaces the hardcoded 0.5× / 1.5× multipliers in Sharder._adaptive_size().
    """
    return Bandit(arms=[0.50, 0.75, 1.00, 1.25, 1.50], seed=seed)


def resource_bandit(stage_ids: list[str], seed: Optional[int] = None) -> Bandit:
    """Bandit for per-stage budget reallocation.

    Each arm is a tuple of (stage_id, factor) pairs encoded as a frozenset,
    representing a candidate budget allocation across stages.
    In practice the caller constructs the arms based on the plan's
    per_stage_band_pct constraints.
    """
    return Bandit(arms=stage_ids, seed=seed)


class SchedulingBandit:
    """Thompson-sampling bandit for cross-stage scheduling priority.

    One BanditArm per pipeline stage. When multiple stages are eligible
    simultaneously, rank() returns them sorted by Thompson-sampled Beta value
    so the scheduler tries the most-promising stage first.

    Reward signal (fed at replica completion via update()):
      WIDEN    → 0.8  downstream hungry — this stage's output is needed, keep going
      HOLD     → 0.7  balanced — good scheduling rate
      THROTTLE → 0.2  downstream flooded — back off this stage
      none     → 0.5  terminal stage or no BP tracking — neutral

    stage_priors: optional per-stage (alpha, beta) warm-start values.  Use to
    give CPU-only source stages a head start so they are not starved during the
    cold-start window before the bandit has accumulated enough observations.
    Example: {"s1_ligand_filter": (2.0, 1.0)} → initial mean 0.67 vs 0.5 default.
    """

    def __init__(
        self,
        stage_names: list[str],
        seed: Optional[int] = None,
        stage_priors: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        self._arms: dict[str, BanditArm] = {}
        for n in stage_names:
            arm = BanditArm(label=n)
            if stage_priors and n in stage_priors:
                arm.alpha, arm.beta = stage_priors[n]
            self._arms[n] = arm
        self._rng = random.Random(seed)

    def rank(self, eligible: list) -> list:
        """Return eligible groups sorted by Thompson-sampled priority (highest first).

        Groups not in the bandit's arm set (e.g. added dynamically) fall back
        to a neutral 0.5 sample so they're still scheduled fairly.
        """
        if len(eligible) <= 1:
            return eligible
        return sorted(
            eligible,
            key=lambda g: (
                self._arms[g.name].sample(self._rng)
                if g.name in self._arms else 0.5
            ),
            reverse=True,
        )

    def update(self, stage_name: str, reward: float) -> None:
        """Update the arm for *stage_name* with *reward* ∈ [0, 1]."""
        arm = self._arms.get(stage_name)
        if arm is not None:
            arm.update(reward)

    def summary(self) -> dict[str, float]:
        """Posterior mean per stage — for logging."""
        return {name: arm.mean for name, arm in self._arms.items()}

    def best(self) -> Optional[str]:
        """Stage name with highest posterior mean."""
        if not self._arms:
            return None
        return max(self._arms, key=lambda n: self._arms[n].mean)

    def __repr__(self) -> str:
        arms_str = "  ".join(
            f"{n}:{arm.mean:.3f}" for n, arm in self._arms.items()
        )
        return f"SchedulingBandit(best={self.best()!r}  [{arms_str}])"


def scheduling_bandit(
    stage_names: list[str],
    seed: Optional[int] = None,
    stage_priors: Optional[dict[str, tuple[float, float]]] = None,
) -> SchedulingBandit:
    """Factory for the cross-stage scheduling bandit."""
    return SchedulingBandit(stage_names=stage_names, seed=seed, stage_priors=stage_priors)
