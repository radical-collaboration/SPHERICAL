#!/usr/bin/env python3
"""
Benchmark runner — measures performance across feature-flag configurations.

Each configuration is a dict of feature overrides applied on top of the
base config.yaml.  For each configuration the campaign is run N_RUNS times
(different random seeds) and metrics are aggregated.

Results are written to benchmark_results.json for consumption by
plot_optimizations.py.

Usage:
    python benchmark.py [--config config.yaml] [--runs 3] [--out benchmark_results.json]
"""

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Benchmark configurations ──────────────────────────────────────────────────

CONFIGURATIONS: dict[str, dict] = {
    # ─── Dumb waterfall baseline ──────────────────────────────────────────────
    # True sequential pipeline: each stage starts ONLY after ALL replicas of the
    # previous stage have finished (dep_threshold_override=9999999 forces the
    # scheduler to wait for upstream.status=="done", not just the first replica).
    # Sharder OFF → FIFO arrival order (random quality).  No priorities → s1
    # monopolises all GPUs.  Underutilises resources at every stage transition.
    "baseline": {
        "features": {"backpressure": False, "monitor": False, "sharder": False, "bandit": False},
       # "dep_threshold_override": 9999999,
        "stage_replicas_overrides": {
            "s2_ml_affinity":   {"min_replicas": 2},
            "s3_docking":       {"min_replicas": 1},
            "s4_md_refinement": {"min_replicas": 1},
            "s5_fep_ranking":   {"min_replicas": 1},
        },
    },

    # ─── Smart sharding axis ─────────────────────────────────────────────────
    # Demonstrates the CANDIDATE-ROUTING + PARTIAL PIPELINE benefit.
    # Sharder ON, stratify=soft: adaptive batch dispatch ranked by score.
    # NO static stage priorities (no bandit) — but min_replicas guarantees a
    # concurrency floor for every downstream stage via Pass 1 of the scheduler.
    # This forces PARTIAL overlap between stages without requiring the bandit
    # to learn it.  Combines quality filtering with basic pipeline configuration.
    "sharding+bp": {
        "features": {"backpressure": True, "monitor": False, "sharder": True, "bandit": False},
        "sharding_overrides": {"stratify": "soft", "use_bandit": True,
                               "min_size": 1, "target_size": 8, "max_size": 32},
        # min_replicas guarantees Pass-1 concurrency floor for downstream stages:
        # scheduler always reserves this many GPU slots even while s1 is running.
        "stage_replicas_overrides": {
            "s2_ml_affinity":   {"min_replicas": 2},
            "s3_docking":       {"min_replicas": 1},
            "s4_md_refinement": {"min_replicas": 1},
            "s5_fep_ranking":   {"min_replicas": 1},
        },
    },

    # ─── Scheduling bandit axis ───────────────────────────────────────────────
    # Demonstrates the GPU-ALLOCATION benefit in isolation.
    # No sharding: FIFO dispatch (sharder=False), random quality order.
    # Cross-stage Thompson-sampling bandit LEARNS downstream-first GPU allocation.
    # No static priorities (bandit must learn them).  No BP.
    "scheduling_bandit": {
        "features": {"backpressure": False, "monitor": False, "sharder": False, "bandit": True},
    },

    # ─── Combined ─────────────────────────────────────────────────────────────
    # Both axes together: adaptive soft sharding (stratify=soft + shard_bandit +
    # BP) AND cross-stage scheduling bandit.
    # NO static priorities — the cross-stage bandit learns optimal GPU allocation
    # via Thompson sampling with depth-based warm-start priors (s5=Beta(5,1),
    # s1=Beta(1,1)).  Static priorities would pre-answer what the bandit is
    # supposed to discover, hiding whether it adds value beyond the static ordering.
    "all_optimizations": {
        "features": {"backpressure": True, "monitor": False, "sharder": True, "bandit": True},
        "sharding_overrides": {"stratify": "soft", "use_bandit": True,
                               "min_size": 1, "target_size": 8, "max_size": 32},
    },
}


def _apply_config_override(base: dict, override: dict) -> dict:
    """Deep-merge override into a copy of base config."""
    cfg = copy.deepcopy(base)
    # Feature flags
    if "features" in override:
        cfg.setdefault("cm", {}).setdefault("features", {}).update(override["features"])
        cfg.setdefault("features", {}).update(override["features"])
    # Sharding overrides: applied to all stages that have a sharding block
    if "sharding_overrides" in override:
        sh_ov = override["sharding_overrides"]
        for stage in cfg.get("stages", []):
            if "sharding" in stage:
                stage["sharding"].update(sh_ov)
    # stage_replicas_overrides: per-stage min_replicas / max_replicas overrides.
    # Used to guarantee a minimum concurrency floor for downstream stages even without
    # a scheduling bandit — Pass 1 of the scheduler ensures min_replicas is always
    # satisfied first, forcing some GPU sharing across stages.
    if "stage_replicas_overrides" in override:
        for stage in cfg.get("stages", []):
            sid = stage["id"]
            if sid in override["stage_replicas_overrides"]:
                stage.update(override["stage_replicas_overrides"][sid])

    # dep_threshold_override: sets dependency_threshold for every DEPENDENT stage to a
    # very large value so it only becomes eligible when upstream.status == "done" —
    # not after the first upstream replica finishes.  Creates a true sequential
    # waterfall: stage N+1 waits for ALL of stage N to complete before starting.
    if "dep_threshold_override" in override:
        dt = int(override["dep_threshold_override"])
        stage_ids_set = {s["id"] for s in cfg.get("stages", [])}
        for stage in cfg.get("stages", []):
            if stage.get("upstream", "") in stage_ids_set:
                stage["dependency_threshold"] = dt

    # Stage priority overrides: sets scheduler priority per stage (higher = scheduled first).
    # Used to give downstream stages static priority without a scheduling bandit.
    if "stage_priority_overrides" in override:
        pri_ov = override["stage_priority_overrides"]
        for stage in cfg.get("stages", []):
            if stage["id"] in pri_ov:
                stage["priority"] = pri_ov[stage["id"]]
    # Dreamer overrides: applied to all stages' dreamer block (flat key update).
    # Used to set trigger_mode and other dreamer simulation parameters.
    if "dreamer_overrides" in override:
        dr_ov = override["dreamer_overrides"]
        for stage in cfg.get("stages", []):
            stage.setdefault("dreamer", {}).update(dr_ov)
    return cfg


async def _run_once(config: dict, seed_offset: int) -> dict:
    """Run one campaign with the given config and return its metrics dict."""
    import random as _random
    # Fix the global random state so score-cascade outcomes are identical across
    # configs within the same run index.  Without this, sequential config runs
    # consume different random numbers from a shared state, making comparisons
    # unfair (different random realizations of the score cascade).
    _random.seed(seed_offset + 1337)

    from src.campaign import AsyncCampaignManager as CampaignManager
    from src.inference.utils import load_config
    import importlib

    # Reset DreamerWorkflow class-level state so trigger counts don't bleed
    # across benchmark runs (class vars persist for the lifetime of the process).
    sys.path.insert(0, str(Path(__file__).parent))
    from dreamer_workflow import DreamerWorkflow
    DreamerWorkflow._group_state = {}
    DreamerWorkflow._trigger_lock = None

    # Translate plan format
    if "stages" in config:
        from run_campaign import _build_from_plan, _build_registry
        cm_cfg = config.get("cm", {})
        config["workflows"] = _build_from_plan(config)
        for key in ("engine", "resources", "telemetry", "workflow_registry", "features"):
            if key in cm_cfg and key not in config:
                config[key] = cm_cfg[key]
        config["debug"] = bool(cm_cfg.get("debug", False))

    # Bump seeds for reproducible variance across runs
    if "provenance" in config:
        for k in config["provenance"].get("seeds", {}):
            config["provenance"]["seeds"][k] += seed_offset

    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend
    backend   = await ConcurrentExecutionBackend()
    asyncflow = await WorkflowEngine.create(backend)

    registry = _build_registry(config)
    cm = CampaignManager.from_config(config, registry, asyncflow=asyncflow)

    # Hard timeout: guards against any stall in cm.wait().
    # All stages GPU-bound; baseline ~1200s/run.  Optimised runs can take
    # longer during bandit warm-up (before it learns to keep s2 running).
    RUN_TIMEOUT_S = 5400

    try:
        await cm.start()
        finished = await cm.wait(timeout=RUN_TIMEOUT_S)
        if not finished:
            raise TimeoutError(
                f"Campaign did not finish within {RUN_TIMEOUT_S}s "
                f"(likely a runaway dreamer replica)"
            )
    finally:
        await cm.close()
        await asyncflow.shutdown()

    m = cm.metrics().to_dict()

    # ── Time-to-target: seconds until the Nth terminal-stage replica finishes ──
    # Extracted from replica_events so plot_optimizations can draw the step curve.
    _TARGET_STAGE = "s5_fep_ranking"
    _TARGET_N     = 5
    s5_finishes = sorted(
        e["t"] for e in m.get("replica_events", [])
        if e["group"] == _TARGET_STAGE and e["event"] == "finish"
    )
    m["time_to_target_s"] = s5_finishes[_TARGET_N - 1] if len(s5_finishes) >= _TARGET_N else None
    return m


async def run_benchmark(
    config_path: str,
    n_runs: int,
    out_path: str,
) -> None:
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    results: dict = {}
    for cfg_name, override in CONFIGURATIONS.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg_name}")
        print(f"{'='*60}")
        cfg_results = []
        for run_idx in range(n_runs):
            print(f"  Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)
            cfg = _apply_config_override(base_config, override)
            t0 = time.time()
            try:
                metrics = await _run_once(cfg, seed_offset=run_idx * 100)
                elapsed = time.time() - t0
                print(f"done in {elapsed:.1f}s  (campaign wall_time={metrics['wall_time_s']:.1f}s)")
                cfg_results.append(metrics)
            except Exception as exc:
                print(f"FAILED: {exc}")
                cfg_results.append({"error": str(exc), "wall_time_s": None})
        results[cfg_name] = cfg_results

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--out",    default="benchmark_results.json")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.config, args.runs, args.out))

#python benchmark.py --runs 3 --out benchmark_results.json