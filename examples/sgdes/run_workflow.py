#!/usr/bin/env python3
"""
SGDES workflow runner for MAYV mutations.

Usage
-----
    # Single-node interactive (Dragon):
    dragon -s run_workflow.py [--config config.yaml]

    # Multi-node via SLURM (Dragon):
    sbatch delta_gpu_sbatch.sh

    # Local asyncio (no Dragon, no GPU affinity):
    python run_workflow.py --config config.yaml   # engine: concurrent in config

Execution backends (set via `engine:` in config.yaml)
------------------------------------------------------
dragon     — DragonExecutionBackendV3; tasks are distributed across nodes using
             per-mutation Policy objects (HOST_NAME + gpu_affinity).  Each mutation
             runs on its own dedicated node/GPU; tasks within a mutation are
             sequential so at most len(mutations) workers run at once.  Capped at
             total_gpus to avoid the Dragon GS bottleneck.
concurrent — LocalExecutionBackend backed by ProcessPoolExecutor; no GPU affinity,
             useful for quick testing without Dragon.

Telemetry (set via config.yaml)
--------------------------------
collect_dragon_telemetry — Dragon runtime metrics via DragonTelemetryCollector;
                           only active when engine=dragon.  Output: dragon_telemetry_dir/
                           (default dragon-telemetry/).

Required environment variables (set in *_gpu_sbatch.sh)
--------------------------------------------------------
SPHERICAL_DIR — root of the SPHERICAL repo (used to extend sys.path)
SGDES_DIR     — root of the SGDES/TRILL fork (used inside the workflow)
CUDA_HOME     — CUDA toolkit root; lib64/ is added to LD_LIBRARY_PATH so that
                foldseek's ggml-CUDA backend can find libcudart on every node.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from radical.asyncflow import WorkflowEngine

_SPHERICAL_ROOT = Path(os.environ.get("SPHERICAL_DIR", Path(__file__).resolve().parents[2]))
if str(_SPHERICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPHERICAL_ROOT))

from sgdes_workflow_asyncflow import SGDESWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_gpus():
    """Return [(hostname, gpu_id), ...] for every GPU visible to Dragon.

    Uses Dragon's native machine API to enumerate all nodes and their GPUs.
    node.gpus may be None on CPU-only nodes (e.g. login nodes), so we guard
    with `or []` to skip them safely.

    Under `dragon -s` (single-node mode) node.hostname returns 'localhost',
    which resolves to host_id=-1 and causes a ~54 s scheduling timeout per
    task.  We substitute the real hostname in that case.
    """
    import socket
    from dragon.native.machine import Node, System

    real_hostname = socket.gethostname()
    all_gpus = []
    for huid in System().nodes:
        node = Node(huid)
        hostname = node.hostname if node.hostname != "localhost" else real_hostname
        for gpu_id in node.gpus or []:
            all_gpus.append((hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=32):
    """Create one Policy per mutation slot with round-robin GPU assignment.

    Each policy pins a worker to a specific node (HOST_NAME) and GPU
    (gpu_affinity) so that Dragon routes each mutation to the correct
    node/GPU in multi-node runs.

    HOST_NAME must be the actual compute node hostname (e.g. 'gpub001').
    Using 'localhost' resolves to host_id=-1 on single-node Dragon (-s)
    and causes a ~54 s scheduling timeout — always pass the real hostname
    returned by find_gpus().
    """
    from dragon.infrastructure.policy import Policy

    policies = []
    i = 0
    for _ in range(nprocs):
        hostname, gpu_id = all_gpus[i]
        policies.append(
            Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=hostname,
                gpu_affinity=[gpu_id],
            )
        )
        i = (i + 1) % len(all_gpus)
    return policies


async def main(config_file: str) -> None:
    config = load_config(config_file)

    backend = config.get("engine", "concurrent")
    print(f"Using execution backend: {backend}")

    # TOTAL_GPUS is set by delta_gpu_sbatch.sh as SLURM_NNODES * SLURM_GPUS_PER_NODE.
    # It is not in config.yaml to avoid confusion — always comes from the sbatch env.
    if "TOTAL_GPUS" not in os.environ:
        raise RuntimeError(
            "TOTAL_GPUS env var not set. "
            "Run via delta_gpu_sbatch.sh or set: export TOTAL_GPUS=<nodes x gpus-per-node>"
        )
    config["total_gpus"] = int(os.environ["TOTAL_GPUS"])
    print(f"Total GPUs: {config['total_gpus']}")

    # ── Dragon telemetry (dragon engine only) ─────────────────────────────────
    dragon_collector = None
    if backend == "dragon" and config.get("collect_dragon_telemetry", False):
        from src.inference.utils import init_collector

        dragon_telemetry_dir = config.get("dragon_telemetry_dir", "dragon-telemetry")
        dragon_collector = init_collector(dragon_telemetry_dir)
        if dragon_collector:
            dragon_collector.start()
            print(f"DragonTelemetryCollector started → {dragon_telemetry_dir}")

    if backend == "dragon":
        import multiprocessing as mp

        mp.set_start_method("dragon")
        from rhapsody.backends import DragonExecutionBackendV3

        num_mutations = len(list(config["mutations"]))
        total_gpus = int(config.get("total_gpus", 1))
        policies = make_policies(find_gpus(), nprocs=num_mutations)

        engine_dragon = await DragonExecutionBackendV3()
        asyncflow = await WorkflowEngine.create(engine_dragon)

    else:  # concurrent
        from concurrent.futures import ProcessPoolExecutor

        from radical.asyncflow import LocalExecutionBackend

        engine_concurrent = LocalExecutionBackend(executor=ProcessPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine_concurrent)
        policies = None

    wf = SGDESWorkflow(config, asyncflow=asyncflow, policies=policies)

    print(
        f"Starting {len(wf.mutations)} mutations "
        f"({wf.total_gpus} concurrent, {wf.foldtune_rounds} foldtune rounds)"
    )
    try:
        await wf.run()
        print("Done.")
    finally:
        await asyncflow.shutdown()
        if dragon_collector:
            dragon_collector.stop()
            print("DragonTelemetryCollector stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGDES MAYV runner")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    asyncio.run(main(args.config))
