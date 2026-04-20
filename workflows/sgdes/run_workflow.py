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
collect_nvml_telemetry   — GPU utilisation via NVML; a NvmlMonitor is started on
                           each worker node via Dragon function tasks so all nodes
                           are captured.  Output: nvml_dir/ (default nvml-telemetry/).
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

# JAX_PLATFORMS=cpu is set in the sbatch to keep pre-Dragon setup CPU-only.
# Clear it here before SGDESWorkflow is imported so that JAX initialises on GPU.
# The DES solver (MutationPredictorSolver.propose / _update_params) runs in the
# head process and is bottlenecked by JAX on CPU (~108 s/step) vs GPU (~5 s/step).
# Dragon's function_task forks the head, but foldseek_search immediately calls
# subprocess.run() so the exec() follows the fork instantly — safe in practice.
os.environ.pop("JAX_PLATFORMS", None)

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


def make_policies(all_gpus):
    """Create one Policy per available GPU slot.

    Each policy pins a worker to a specific node (HOST_NAME) and GPU
    (gpu_affinity).  The list length equals len(all_gpus); run() enforces
    a semaphore so at most len(policies) mutations run concurrently and no
    two active mutations share the same node/GPU.

    HOST_NAME must be the actual compute node hostname (e.g. 'gpub001').
    Using 'localhost' resolves to host_id=-1 on single-node Dragon (-s)
    and causes a ~54 s scheduling timeout — always pass the real hostname
    returned by find_gpus().
    """
    from dragon.infrastructure.policy import Policy

    return [
        Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=hostname,
            gpu_affinity=[gpu_id],
        )
        for hostname, gpu_id in all_gpus
    ]


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

    # DES_WORKDIR_BASE: where _run_des creates its per-round workdir (input_db,
    # cand_db, candidate FASTAs, result TSVs).
    #
    # Single-node (SLURM_NNODES == 1): use node-local DES_TMPDIR (/tmp).
    #   foldseek createdb writes many small index files; on Lustre this costs
    #   ~130 s/step vs ~10 s/step on /tmp regardless of CPU/GPU ProstT5 mode.
    #
    # Multi-node (SLURM_NNODES > 1): use Lustre (None → falls back to
    #   os.path.dirname(des_fasta) in _run_des).  Required because
    #   foldseek_search is a function_task that may land on any node and must
    #   be able to read the DB files written by foldseek_createdb.
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    if nnodes == 1:
        config["des_workdir_base"] = os.environ.get("DES_TMPDIR", "/tmp")
    else:
        config["des_workdir_base"] = None  # Lustre fallback in _run_des
    print(f"Nodes: {nnodes}  DES workdir base: {config['des_workdir_base'] or 'Lustre (multi-node)'}")

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
        policies = make_policies(find_gpus())

        engine_dragon = await DragonExecutionBackendV3()
        asyncflow = await WorkflowEngine.create(engine_dragon)

    else:  # concurrent
        from concurrent.futures import ProcessPoolExecutor

        from radical.asyncflow import LocalExecutionBackend

        engine_concurrent = LocalExecutionBackend(executor=ProcessPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine_concurrent)
        policies = None

    wf = SGDESWorkflow(config, asyncflow=asyncflow, policies=policies)

    # NvmlMonitor is started per-node inside SGDESWorkflow._sgdes_async via
    # Dragon function tasks (start_node_telemetry / stop_node_telemetry), so
    # all worker nodes are captured automatically.  No head-node monitor needed.

    num_slots = len(policies) if policies is not None else len(wf.mutations)
    print(
        f"Starting {len(wf.mutations)} mutations "
        f"({num_slots} concurrent, {wf.foldtune_rounds} foldtune rounds)"
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
