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

from radical.asyncflow import WorkflowEngine

_SPHERICAL_ROOT = Path(os.environ.get("SPHERICAL_DIR", Path(__file__).resolve().parents[2]))
if str(_SPHERICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPHERICAL_ROOT))

from sgdes_workflow import SGDESWorkflow  # noqa: E402

from src.utils.workflow import find_gpus, load_config, make_policies  # noqa: E402


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
    print(
        f"Nodes: {nnodes}  DES workdir base: {config['des_workdir_base'] or 'Lustre (multi-node)'}"
    )

    if backend == "dragon":
        import multiprocessing as mp

        mp.set_start_method("dragon")
        from rhapsody.backends import DragonExecutionBackendV3

        num_mutations = len(list(config["mutations"]))
        policies = make_policies(find_gpus(), nprocs=num_mutations)

        engine_dragon = await DragonExecutionBackendV3()
        asyncflow = await WorkflowEngine.create(engine_dragon)

    else:  # concurrent
        from concurrent.futures import ProcessPoolExecutor

        from rhapsody.backends import ConcurrentExecutionBackend

        engine_concurrent = ConcurrentExecutionBackend(executor=ProcessPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine_concurrent)
        policies = None

    telemetry = None
    if config.get("collect_telemetry", False):
        telemetry_dir = config.get("telemetry_dir", "telemetry_output")
        if hasattr(asyncflow, "start_telemetry"):
            telemetry = await asyncflow.start_telemetry(
                resource_poll_interval=0.5,
                checkpoint_path=telemetry_dir,
            )
            print(f"Started Asyncflow telemetry → {telemetry_dir}")

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
        await wf.run()
        print("Done.")
    finally:
        await asyncflow.shutdown()
        if telemetry:
            await telemetry.stop()
            print("Asyncflow telemetry stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGDES MAYV runner")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    asyncio.run(main(args.config))
