#!/usr/bin/env python3
"""
SGDES runner for MAYV mutations.

Usage
-----
    python run_with_cm.py [--config config.yaml]

Runs all mutations listed in config.yaml via SGDESWorkflow.
Up to total_gpus mutations run concurrently; each calls `trill workflow sgdes`.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from radical.asyncflow import WorkflowEngine
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node, System
from src.utils.nvml_monitor import NvmlMonitor

import yaml

_SPHERICAL_ROOT = Path("/ocean/projects/dmr170002p/goliyad/htp/SPHERICAL")
if str(_SPHERICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPHERICAL_ROOT))

from sgdes_workflow_asyncflow import SGDESWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def find_gpus():

    all_gpus = []
    # loop through all nodes Dragon is running on
    for huid in System().nodes:
        node = Node(huid)
        # loop through however many GPUs it may have
        for gpu_id in node.gpus:
            all_gpus.append((node.hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=32):
    """Create per-process policies with round-robin GPU assignment."""
    policies = []
    i = 0
    for _worker in range(nprocs):
        policies.append(
            Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=all_gpus[i][0],
                gpu_affinity=[all_gpus[i][1]],
            )
        )
        i += 1
        if i == len(all_gpus):
            i = 0
    return policies

async def main(config_file: str) -> None:
    config = load_config(config_file)

    backend = config.get("engine", "concurrent") 

    if backend == "dragon":
        from rhapsody.backends import DragonExecutionBackendV3
        engine_dragon = await DragonExecutionBackendV3(num_workers=16)
        asyncflow = await WorkflowEngine.create(engine_dragon)
        policies = make_policies(find_gpus(), nprocs=len(list(config["mutations"])))
    else:  # concurrent
        #from rhapsody.backends import ConcurrentExecutionBackend
        engine_dragon = None
        #engine_concurrent = await ConcurrentExecutionBackend()
        from concurrent.futures import ThreadPoolExecutor
        from radical.asyncflow import LocalExecutionBackend
        engine_concurrent = LocalExecutionBackend(executor=ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine_concurrent)

    wf = SGDESWorkflow(config, asyncflow=asyncflow)

    collect_telemetry = config.get("collect_telemetry", True)
    if collect_telemetry:
        nvml_dir = config.get("nvml_telemetry_dir", "data/nvml-telemetry")
        nvml_rate = float(config.get("nvml_collection_rate", 1.0))
        nvml_checkpoint = float(config.get("nvml_checkpoint_interval", 30.0))

        nvml_dir = 'nvml-telemetry'
        nvml_monitor = NvmlMonitor(
            output_dir=nvml_dir,
            collection_rate=nvml_rate,
            checkpoint_interval=nvml_checkpoint,
        )
        nvml_monitor.start()

    print(
        f"Starting {len(wf.mutations)} mutations "
        f"({wf.total_gpus} concurrent, {wf.foldtune_rounds} foldtune rounds)"
    )
    try:
        await wf.run()
        print("Done.")
    finally:
        if collect_telemetry:
            nvml_monitor.stop()
        await asyncflow.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGDES MAYV runner")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
