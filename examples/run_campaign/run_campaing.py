#!/usr/bin/env python3
# Limit OpenBLAS/OMP threads before any numpy import to avoid pthread_create
# failures on login nodes where process counts are restricted.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

"""
Campaign runner — starts async workflow replicas with optional dependency
ordering between workflow groups.

Usage
-----
    python run_campaing.py --config config.yaml

Config file structure
---------------------
    # ── Per-workflow sections ─────────────────────────────────────────────
    workflows:
      ddsim:
        replicas:     8
        min_replicas: 2
        max_replicas: 4
        dependencies: []
        ddsim_config: "/path/to/ddmd_config.yaml"

      inference:
        replicas:     1
        dependencies: [ddsim]  # starts only after all ddsim replicas finish
        num_gpus_per_service: 4

If no config file is provided, hard-coded defaults are used for local testing.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.campaign import AsyncCampaignManager
from src.inference.utils import load_config, init_collector
from src.utils.nvml_monitor import NvmlMonitor


def _expand_workflow_configs(config: dict, config_dir: Path) -> dict:
    """
    For each workflow entry that has a ``config_file`` key, load that YAML and
    merge its contents into the workflow config dict.  The per-workflow file
    provides workflow-specific parameters; any keys already present in the
    workflow entry (scheduling params) take precedence.
    """
    for wf_cfg in config.get("workflows", {}).values():
        cfg_file = wf_cfg.pop("config_file", None)
        if not cfg_file:
            continue
        cfg_path = config_dir / cfg_file
        with open(cfg_path) as f:
            wf_specific = yaml.safe_load(f) or {}
        # Scheduling params in config.yaml win; workflow file fills the rest.
        wf_specific.update(wf_cfg)
        wf_cfg.clear()
        wf_cfg.update(wf_specific)
    return config

from ddsim_workflow import DDSimWorkflow
from ddmd_workflow import DDMdWrapperWorkflow
from inference_workflow import InferenceWorkflow
from miniapps_workflow import MiniAppsWrapperWorkflow

# Maps config workflow names → workflow classes
WORKFLOW_REGISTRY = {
    "dummy":     DDSimWorkflow,
    "md":        DDMdWrapperWorkflow,
    "miniapps":  MiniAppsWrapperWorkflow,
    "inference": InferenceWorkflow,
}

_DEFAULT_CONFIG_FILE = Path(__file__).parent / "config.yaml"

DEFAULT_CONFIG = {
    "workflows": {
        "ddsim": {
            "replicas":               2,
            "dependencies":           [],
            "engine":                 "concurrent",
            "home_dir":               str(Path.home() / "DDSim"),
            "num_inputs":             5,
            "max_sim_batch":          4,
            "training_cores":         1,
            "training_threshold":     0.5,
            "prediction_threshold":   0.5,
            "start_training_threshold": 1,
            "training_epochs":        1,
            "free_resources_for_train": True,
            "sleep_time":             30,
            "ddsim_config": str(
                Path("/ocean/projects/dmr170002p/goliyad/DeepDriveSim")
                / "workflows/ddmd_workflow/data/new_lassen-keras-dbscan.yaml"
            ),
        },
        "inference": {"replicas": 1, "dependencies": ["ddsim"], "dependency_threshold": 1},
    },
}


async def main(config_file: Optional[str]) -> None:
    if config_file is None and _DEFAULT_CONFIG_FILE.exists():
        config_file = str(_DEFAULT_CONFIG_FILE)
    config = load_config(config_file) if config_file else DEFAULT_CONFIG
    config_dir = Path(config_file).parent if config_file else Path(__file__).parent
    _expand_workflow_configs(config, config_dir)

    # ── Telemetry collector (Dragon only; no-op when Dragon not active) ────
    tel_cfg = config.get("telemetry", {})
    collector = None
    if tel_cfg.get("collect_telemetry", False):
        telemetry_dir = tel_cfg.get("telemetry_dir", "data/telemetry")
        collector = init_collector(telemetry_dir)
        if collector:
            collector.start()
            print(f"DragonTelemetryCollector started → {telemetry_dir}")

    # ── NVML monitor (runs in main process; provides ground-truth GPU util) ─
    nvml_dir = tel_cfg.get("nvml_telemetry_dir", "data/nvml-telemetry")
    nvml_rate = float(tel_cfg.get("nvml_collection_rate", 1.0))
    nvml_checkpoint = float(tel_cfg.get("nvml_checkpoint_interval", 30.0))
    nvml_monitor = NvmlMonitor(
        output_dir=nvml_dir,
        collection_rate=nvml_rate,
        checkpoint_interval=nvml_checkpoint,
    )
    nvml_monitor.start()

    cm = AsyncCampaignManager.from_config(config, WORKFLOW_REGISTRY)

    groups = config.get("workflows", {})
    print(
        "Campaign: "
        + ", ".join(
            f"{name}: {cfg.get('replicas', 1)} replica(s) "
            f"deps={cfg.get('dependencies', [])}"
            for name, cfg in groups.items()
        )
    )

    try:
        await cm.start()   # launch groups with no unmet dependencies
        await cm.wait()    # block until all groups (including dependents) finish
    finally:
        if collector:
            collector.stop()
            print("DragonTelemetryCollector stopped")
        nvml_monitor.stop()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n── Campaign complete ──")
    final_status = cm.status()
    for name, info in final_status["groups"].items():
        print(
            f"  {name}: status={info['status']}  "
            f"replicas={info['replicas_finished']}/{info['replicas_total']}"
        )

    print("\n── Replica counts per workflow ──")
    for name, s in cm.stats().items():
        print(
            f"  {name}: "
            f"replicas_started={s.replicas_started}  "
            f"replicas_finished={s.replicas_finished}"
        )

    await cm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPHERICAL campaign runner")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (defaults to config.yaml next to this script)",
    )

    args = parser.parse_args()
    asyncio.run(main(args.config))
