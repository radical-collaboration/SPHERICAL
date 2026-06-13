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
        replicas:          8
        concurrency_floor: 2
        concurrency_cap:   4
        dependencies:      []
        ddsim_config:      "/path/to/ddmd_config.yaml"

      inference:
        replicas:     1
        dependencies: [ddsim]  # starts only after all ddsim replicas finish
        num_gpus_per_service: 4

If no config file is provided, hard-coded defaults are used for local testing.
"""

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.campaign import AsyncCampaignManager as CampaignManager  # noqa: E402
from src.inference.utils import load_config  # noqa: E402
from src.utils.workflow import _expand_env  # noqa: E402


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
        cfg_path = Path(os.path.expandvars(cfg_file))
        if not cfg_path.is_absolute():
            cfg_path = config_dir / cfg_path
        with open(cfg_path) as f:
            wf_specific = _expand_env(yaml.safe_load(f) or {})
        # Scheduling params in config.yaml win; workflow file fills the rest.
        wf_specific.update(wf_cfg)
        wf_cfg.clear()
        wf_cfg.update(wf_specific)
    return config


def _build_registry(config: dict) -> dict:
    """Dynamically import workflow classes from the 'workflow_registry' config section."""
    import importlib

    registry = {}
    for name, cls_path in config.get("workflow_registry", {}).items():
        module_name, cls_name = cls_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        registry[name] = getattr(module, cls_name)
    return registry


async def main(config_file: str) -> None:
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    config = load_config(config_file)
    config_dir = config_path.parent
    _expand_workflow_configs(config, config_dir)

    engine_type = config.get("engine", "dragon")

    # ── Build backend and asyncflow (mirrors workflow run_workflow.py pattern) ─
    engine_dragon = None
    asyncflow = None

    if engine_type == "dragon":
        try:
            from radical.asyncflow import WorkflowEngine
            from rhapsody.backends import DragonExecutionBackendV3

            engine_dragon = await DragonExecutionBackendV3()
            asyncflow = await WorkflowEngine.create(engine_dragon)
            print("Dragon backend started")
        except ImportError:
            engine_type = "concurrent"

    else:
        from radical.asyncflow import WorkflowEngine
        from rhapsody.backends import ConcurrentExecutionBackend

        backend = await ConcurrentExecutionBackend()
        asyncflow = await WorkflowEngine.create(backend)
        print("ConcurrentExecutionBackend started")

    # ── Telemetry ─────────────────────────────────────────────────────────────
    tel_cfg = config.get("telemetry", {})
    telemetry = None
    if tel_cfg.get("collect_telemetry", False):
        telemetry_dir = tel_cfg.get("telemetry_dir", "data/telemetry-results")
        if hasattr(asyncflow, "start_telemetry"):
            telemetry = await asyncflow.start_telemetry(
                resource_poll_interval=0.5,
                checkpoint_path=telemetry_dir,
            )
            print(f"Started Asyncflow telemetry → {telemetry_dir}")

    # ── Campaign ──────────────────────────────────────────────────────────────
    registry = _build_registry(config)
    cm = CampaignManager.from_config(
        config,
        registry,
        asyncflow=asyncflow,
        engine_dragon=engine_dragon,
    )

    groups = config.get("workflows", {})
    print(
        "Campaign: "
        + ", ".join(
            f"{name}: {cfg.get('replicas', 1)} replica(s) deps={cfg.get('dependencies', [])}"
            for name, cfg in groups.items()
        )
    )

    try:
        await cm.start()  # launch groups with no unmet dependencies
        await cm.wait()  # block until all groups (including dependents) finish
    finally:
        await cm.close()

        if telemetry:
            await telemetry.stop()
            print("Asyncflow telemetry stopped")

        await asyncflow.shutdown()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPHERICAL campaign runner")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )

    args = parser.parse_args()
    asyncio.run(main(args.config))
