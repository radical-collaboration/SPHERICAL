"""
DDMdWrapperWorkflow — wraps the real DDMdWorkflow from DeepDriveSim.
"""

import importlib.util
import sys
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path

import yaml


# Make DeepDriveSim importable
_DDSIM_ROOT = Path("/ocean/projects/dmr170002p/goliyad/DeepDriveSim")
if str(_DDSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDSIM_ROOT))

from src.campaign import BaseWorkflow


@lru_cache(maxsize=1)
def _get_workflow_class():
    spec = importlib.util.spec_from_file_location(
        "ddmd_workflow",
        _DDSIM_ROOT / "workflows/ddmd_workflow/ddmd_workflow.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DDMdWorkflow


# Pre-load at module import time so the lru_cache is warm before Dragon's
# worker pool starts.  Calling _get_workflow_class() inside an active Dragon
# event loop causes an import-lock deadlock: Dragon worker threads hold the
# import lock while initialising, and exec_module() blocks waiting for it.
_get_workflow_class()


class DDMdWrapperWorkflow(BaseWorkflow):
    """Async wrapper that runs one replica of the DDMd pipeline."""

    workflow_id = "ddmd"

    async def run(self, replica_id: str) -> None:
        WorkflowClass = _get_workflow_class()  # returns instantly from lru_cache

        asyncflow = self.asyncflow
        cfg = self.config or {}
        ddsim_config = cfg.get("ddsim_config")
        if not ddsim_config:
            raise ValueError(f"[{replica_id}] 'ddsim_config' missing from workflow config.")

        replica_config_path = self._make_replica_config(ddsim_config, replica_id)
        try:
            workflow = WorkflowClass(
                asyncflow=asyncflow,
                config=replica_config_path,
                name=replica_id,
                on_ready=self._on_ready,
                policies=self.policies,
                engine_dragon=self.engine_dragon,
            )
        except Exception:
            print(
                f"[{replica_id}] DDMdWorkflow.__init__ raised:\n" + traceback.format_exc(),
                flush=True,
            )
            raise

        try:
            await workflow.start()
        finally:
            Path(replica_config_path).unlink(missing_ok=True)

    @staticmethod
    def _make_replica_config(base_config_path: str, replica_id: str) -> str:
        with open(base_config_path) as f:
            cfg = yaml.safe_load(f)

        base_exp_dir = Path(cfg["experiment_directory"])
        cfg["experiment_directory"] = str(base_exp_dir.parent / f"{base_exp_dir.name}_{replica_id}")

        if cfg.get("node_local_path"):
            cfg["node_local_path"] = str(Path(cfg["node_local_path"]) / replica_id)

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            prefix=f"ddmd_{replica_id}_",
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        return tmp.name
