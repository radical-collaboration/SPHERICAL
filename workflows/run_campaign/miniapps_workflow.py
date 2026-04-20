"""
MiniAppsWorkflow — wraps MiniAppsWorkflow from DeepDriveSim.
"""

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

# Make DeepDriveSim importable
_DDSIM_ROOT = Path("/ocean/projects/dmr170002p/goliyad/DeepDriveSim")
if str(_DDSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDSIM_ROOT))

from src.campaign import BaseWorkflow


@lru_cache(maxsize=1)
def _get_workflow_class():
    spec = importlib.util.spec_from_file_location(
        "miniapps_workflow_asyncflow",
        _DDSIM_ROOT / "workflows/miniapps_workflow/miniapps_workflow_asyncflow.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MiniAppsWorkflowAsyncflow


# Pre-load at module import time to warm the cache before Dragon workers start.
_get_workflow_class()


class MiniAppsWrapperWorkflow(BaseWorkflow):
    """Async wrapper that runs one replica of the MiniApps pipeline."""

    workflow_id = "miniapps"

    async def run(self, replica_id: str) -> None:
        # Workaround: concurrent asyncflow backend does not apply process_template.env
        # to subprocess children, so CUDA_VISIBLE_DEVICES must be set in os.environ
        # before any subprocesses are spawned by the workflow.
        if self.policies:
            gpu_id = str(self.policies[0].gpu_affinity[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        elif self.config and self.config.get("assigned_gpu_ids"):
            gpu_id = str(self.config["assigned_gpu_ids"][0])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        WorkflowClass = _get_workflow_class()
        asyncflow = self.asyncflow
        cfg = self.config or {}
        base_home = Path(cfg.get("home_dir", Path.home() / "MiniApps")).expanduser()
        replica_home = base_home / replica_id

        if not cfg.get("src_dir"):
            cfg = {**cfg, "src_dir": str(_DDSIM_ROOT / "workflows/miniapps_workflow")}

        workflow = WorkflowClass(
            config=cfg,
            asyncflow=asyncflow,
            home_dir=replica_home,
            name=replica_id,
            on_ready=self._on_ready,
            policies=self.policies,
        )

        await workflow.start()
