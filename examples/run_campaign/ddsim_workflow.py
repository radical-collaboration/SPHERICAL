"""
DDSimWorkflow — runs DummyWorkflow as a subprocess using dummy_workflow conda env.

DummyWorkflow depends on rose (ML surrogate library) which lives only in the
dummy_workflow env.  DDSimWorkflow launches run_dummy.py via dummy_workflow
Python and communicates the ready signal via a sentinel file.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from src.campaign import BaseWorkflow

_DDSIM_ROOT = Path("/ocean/projects/dmr170002p/goliyad/DeepDriveSim")
_RUN_DUMMY  = _DDSIM_ROOT / "workflows/dummy_workflow/run_dummy.py"
_DUMMY_PYTHON = Path("/ocean/projects/dmr170002p/goliyad/conda_env/dummy_workflow/bin/python")


class DDSimWorkflow(BaseWorkflow):
    """
    Async wrapper that runs one DummyWorkflow replica in a subprocess.

    The subprocess uses dummy_workflow Python (which has rose/rhapsody).
    The CM ready signal is communicated via a sentinel file that this wrapper
    watches and forwards to self._on_ready().
    """

    workflow_id = "ddsim"

    async def run(self, replica_id: str) -> None:
        cfg  = self.config or {}
        base_home    = Path(cfg.get("home_dir", Path.home() / "DDSim")).expanduser()
        replica_home = base_home / replica_id
        replica_home.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"dummy_{replica_id}_"
        ) as f:
            json.dump(cfg, f)
            config_json = f.name

        ready_signal = replica_home / ".ready"
        ready_signal.unlink(missing_ok=True)

        cmd = [
            str(_DUMMY_PYTHON), str(_RUN_DUMMY),
            "--config-json",       config_json,
            "--home-dir",          str(replica_home),
            "--replica-id",        replica_id,
            "--ready-signal-path", str(ready_signal),
        ]

        proc = await asyncio.create_subprocess_exec(*cmd)

        # Watch for ready sentinel in parallel with the subprocess.
        async def _watch_ready():
            while not ready_signal.exists():
                if proc.returncode is not None:
                    return
                await asyncio.sleep(1.0)
            if self._on_ready is not None:
                result = self._on_ready()
                if asyncio.iscoroutine(result):
                    await result

        watch_task = asyncio.get_running_loop().create_task(_watch_ready())
        try:
            await proc.wait()
        finally:
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
            Path(config_json).unlink(missing_ok=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"DummyWorkflow subprocess for {replica_id!r} exited with code {proc.returncode}"
            )
