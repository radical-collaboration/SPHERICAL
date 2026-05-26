"""
CampaignManager — synchronous shim around AsyncCampaignManager.

Runs a dedicated event loop in a background thread so callers without an
async context can orchestrate workflows with plain blocking calls.
"""

import asyncio
from typing import Optional

from .base_workflow import BaseWorkflow
from .campaign_manager import AsyncCampaignManager
from .types import WorkflowStats


class CampaignManager:
    """Synchronous campaign manager — thin wrapper around AsyncCampaignManager."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        engine: str = "concurrent",
        total_cpus: int = 0,
        total_gpus: int = 0,
        num_workers: Optional[int] = None,
        debug: bool = False,
        asyncflow=None,
        engine_dragon=None,
        features: Optional[dict] = None,
    ) -> None:
        import threading

        self._acm = AsyncCampaignManager(
            max_workers=max_workers,
            engine=engine,
            total_cpus=total_cpus,
            total_gpus=total_gpus,
            num_workers=num_workers,
            debug=debug,
            asyncflow=asyncflow,
            engine_dragon=engine_dragon,
            features=features,
        )

        async def _noop_init() -> None:
            pass

        self._acm._setup_resources = _noop_init

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="CampaignManagerLoop"
        )
        self._thread.start()

    def register_group(self, *args, **kwargs) -> None:
        self._acm.register_group(*args, **kwargs)

    def start(self) -> None:
        future = asyncio.run_coroutine_threadsafe(self._acm.start(), self._loop)
        future.result()

    def wait(self, timeout: Optional[float] = None) -> bool:
        future = asyncio.run_coroutine_threadsafe(self._acm.wait(timeout=timeout), self._loop)
        outer_timeout = (timeout + 2.0) if timeout is not None else None
        try:
            return bool(future.result(timeout=outer_timeout))
        except Exception:
            return False

    def close(self) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(self._acm.close(), self._loop)
            future.result(timeout=5.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def status(self) -> dict:
        return self._acm.status()

    def stats(self) -> dict[str, WorkflowStats]:
        return self._acm.stats()

    @classmethod
    def from_config(
        cls,
        config: dict,
        workflow_registry: dict[str, type[BaseWorkflow]],
        **kwargs,
    ) -> "CampaignManager":
        res_cfg = config.get("resources", {})
        num_workers = config.get("num_workers")
        cm = cls(
            max_workers=config.get("max_workers"),
            engine=config.get("engine", "concurrent"),
            total_cpus=int(res_cfg.get("total_cpus", 0)),
            total_gpus=int(res_cfg.get("total_gpus", 0)),
            num_workers=int(num_workers) if num_workers is not None else None,
            debug=bool(config.get("debug", False)),
            **kwargs,
        )

        _cm_keys = {
            "replicas", "dependencies", "dependency_threshold", "priority",
            "min_replicas", "max_replicas", "required_cpus", "required_gpus",
            "concurrency_cap",
        }

        for name, wf_cfg in config.get("workflows", {}).items():
            wf_class = workflow_registry.get(name)
            if wf_class is None:
                continue
            has_deps = bool(wf_cfg.get("dependencies", []))
            default_replicas = 0 if has_deps else 1
            max_replicas = int(wf_cfg.get("max_replicas") or
                               wf_cfg.get("concurrency_cap") or 0)
            cm.register_group(
                name=name,
                workflow_class=wf_class,
                replicas=int(wf_cfg.get("replicas", default_replicas)),
                dependencies=list(wf_cfg.get("dependencies", [])),
                dep_threshold=int(wf_cfg.get("dependency_threshold", 1)),
                priority=int(wf_cfg.get("priority", 0)),
                min_replicas=int(wf_cfg.get("min_replicas", 0)),
                max_replicas=max_replicas,
                required_cpus=int(wf_cfg.get("required_cpus", 0)),
                required_gpus=int(wf_cfg.get("required_gpus", 0)),
                config={k: v for k, v in wf_cfg.items() if k not in _cm_keys} or None,
            )

        return cm
