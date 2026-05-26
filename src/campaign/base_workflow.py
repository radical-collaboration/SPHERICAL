"""
BaseWorkflow — base class for all campaign workflow implementations.

Subclass contract
-----------------
- ``workflow_id``  (class attr, str): unique prefix for replica IDs.
- ``run(replica_id)`` **or** ``start(replica_id)``: execute the workflow.
  Exactly one must be defined.  Async coroutines are awaited directly;
  sync functions are run via ``asyncio.to_thread``.
- ``on_replica_done(replica_id, cm, final_state)`` (optional): hook called
  by the CM after the entry-point returns or raises.
  ``final_state`` is ``"done"`` or ``"failed"``.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .campaign_manager import AsyncCampaignManager


class BaseWorkflow:
    workflow_id: str = "base"

    def __init__(
        self,
        config: Optional[dict] = None,
        _cm: Optional["AsyncCampaignManager"] = None,
        _group_name: Optional[str] = None,
        asyncflow: Optional[object] = None,
        policies: Optional[list] = None,
        engine_dragon: Optional[object] = None,
    ) -> None:
        self.config = config
        self._cm = _cm
        self._group_name = _group_name
        self.asyncflow = asyncflow
        # One Dragon Policy per assigned GPU, injected by AsyncCampaignManager.
        self.policies: list = policies or []
        self.engine_dragon: Optional[object] = engine_dragon

    async def _trigger_dependent(
        self,
        name: str,
        replicas: int = 1,
        **kwargs,
    ) -> None:
        """Tell the CM to activate a dependent workflow group."""
        if self._cm is not None:
            await self._cm.trigger_dependent(name, replicas=replicas, **kwargs)

    async def _trigger_batch(
        self,
        name: str,
        candidates: list[dict],
    ) -> None:
        """Add multiple candidates to the sharder buffer in one scheduler cycle.

        Fills the buffer with N candidates before dispatch() runs, enabling
        meaningful priority ranking.  Each dict must contain ``candidate_id``
        and may include ``score``, ``scaffold_class``, ``surrogate_pred``,
        ``surrogate_unc``.
        """
        if self._cm is not None:
            await self._cm.trigger_batch(name, candidates)

    async def _signal_done(self) -> None:
        """Signal the CM that this workflow has finished producing data."""
        if self._cm is not None and self._group_name is not None:
            await self._cm.signal_done(self._group_name)

    def run(self, replica_id: str) -> None:
        """Execute the workflow for one replica. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__}.run() not implemented (replica_id={replica_id!r})"
        )

    def on_replica_done(
        self,
        replica_id: str,
        cm: "AsyncCampaignManager",
        final_state: str,
    ) -> None:
        """Hook called after this replica's entry-point finishes. No-op by default."""
