"""
Shared data types for the campaign manager.

  _GroupInfo    — internal per-group runtime state (not public API)
  ResourcePool  — CPU/GPU availability tracker
  WorkflowStats — public per-group statistics snapshot
"""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base_workflow import BaseWorkflow


@dataclass
class _GroupInfo:
    name: str
    workflow_class: "type[BaseWorkflow]"
    replicas: int
    dependencies: list[str]
    group_config: Optional[dict]
    configured_replicas: int = 0
    min_replicas: int = 0
    max_replicas: int = 0
    priority: int = 0
    required_cpus: int = 0
    required_gpus: int = 0
    dep_threshold: int = 1
    entry_point: str = "run"
    status: str = "pending"
    started_count: int = 0
    running_count: int = 0
    finished_replicas: int = 0
    # Set to True when the workflow explicitly signals it has produced enough
    # data (via cm.signal_ready).  Takes precedence over dep_threshold check.
    ready: bool = False
    # GPU IDs currently held by all running replicas of this group.
    # Populated by _allocate_locked; cleared by _on_replica_finished.
    running_gpu_ids: list[int] = field(default_factory=list)
    # Candidate IDs waiting to be assigned to the next replica that starts.
    # Populated by _flush_sharders_locked when dispatch returns candidate IDs;
    # consumed FIFO by _allocate_locked so replica_idx → candidate_id is stable.
    _pending_candidates: deque = field(default_factory=deque, repr=False)


@dataclass
class ResourcePool:
    """
    Tracks available CPU cores and GPU slots for the campaign.

    Both counters are optional: a value of 0 disables tracking for that
    resource type (unlimited).
    """

    total_cpus: int = 0
    total_gpus: int = 0
    available_cpus: int = field(default=0, init=False)
    available_gpus: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.available_cpus = self.total_cpus
        self.available_gpus = self.total_gpus

    def can_fit(self, cpus: int, gpus: int) -> bool:
        if self.total_cpus > 0 and cpus > self.available_cpus:
            return False
        if self.total_gpus > 0 and gpus > self.available_gpus:
            return False
        return True

    def allocate(self, cpus: int, gpus: int) -> None:
        self.available_cpus -= cpus
        self.available_gpus -= gpus

    def release(self, cpus: int, gpus: int) -> None:
        self.available_cpus += cpus
        self.available_gpus += gpus

    def usage_str(self) -> str:
        parts = []
        if self.total_cpus > 0:
            parts.append(f"cpus={self.total_cpus - self.available_cpus}/{self.total_cpus}")
        if self.total_gpus > 0:
            parts.append(f"gpus={self.total_gpus - self.available_gpus}/{self.total_gpus}")
        return "  ".join(parts) if parts else "—"

    def available_str(self) -> str:
        parts = []
        if self.total_cpus > 0:
            parts.append(f"cpus={self.available_cpus}/{self.total_cpus}")
        if self.total_gpus > 0:
            parts.append(f"gpus={self.available_gpus}/{self.total_gpus}")
        return "  ".join(parts) if parts else "—"

    def as_dict(self) -> dict:
        return {
            "total_cpus": self.total_cpus,
            "available_cpus": self.available_cpus,
            "total_gpus": self.total_gpus,
            "available_gpus": self.available_gpus,
        }


@dataclass
class WorkflowStats:
    """Cumulative statistics for one workflow group."""
    replicas_started: int = 0
    replicas_finished: int = 0
