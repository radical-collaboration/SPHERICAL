"""Campaign management for multi-workflow orchestration."""

from .campaign_manager import AsyncCampaignManager
from .base_workflow import BaseWorkflow
from .sync_wrapper import CampaignManager
from .types import ResourcePool, WorkflowStats
from .backpressure import BackpressureNegotiator, BPState
from .candidate_log import CandidateLog, CandidateHistory, StageResult
from .monitor import Monitor, DriftEvent, DriftKind
from .profiles import ProfileWeights, PROFILES, get_profile
from .sharder import Sharder, ShardingSpec
from .bandit import Bandit, BanditArm, shard_bandit, resource_bandit, SchedulingBandit, scheduling_bandit

__all__ = [
    "AsyncCampaignManager",
    "CampaignManager",
    "BaseWorkflow",
    "ResourcePool",
    "WorkflowStats",
    "BackpressureNegotiator",
    "BPState",
    "CandidateLog",
    "CandidateHistory",
    "StageResult",
    "Monitor",
    "DriftEvent",
    "DriftKind",
    "ProfileWeights",
    "PROFILES",
    "get_profile",
    "Sharder",
    "ShardingSpec",
    "Bandit",
    "BanditArm",
    "shard_bandit",
    "resource_bandit",
    "SchedulingBandit",
    "scheduling_bandit",
]
