"""Campaign management for multi-workflow orchestration."""

from .base_workflow import BaseWorkflow, TaskSpec
from .campaign_manager import CampaignManager
from .resource_manager import ResourceManager

__all__ = ["CampaignManager", "BaseWorkflow", "TaskSpec", "ResourceManager"]
