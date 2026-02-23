"""Campaign management for multi-workflow orchestration."""

from .base_workflow import BaseWorkflow, TaskSpec
from .campaign_manager import CampaignManager

__all__ = ["CampaignManager", "BaseWorkflow", "TaskSpec"]
