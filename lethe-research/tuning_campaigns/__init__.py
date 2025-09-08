"""
Lethe Tuning Campaigns System
=============================

Production system for running concrete optimization campaigns that build on the 
existing Gap→Tune→Verify framework. Implements sophisticated priority scoring,
Bayesian optimization, and automated validation pipelines for systematic model
optimization.
"""

from priority_scoring import PriorityScorer, SliceCandidate, CampaignPriority
from campaign_manager import CampaignManager, Campaign, CampaignSpec
from specific_campaigns import (
    ZhQACampaign, 
    JSONPassKeyCampaign, 
    CodeDebugCampaign, 
    RetrieveKVCampaign
)
from validation import CampaignValidator, PromotionPipeline, Guardrails
from microsite_integration import MicrositeIntegrator
from monitoring import CampaignMonitor, CampaignReporter

__version__ = "1.0.0"
__all__ = [
    "PriorityScorer", "SliceCandidate", "CampaignPriority",
    "CampaignManager", "Campaign", "CampaignSpec", 
    "ZhQACampaign", "JSONPassKeyCampaign", "CodeDebugCampaign", "RetrieveKVCampaign",
    "CampaignValidator", "PromotionPipeline", "Guardrails",
    "MicrositeIntegrator",
    "CampaignMonitor", "CampaignReporter"
]