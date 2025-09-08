#!/usr/bin/env python3
"""
Microsite Integration for Tuning Campaigns
==========================================

Integrates with the microsite to auto-annotate validated configurations with
"Tuned-vX (Validated)" labels and provide comprehensive campaign results.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

from validation import PromotionDecision
from campaign_manager import Campaign
from priority_scoring import CampaignPriority

logger = logging.getLogger(__name__)

@dataclass
class MicrositeConfigAnnotation:
    """Annotation for a validated configuration on the microsite"""
    config_id: str
    campaign_id: str
    campaign_name: str
    slice_name: str
    
    # Validation status
    validation_status: str  # "validated", "conditional", "rejected"
    validation_timestamp: str
    
    # Performance metrics
    delta_p5: float
    delta_p5_ci: List[float]  # [lower, upper]
    latency_impact: float
    
    # Configuration details
    parameters: Dict[str, Any]
    gates_passed: List[str]
    
    # Optional fields with defaults
    validation_version: str = "v1.0"
    detailed_results_url: Optional[str] = None
    campaign_artifacts_path: Optional[str] = None
    
    def to_microsite_format(self) -> Dict[str, Any]:
        """Convert to microsite display format"""
        return {
            "config_id": self.config_id,
            "display_name": f"Tuned-{self.validation_version} (Validated)",
            "campaign_info": {
                "campaign_id": self.campaign_id,
                "campaign_name": self.campaign_name,
                "slice_name": self.slice_name,
                "validation_timestamp": self.validation_timestamp
            },
            "performance": {
                "delta_p5": {
                    "value": self.delta_p5,
                    "confidence_interval": self.delta_p5_ci,
                    "display": f"+{self.delta_p5:.1%} P@5"
                },
                "latency_impact": {
                    "value": self.latency_impact,
                    "display": f"{self.latency_impact:+.1f}ms P95"
                }
            },
            "validation": {
                "status": self.validation_status,
                "gates_passed": self.gates_passed,
                "version": self.validation_version
            },
            "configuration": {
                "parameters": self.parameters,
                "tuning_approach": "Bayesian Optimization with Campaign Gates"
            },
            "links": {
                "detailed_results": self.detailed_results_url,
                "campaign_artifacts": self.campaign_artifacts_path
            }
        }

@dataclass 
class MicrositeCampaignSummary:
    """Summary of campaign for microsite dashboard"""
    campaign_id: str
    campaign_name: str
    slice_name: str
    budget_tier: int
    
    # Status and timing
    status: str
    start_time: str
    end_time: Optional[str]
    duration_hours: Optional[float]
    
    # Results summary
    total_trials: int
    successful_trials: int
    best_improvement: Optional[float]
    validation_decision: Optional[str]
    
    # Artifacts
    configuration_url: Optional[str] = None
    detailed_report_url: Optional[str] = None
    
    def to_dashboard_card(self) -> Dict[str, Any]:
        """Convert to microsite dashboard card format"""
        status_colors = {
            "completed": "success",
            "running": "primary", 
            "failed": "danger",
            "cancelled": "secondary"
        }
        
        return {
            "campaign_id": self.campaign_id,
            "title": self.campaign_name,
            "subtitle": f"{self.slice_name} (Budget: {self.budget_tier}%)",
            "status": {
                "label": self.status.title(),
                "color": status_colors.get(self.status, "secondary")
            },
            "metrics": {
                "trials": f"{self.successful_trials}/{self.total_trials}",
                "improvement": f"+{self.best_improvement:.1%}" if self.best_improvement else "N/A",
                "duration": f"{self.duration_hours:.1f}h" if self.duration_hours else "N/A"
            },
            "validation": {
                "decision": self.validation_decision or "Pending",
                "decision_color": {
                    "approve": "success",
                    "conditional": "warning", 
                    "reject": "danger"
                }.get(self.validation_decision, "secondary")
            },
            "links": {
                "configuration": self.configuration_url,
                "detailed_report": self.detailed_report_url
            }
        }

class MicrositeIntegrator:
    """Main integration class for microsite communication"""
    
    def __init__(self, 
                 microsite_api_base: str = "http://localhost:3000/api",
                 artifacts_base_url: str = "https://lethe.ai/campaigns"):
        self.microsite_api_base = microsite_api_base
        self.artifacts_base_url = artifacts_base_url
        self.integration_log: List[Dict[str, Any]] = []
    
    def annotate_validated_configuration(self, 
                                       campaign: Campaign,
                                       promotion_decision: PromotionDecision) -> Optional[str]:
        """Annotate validated configuration on microsite"""
        
        if promotion_decision.decision != "approve":
            logger.info(f"Configuration not validated for promotion: {promotion_decision.decision}")
            return None
        
        best_trial = campaign.best_trial
        if not best_trial:
            logger.error("No best trial found for validated campaign")
            return None
        
        # Generate configuration annotation
        annotation = MicrositeConfigAnnotation(
            config_id=f"{campaign.campaign_id}_{best_trial.trial_id}",
            campaign_id=campaign.campaign_id,
            campaign_name=campaign.spec.name,
            slice_name=campaign.spec.slice_candidate.slice_name,
            validation_status="validated",
            validation_timestamp=datetime.now().isoformat(),
            delta_p5=best_trial.metrics.get("delta_p5", 0.0),
            delta_p5_ci=[
                best_trial.metrics.get("ci_lower", 0.0),
                best_trial.metrics.get("ci_upper", 0.0)
            ],
            latency_impact=best_trial.metrics.get("latency_p95_delta", 0.0),
            parameters=best_trial.parameters,
            gates_passed=[
                result.check_name for result in promotion_decision.gate_results 
                if result.passed
            ],
            detailed_results_url=f"{self.artifacts_base_url}/{campaign.campaign_id}/results.html",
            campaign_artifacts_path=f"{self.artifacts_base_url}/{campaign.campaign_id}"
        )
        
        # Send to microsite
        success = self._post_configuration_annotation(annotation)
        
        if success:
            logger.info(f"Successfully annotated configuration {annotation.config_id} on microsite")
            
            # Log integration
            self.integration_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "configuration_annotation",
                "campaign_id": campaign.campaign_id,
                "config_id": annotation.config_id,
                "status": "success"
            })
            
            return annotation.config_id
        else:
            logger.error(f"Failed to annotate configuration {annotation.config_id}")
            return None
    
    def update_campaign_dashboard(self, 
                                campaigns: List[Campaign]) -> bool:
        """Update microsite campaign dashboard"""
        
        summaries = []
        for campaign in campaigns:
            summary = self._create_campaign_summary(campaign)
            summaries.append(summary)
        
        dashboard_data = {
            "campaigns": [s.to_dashboard_card() for s in summaries],
            "last_updated": datetime.now().isoformat(),
            "total_campaigns": len(campaigns),
            "active_campaigns": len([c for c in campaigns if c.status.value == "running"]),
            "completed_campaigns": len([c for c in campaigns if c.status.value == "completed"])
        }
        
        success = self._post_dashboard_update(dashboard_data)
        
        if success:
            logger.info(f"Successfully updated campaign dashboard with {len(campaigns)} campaigns")
        else:
            logger.error("Failed to update campaign dashboard")
        
        return success
    
    def create_campaign_report(self, 
                              campaign: Campaign,
                              promotion_decision: Optional[PromotionDecision] = None) -> str:
        """Create comprehensive campaign report for microsite"""
        
        report = {
            "campaign_id": campaign.campaign_id,
            "report_generated": datetime.now().isoformat(),
            "campaign": {
                "name": campaign.spec.name,
                "slice_name": campaign.spec.slice_candidate.slice_name,
                "budget_tier": campaign.spec.budget_tier,
                "description": campaign.spec.description,
                "objective": campaign.spec.objective_function,
                "expected_improvement": campaign.spec.expected_improvement
            },
            "execution": {
                "status": campaign.status.value,
                "total_trials": len(campaign.trials),
                "successful_trials": len([t for t in campaign.trials if t.status.value == "completed"]),
                "start_time": campaign.start_time.isoformat() if campaign.start_time else None,
                "end_time": campaign.end_time.isoformat() if campaign.end_time else None,
                "duration_hours": (
                    (campaign.end_time - campaign.start_time).total_seconds() / 3600
                    if campaign.start_time and campaign.end_time else None
                )
            },
            "optimization_space": {
                "knobs": [
                    {
                        "name": knob.name,
                        "type": knob.knob_type,
                        "bounds": knob.bounds,
                        "description": knob.description,
                        "risk_level": knob.risk_level
                    }
                    for knob in campaign.spec.knob_spaces
                ]
            },
            "validation_gates": campaign.spec.gates,
            "results": self._create_results_analysis(campaign),
            "best_configuration": (
                {
                    "trial_id": campaign.best_trial.trial_id,
                    "parameters": campaign.best_trial.parameters,
                    "metrics": campaign.best_trial.metrics,
                    "objective_value": campaign.best_objective
                }
                if campaign.best_trial else None
            ),
            "promotion_decision": (
                {
                    "decision": promotion_decision.decision,
                    "confidence_score": promotion_decision.confidence_score,
                    "rationale": promotion_decision.rationale,
                    "conditions": promotion_decision.conditions,
                    "next_steps": promotion_decision.next_steps
                }
                if promotion_decision else None
            ),
            "raw_data": {
                "trials": [trial.to_dict() for trial in campaign.trials],
                "campaign_spec": asdict(campaign.spec)
            }
        }
        
        # Save report to artifacts
        report_path = self._save_campaign_report(campaign.campaign_id, report)
        
        # Post summary to microsite
        self._post_campaign_report_summary(campaign.campaign_id, report)
        
        return report_path
    
    def create_priority_analysis_report(self, 
                                      priorities: List[CampaignPriority]) -> str:
        """Create priority analysis report for microsite"""
        
        report = {
            "analysis_generated": datetime.now().isoformat(),
            "total_candidates": len(priorities),
            "top_candidates": [
                {
                    "rank": p.rank,
                    "slice_name": p.slice_candidate.slice_name,
                    "budget_tier": p.slice_candidate.budget_tier,
                    "priority_score": p.priority_score,
                    "gap_p5": p.slice_candidate.gap_p5,
                    "traffic_weight": p.slice_candidate.total_weight,
                    "expected_improvement": p.priority_score * 0.1  # Rough estimate
                }
                for p in priorities[:10]
            ],
            "score_distribution": {
                "mean": float(np.mean([p.priority_score for p in priorities])),
                "std": float(np.std([p.priority_score for p in priorities])),
                "median": float(np.median([p.priority_score for p in priorities])),
                "range": [
                    float(min(p.priority_score for p in priorities)),
                    float(max(p.priority_score for p in priorities))
                ]
            },
            "budget_tier_breakdown": self._analyze_priorities_by_budget(priorities),
            "domain_breakdown": self._analyze_priorities_by_domain(priorities)
        }
        
        # Save and post to microsite
        report_path = self._save_priority_report(report)
        self._post_priority_analysis(report)
        
        return report_path
    
    def _create_campaign_summary(self, campaign: Campaign) -> MicrositeCampaignSummary:
        """Create campaign summary for dashboard"""
        duration_hours = None
        if campaign.start_time and campaign.end_time:
            duration_hours = (campaign.end_time - campaign.start_time).total_seconds() / 3600
        elif campaign.start_time:
            duration_hours = (datetime.now() - campaign.start_time).total_seconds() / 3600
        
        return MicrositeCampaignSummary(
            campaign_id=campaign.campaign_id,
            campaign_name=campaign.spec.name,
            slice_name=campaign.spec.slice_candidate.slice_name,
            budget_tier=campaign.spec.budget_tier,
            status=campaign.status.value,
            start_time=campaign.start_time.isoformat() if campaign.start_time else datetime.now().isoformat(),
            end_time=campaign.end_time.isoformat() if campaign.end_time else None,
            duration_hours=duration_hours,
            total_trials=len(campaign.trials),
            successful_trials=len([t for t in campaign.trials if t.status.value == "completed"]),
            best_improvement=campaign.best_objective,
            validation_decision=None,  # Would be set by promotion pipeline
            configuration_url=f"{self.artifacts_base_url}/{campaign.campaign_id}/config.json",
            detailed_report_url=f"{self.artifacts_base_url}/{campaign.campaign_id}/report.html"
        )
    
    def _create_results_analysis(self, campaign: Campaign) -> Dict[str, Any]:
        """Create results analysis section"""
        completed_trials = [t for t in campaign.trials if t.status.value == "completed"]
        
        if not completed_trials:
            return {"message": "No completed trials"}
        
        # Extract objective values
        objectives = [t.objective_value for t in completed_trials if t.objective_value is not None]
        
        analysis = {
            "optimization_progress": {
                "total_trials": len(completed_trials),
                "successful_evaluations": len(objectives),
                "best_objective": max(objectives) if objectives else None,
                "mean_objective": float(np.mean(objectives)) if objectives else None,
                "improvement_over_baseline": campaign.best_objective if campaign.best_objective else None
            },
            "parameter_exploration": self._analyze_parameter_exploration(completed_trials),
            "convergence_analysis": self._analyze_convergence(objectives) if objectives else None
        }
        
        return analysis
    
    def _analyze_parameter_exploration(self, trials: List) -> Dict[str, Any]:
        """Analyze parameter space exploration"""
        if not trials:
            return {}
        
        # Collect all parameter values
        param_ranges = {}
        for trial in trials:
            for param_name, param_value in trial.parameters.items():
                if param_name not in param_ranges:
                    param_ranges[param_name] = []
                param_ranges[param_name].append(param_value)
        
        exploration_analysis = {}
        for param_name, values in param_ranges.items():
            if all(isinstance(v, (int, float)) for v in values):
                exploration_analysis[param_name] = {
                    "explored_range": [float(min(values)), float(max(values))],
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
            else:
                # Categorical parameter
                unique_values = list(set(values))
                exploration_analysis[param_name] = {
                    "unique_values": unique_values,
                    "most_common": max(set(values), key=values.count)
                }
        
        return exploration_analysis
    
    def _analyze_convergence(self, objectives: List[float]) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        if len(objectives) < 3:
            return {"message": "Insufficient data for convergence analysis"}
        
        # Calculate cumulative best
        cumulative_best = []
        current_best = objectives[0]
        
        for obj in objectives:
            current_best = max(current_best, obj)
            cumulative_best.append(current_best)
        
        # Simple convergence metrics
        final_best = cumulative_best[-1]
        halfway_best = cumulative_best[len(cumulative_best) // 2]
        
        return {
            "convergence_curve": cumulative_best,
            "final_improvement": final_best,
            "halfway_improvement": halfway_best,
            "convergence_rate": (final_best - halfway_best) / (len(cumulative_best) / 2),
            "stabilized": abs(cumulative_best[-1] - cumulative_best[-min(5, len(cumulative_best))]) < 0.001
        }
    
    def _analyze_priorities_by_budget(self, priorities: List[CampaignPriority]) -> Dict[int, Dict[str, Any]]:
        """Analyze priorities by budget tier"""
        by_budget = {}
        
        for tier in [8, 15, 30]:
            tier_priorities = [p for p in priorities if p.slice_candidate.budget_tier == tier]
            if tier_priorities:
                scores = [p.priority_score for p in tier_priorities]
                by_budget[tier] = {
                    "count": len(tier_priorities),
                    "mean_score": float(np.mean(scores)),
                    "max_score": float(max(scores)),
                    "top_candidate": tier_priorities[0].slice_candidate.slice_name
                }
        
        return by_budget
    
    def _analyze_priorities_by_domain(self, priorities: List[CampaignPriority]) -> Dict[str, Dict[str, Any]]:
        """Analyze priorities by domain"""
        by_domain = {}
        domains = set(p.slice_candidate.domain for p in priorities)
        
        for domain in domains:
            domain_priorities = [p for p in priorities if p.slice_candidate.domain == domain]
            scores = [p.priority_score for p in domain_priorities]
            by_domain[domain] = {
                "count": len(domain_priorities),
                "mean_score": float(np.mean(scores)),
                "top_candidate": max(domain_priorities, 
                                   key=lambda p: p.priority_score).slice_candidate.slice_name
            }
        
        return by_domain
    
    def _post_configuration_annotation(self, annotation: MicrositeConfigAnnotation) -> bool:
        """Post configuration annotation to microsite API"""
        try:
            # In a real implementation, this would make an HTTP POST request
            # For now, simulate success
            logger.info(f"[MOCK] Posting configuration annotation: {annotation.config_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to post configuration annotation: {str(e)}")
            return False
    
    def _post_dashboard_update(self, dashboard_data: Dict[str, Any]) -> bool:
        """Post dashboard update to microsite"""
        try:
            # Mock implementation
            logger.info(f"[MOCK] Updating dashboard with {dashboard_data['total_campaigns']} campaigns")
            return True
        except Exception as e:
            logger.error(f"Failed to update dashboard: {str(e)}")
            return False
    
    def _post_campaign_report_summary(self, campaign_id: str, report: Dict[str, Any]) -> bool:
        """Post campaign report summary to microsite"""
        try:
            # Mock implementation
            logger.info(f"[MOCK] Posting campaign report summary for {campaign_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to post campaign report: {str(e)}")
            return False
    
    def _post_priority_analysis(self, report: Dict[str, Any]) -> bool:
        """Post priority analysis to microsite"""
        try:
            logger.info(f"[MOCK] Posting priority analysis with {report['total_candidates']} candidates")
            return True
        except Exception as e:
            logger.error(f"Failed to post priority analysis: {str(e)}")
            return False
    
    def _save_campaign_report(self, campaign_id: str, report: Dict[str, Any]) -> str:
        """Save campaign report to artifacts directory"""
        artifacts_dir = Path("./campaign_artifacts") / campaign_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = artifacts_dir / "detailed_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved campaign report to {report_path}")
        return str(report_path)
    
    def _save_priority_report(self, report: Dict[str, Any]) -> str:
        """Save priority analysis report"""
        artifacts_dir = Path("./campaign_artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = artifacts_dir / f"priority_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved priority analysis to {report_path}")
        return str(report_path)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and logs"""
        return {
            "integration_active": True,
            "microsite_api_base": self.microsite_api_base,
            "artifacts_base_url": self.artifacts_base_url,
            "total_integrations": len(self.integration_log),
            "recent_integrations": self.integration_log[-10:],  # Last 10
            "last_integration": self.integration_log[-1] if self.integration_log else None
        }

if __name__ == "__main__":
    # Test microsite integration
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from campaign_manager import Campaign, CampaignSpec, Trial, CampaignStatus, TrialStatus
    from priority_scoring import SliceCandidate
    
    # Create test campaign
    candidate = SliceCandidate(
        slice_name="test.integration@15%",
        budget_tier=15,
        domain="test",
        complexity="integration",
        lethe_p5=0.70,
        competitor_p5=0.85,
        ci_width=0.02,
        sensitivity_k2=0.1, sensitivity_lambda=0.08, sensitivity_mu=0.05,
        sensitivity_r=0.06, sensitivity_tau=0.04,
        traffic_weight=1.0, tenant_weight=1.0,
        kv_prefix_drop_risk=0.02, ece_drift_risk=0.01,
        latency_inflation_risk=0.03, complexity_risk=0.1,
        sample_size=200, last_updated="2025-01-15"
    )
    
    spec = CampaignSpec(
        name="Test Microsite Integration",
        slice_candidate=candidate,
        knob_spaces=[],
        n_trials=10
    )
    
    # Create mock campaign with results
    campaign = Campaign(
        campaign_id="test_campaign_001",
        spec=spec,
        status=CampaignStatus.COMPLETED
    )
    
    # Add mock trial
    trial = Trial(
        trial_id="test_trial_001",
        campaign_id=campaign.campaign_id,
        trial_number=1,
        parameters={"lambda": 0.05, "K2": 25},
        status=TrialStatus.COMPLETED,
        metrics={
            "delta_p5": 0.028,
            "latency_p95_delta": 0.5,
            "ci_lower": 0.015,
            "ci_upper": 0.041
        },
        objective_value=0.028
    )
    
    campaign.trials = [trial]
    campaign.best_trial = trial
    campaign.best_objective = 0.028
    
    # Test integration
    integrator = MicrositeIntegrator()
    
    # Test dashboard update
    success = integrator.update_campaign_dashboard([campaign])
    print(f"Dashboard update: {'SUCCESS' if success else 'FAILED'}")
    
    # Test campaign report
    report_path = integrator.create_campaign_report(campaign)
    print(f"Campaign report saved to: {report_path}")
    
    # Test integration status
    status = integrator.get_integration_status()
    print(f"Integration status: {status['integration_active']}")
    print(f"Total integrations: {status['total_integrations']}")