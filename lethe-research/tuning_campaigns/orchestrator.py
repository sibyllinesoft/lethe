#!/usr/bin/env python3
"""
Tuning Campaigns Orchestrator
=============================

Main orchestrator that implements the complete two-week campaign execution plan:

Week 1 (fast wins, low risk):
• Zh.QA @ 8% (code-switch fragility) 
• JSON/PassKey @ 15% (fact needles)

Week 2 (harder, higher ROI):
• Code.Debug @ 15% (long closures)
• Retrieve.KV @ 30% (KV stability)

Integrates all components: priority scoring, campaign management, validation,
guardrails, promotion pipeline, and microsite integration.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from priority_scoring import PriorityScorer, SliceCandidate, load_candidates_from_gap_analysis
from campaign_manager import CampaignManager, Campaign, CampaignStatus
from specific_campaigns import CampaignFactory, create_demo_slice_candidates
from validation import PromotionPipeline, PromotionDecision, CampaignValidator, Guardrails
from microsite_integration import MicrositeIntegrator
from monitoring import CampaignMonitor, CampaignReporter

logger = logging.getLogger(__name__)

@dataclass
class ExecutionPlan:
    """Two-week execution plan"""
    week1_campaigns: List[str]  # ["zh_qa_8", "json_passkey_15"]
    week2_campaigns: List[str]  # ["code_debug_15", "retrieve_kv_30"] 
    
    # Timing
    week1_start: datetime
    week2_start: datetime
    completion_deadline: datetime
    
    # Configuration
    max_concurrent_campaigns: int = 2
    budget_allocation: Dict[str, int] = None  # trials per campaign
    
    def __post_init__(self):
        if self.budget_allocation is None:
            self.budget_allocation = {
                "zh_qa_8": 15,
                "json_passkey_15": 16, 
                "code_debug_15": 18,
                "retrieve_kv_30": 14
            }

class CampaignOrchestrator:
    """Main orchestrator for tuning campaigns system"""
    
    def __init__(self, 
                 output_dir: str = "./tuning_campaigns_output",
                 gap_analysis_path: Optional[str] = None,
                 evaluator_function: Optional[Callable] = None):
        """
        Initialize the orchestrator.
        
        Args:
            output_dir: Base directory for all campaign artifacts
            gap_analysis_path: Path to existing Gap→Tune→Verify analysis results
            evaluator_function: Function to evaluate trial configurations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all components
        self.priority_scorer = PriorityScorer(risk_penalty_weight=0.5)
        self.campaign_manager = CampaignManager(
            output_dir=str(self.output_dir / "campaigns"),
            evaluator_function=evaluator_function or self._mock_evaluator
        )
        self.validator = CampaignValidator()
        self.guardrails = Guardrails()
        self.promotion_pipeline = PromotionPipeline()
        self.microsite_integrator = MicrositeIntegrator(
            artifacts_base_url=f"https://lethe.ai/campaigns/{datetime.now().strftime('%Y%m%d')}"
        )
        self.monitor = CampaignMonitor(
            db_path=str(self.output_dir / "monitoring.db"),
            polling_interval=60.0  # 1 minute
        )
        self.reporter = CampaignReporter(self.monitor)
        
        # Execution state
        self.execution_plan: Optional[ExecutionPlan] = None
        self.slice_candidates: Dict[str, SliceCandidate] = {}
        self.active_campaigns: Dict[str, Campaign] = {}
        self.completed_campaigns: Dict[str, Campaign] = {}
        self.promotion_decisions: Dict[str, PromotionDecision] = {}
        
        # Load candidates
        if gap_analysis_path and Path(gap_analysis_path).exists():
            self.slice_candidates = {
                c.slice_name.replace('@', '_').replace('.', '_').replace('%', '').lower(): c
                for c in load_candidates_from_gap_analysis(gap_analysis_path)
            }
            logger.info(f"Loaded {len(self.slice_candidates)} candidates from gap analysis")
        else:
            # Use demo candidates for testing
            self.slice_candidates = create_demo_slice_candidates()
            logger.info("Using demo slice candidates")
        
        logger.info(f"Initialized CampaignOrchestrator with output dir: {self.output_dir}")
    
    def create_two_week_execution_plan(self) -> ExecutionPlan:
        """Create the standard two-week execution plan"""
        now = datetime.now()
        
        plan = ExecutionPlan(
            week1_campaigns=CampaignFactory.create_week1_campaigns(),
            week2_campaigns=CampaignFactory.create_week2_campaigns(),
            week1_start=now,
            week2_start=now + timedelta(days=7),
            completion_deadline=now + timedelta(days=14),
            max_concurrent_campaigns=2
        )
        
        self.execution_plan = plan
        logger.info(f"Created two-week execution plan: Week 1: {plan.week1_campaigns}, Week 2: {plan.week2_campaigns}")
        return plan
    
    def execute_priority_analysis(self) -> List[Any]:
        """Run priority analysis on all slice candidates"""
        logger.info("Starting priority analysis...")
        
        candidates = list(self.slice_candidates.values())
        priorities = self.priority_scorer.score_all_candidates(candidates)
        
        # Export results
        analysis_dir = self.output_dir / "priority_analysis"
        self.priority_scorer.export_results(priorities, analysis_dir)
        
        # Create microsite report
        self.microsite_integrator.create_priority_analysis_report(priorities)
        
        logger.info(f"Completed priority analysis. Top candidate: {priorities[0].slice_candidate.slice_name} (score: {priorities[0].priority_score:.4f})")
        
        return priorities
    
    def start_execution(self, 
                       execution_plan: Optional[ExecutionPlan] = None,
                       dry_run: bool = False) -> None:
        """Start executing the campaign plan"""
        
        if execution_plan:
            self.execution_plan = execution_plan
        elif not self.execution_plan:
            self.execution_plan = self.create_two_week_execution_plan()
        
        logger.info(f"Starting campaign execution (dry_run={dry_run})")
        
        # Start monitoring
        self.monitor.start_monitoring(self.campaign_manager)
        
        if not dry_run:
            # Execute Week 1 campaigns immediately
            self._execute_campaign_week(
                self.execution_plan.week1_campaigns,
                week_number=1
            )
            
            # Schedule Week 2 campaigns
            self._schedule_week2_campaigns()
        else:
            logger.info("DRY RUN: Would start campaigns but dry_run=True")
            self._log_execution_plan()
    
    def _execute_campaign_week(self, campaign_types: List[str], week_number: int) -> None:
        """Execute campaigns for a specific week"""
        logger.info(f"Starting Week {week_number} campaigns: {campaign_types}")
        
        for campaign_type in campaign_types:
            if campaign_type not in self.slice_candidates:
                logger.error(f"No slice candidate found for campaign type: {campaign_type}")
                continue
            
            # Create campaign
            candidate = self.slice_candidates[campaign_type]
            spec = CampaignFactory.create_campaign_spec(campaign_type, candidate)
            
            # Adjust trial count based on budget allocation
            if self.execution_plan and campaign_type in self.execution_plan.budget_allocation:
                spec.n_trials = self.execution_plan.budget_allocation[campaign_type]
            
            campaign = self.campaign_manager.create_campaign(spec)
            self.active_campaigns[campaign_type] = campaign
            
            # Start campaign
            self.campaign_manager.start_campaign(campaign.campaign_id)
            
            logger.info(f"Started {campaign_type} campaign: {campaign.campaign_id}")
    
    def _schedule_week2_campaigns(self) -> None:
        """Schedule Week 2 campaigns to start after Week 1 completes"""
        def check_and_start_week2():
            while True:
                time.sleep(3600)  # Check every hour
                
                # Check if Week 1 campaigns are complete
                week1_complete = all(
                    self.active_campaigns.get(ct, {}).status in [CampaignStatus.COMPLETED, CampaignStatus.FAILED, CampaignStatus.CANCELLED]
                    for ct in self.execution_plan.week1_campaigns
                    if ct in self.active_campaigns
                )
                
                # Or if we've reached Week 2 start time
                week2_time = datetime.now() >= self.execution_plan.week2_start
                
                if week1_complete or week2_time:
                    logger.info("Starting Week 2 campaigns...")
                    self._execute_campaign_week(
                        self.execution_plan.week2_campaigns,
                        week_number=2
                    )
                    break
        
        # Start in background thread
        import threading
        thread = threading.Thread(target=check_and_start_week2, daemon=True)
        thread.start()
    
    def check_campaign_status(self, campaign_type: str) -> Dict[str, Any]:
        """Check status of a specific campaign"""
        if campaign_type not in self.active_campaigns:
            return {"error": f"Campaign {campaign_type} not found"}
        
        campaign = self.active_campaigns[campaign_type]
        
        # Get monitoring health status
        health = self.monitor.get_campaign_health(campaign.campaign_id)
        
        # Get basic campaign status
        status = self.campaign_manager.get_campaign_status(campaign.campaign_id)
        
        return {
            **status,
            "health": {
                "overall_health": health.overall_health,
                "success_rate": health.success_rate,
                "resource_utilization": health.resource_utilization,
                "active_alerts": health.active_alerts
            },
            "validation_ready": campaign.status == CampaignStatus.COMPLETED and campaign.best_trial is not None
        }
    
    def validate_campaign_for_promotion(self, campaign_type: str) -> PromotionDecision:
        """Validate completed campaign for promotion"""
        if campaign_type not in self.active_campaigns:
            raise ValueError(f"Campaign {campaign_type} not found")
        
        campaign = self.active_campaigns[campaign_type]
        
        if campaign.status != CampaignStatus.COMPLETED:
            raise ValueError(f"Campaign {campaign_type} is not completed (status: {campaign.status})")
        
        # Mock baseline metrics (in practice would come from existing system)
        baseline_metrics = {
            "delta_p5": 0.0,
            "latency_p95": 100.0,
            "memory_mb": 500.0,
            "kv_prefix_reuse": 1.0
        }
        
        # Mock cross-dataset results (in practice would run full evaluation)
        cross_dataset_results = self._generate_mock_cross_dataset_results(campaign)
        
        # Run promotion evaluation
        promotion_decision = self.promotion_pipeline.evaluate_for_promotion(
            campaign=campaign,
            baseline_metrics=baseline_metrics,
            cross_dataset_results=cross_dataset_results
        )
        
        # Store decision
        self.promotion_decisions[campaign_type] = promotion_decision
        
        # If approved, integrate with microsite
        if promotion_decision.decision == "approve":
            config_id = self.microsite_integrator.annotate_validated_configuration(
                campaign, promotion_decision
            )
            logger.info(f"Configuration {config_id} annotated on microsite as validated")
        
        # Generate detailed report
        self.microsite_integrator.create_campaign_report(campaign, promotion_decision)
        
        logger.info(f"Campaign {campaign_type} validation decision: {promotion_decision.decision} (confidence: {promotion_decision.confidence_score:.2f})")
        
        return promotion_decision
    
    def _generate_mock_cross_dataset_results(self, campaign: Campaign) -> Dict[str, Dict[str, Any]]:
        """Generate mock cross-dataset validation results"""
        # In practice, this would run the best configuration across all test datasets
        budget_tier = campaign.spec.budget_tier
        
        datasets = [f"dataset_{i}@{budget_tier}%" for i in range(3)]
        results = {}
        
        for dataset in datasets:
            # Mock results with some variation
            base_improvement = campaign.best_objective if campaign.best_objective else 0.02
            noise = np.random.normal(0, 0.005)
            
            results[dataset] = {
                "budget_tier": budget_tier,
                "delta_p5": base_improvement + noise,
                "ci_lower": base_improvement + noise - 0.008,
                "ci_upper": base_improvement + noise + 0.008,
                "p_value": max(0.001, 0.05 - abs(noise) * 10),  # Lower p-value for better results
                "sample_size": 200,
                "latency_delta": np.random.normal(0.5, 1.0),
                "kv_drop": max(0, np.random.normal(0.01, 0.005))
            }
        
        return results
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        if not self.execution_plan:
            return {"error": "No execution plan active"}
        
        now = datetime.now()
        
        # Campaign statuses
        campaign_statuses = {}
        for campaign_type in (self.execution_plan.week1_campaigns + self.execution_plan.week2_campaigns):
            if campaign_type in self.active_campaigns:
                campaign_statuses[campaign_type] = self.check_campaign_status(campaign_type)
        
        # Timing info
        timing = {
            "current_time": now.isoformat(),
            "week1_start": self.execution_plan.week1_start.isoformat(),
            "week2_start": self.execution_plan.week2_start.isoformat(),
            "completion_deadline": self.execution_plan.completion_deadline.isoformat(),
            "days_remaining": (self.execution_plan.completion_deadline - now).days,
            "in_week1": now < self.execution_plan.week2_start,
            "in_week2": self.execution_plan.week2_start <= now < self.execution_plan.completion_deadline
        }
        
        # Summary metrics
        total_campaigns = len(self.execution_plan.week1_campaigns + self.execution_plan.week2_campaigns)
        active_campaigns = len([c for c in campaign_statuses.values() if c.get("status") == "running"])
        completed_campaigns = len([c for c in campaign_statuses.values() if c.get("status") == "completed"])
        validated_campaigns = len(self.promotion_decisions)
        
        return {
            "execution_plan": {
                "week1_campaigns": self.execution_plan.week1_campaigns,
                "week2_campaigns": self.execution_plan.week2_campaigns,
                "timing": timing
            },
            "campaigns": campaign_statuses,
            "summary": {
                "total_campaigns": total_campaigns,
                "active_campaigns": active_campaigns,
                "completed_campaigns": completed_campaigns,
                "validated_campaigns": validated_campaigns,
                "success_rate": completed_campaigns / max(1, total_campaigns)
            },
            "promotion_decisions": {
                ct: {
                    "decision": pd.decision,
                    "confidence_score": pd.confidence_score,
                    "rationale": pd.rationale
                } 
                for ct, pd in self.promotion_decisions.items()
            }
        }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        logger.info("Generating final campaign report...")
        
        # Get dashboard data
        dashboard = self.reporter.generate_multi_campaign_dashboard(
            list(self.active_campaigns.keys())
        )
        
        # Individual campaign reports
        campaign_reports = {}
        for campaign_type, campaign in self.active_campaigns.items():
            try:
                report = self.reporter.generate_campaign_summary_report(campaign.campaign_id)
                campaign_reports[campaign_type] = report
            except Exception as e:
                logger.error(f"Failed to generate report for {campaign_type}: {str(e)}")
                campaign_reports[campaign_type] = {"error": str(e)}
        
        # Overall analysis
        successful_campaigns = [
            ct for ct, pd in self.promotion_decisions.items()
            if pd.decision == "approve"
        ]
        
        final_report = {
            "report_generated": datetime.now().isoformat(),
            "execution_summary": self.get_overall_status(),
            "dashboard": dashboard,
            "campaign_reports": campaign_reports,
            "validation_results": {
                ct: {
                    "decision": pd.decision,
                    "confidence_score": pd.confidence_score,
                    "rationale": pd.rationale,
                    "gates_passed": len([g for g in pd.gate_results if g.passed]),
                    "total_gates": len(pd.gate_results),
                    "guardrail_violations": len(pd.guardrail_violations),
                    "statistical_significance": pd.statistical_tests.get("overall_significant", False)
                }
                for ct, pd in self.promotion_decisions.items()
            },
            "recommendations": {
                "successful_campaigns": successful_campaigns,
                "promoted_configurations": len(successful_campaigns),
                "next_steps": self._generate_next_steps(),
                "lessons_learned": self._extract_lessons_learned()
            }
        }
        
        # Save final report
        report_path = self.output_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {report_path}")
        return final_report
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps recommendations"""
        next_steps = []
        
        # Based on successful campaigns
        successful_campaigns = [
            ct for ct, pd in self.promotion_decisions.items()
            if pd.decision == "approve"
        ]
        
        if successful_campaigns:
            next_steps.append(f"Deploy validated configurations from {len(successful_campaigns)} successful campaigns")
            next_steps.append("Monitor production performance and collect user feedback")
            next_steps.append("Plan next round of optimization targeting remaining gaps")
        else:
            next_steps.append("Analyze campaign failures and adjust optimization approach")
            next_steps.append("Consider expanding knob ranges or modifying gates")
        
        # Add specific recommendations
        next_steps.extend([
            "Update Gap→Tune→Verify framework with learnings",
            "Expand campaign automation based on validation pipeline performance",
            "Consider additional slice candidates for future campaigns"
        ])
        
        return next_steps
    
    def _extract_lessons_learned(self) -> List[str]:
        """Extract lessons learned from campaigns"""
        lessons = [
            "Bayesian optimization with campaign gates provides systematic improvement",
            "Comprehensive guardrails prevent production risks during optimization",
            "Statistical validation with Holm correction reduces false positives",
            "Microsite integration enables transparent communication of results"
        ]
        
        # Add campaign-specific lessons
        for campaign_type, campaign in self.active_campaigns.items():
            if campaign.status == CampaignStatus.COMPLETED:
                if campaign.best_objective and campaign.best_objective > 0.02:
                    lessons.append(f"{campaign_type}: Achieved significant improvement ({campaign.best_objective:.1%})")
                elif campaign_type in self.promotion_decisions:
                    pd = self.promotion_decisions[campaign_type]
                    if pd.decision == "reject":
                        lessons.append(f"{campaign_type}: {pd.rationale}")
        
        return lessons
    
    def _log_execution_plan(self) -> None:
        """Log the execution plan details"""
        plan = self.execution_plan
        
        logger.info("=== EXECUTION PLAN ===")
        logger.info(f"Week 1 Start: {plan.week1_start}")
        logger.info(f"Week 1 Campaigns: {plan.week1_campaigns}")
        for ct in plan.week1_campaigns:
            trials = plan.budget_allocation.get(ct, 15)
            logger.info(f"  - {ct}: {trials} trials")
        
        logger.info(f"Week 2 Start: {plan.week2_start}")
        logger.info(f"Week 2 Campaigns: {plan.week2_campaigns}")
        for ct in plan.week2_campaigns:
            trials = plan.budget_allocation.get(ct, 15)
            logger.info(f"  - {ct}: {trials} trials")
        
        logger.info(f"Completion Deadline: {plan.completion_deadline}")
        logger.info(f"Max Concurrent: {plan.max_concurrent_campaigns}")
    
    def _mock_evaluator(self, parameters: Dict[str, Any], spec) -> Dict[str, float]:
        """Mock evaluator for testing (replace with real implementation)"""
        import numpy as np
        time.sleep(np.random.uniform(2, 8))  # Simulate evaluation time
        
        # Mock improvement based on parameters  
        base_improvement = 0.015  # 1.5pp base
        
        # Parameter-based improvements
        lambda_boost = parameters.get("lambda_hybrid", 0.05) * 0.5
        k2_boost = parameters.get("K2_multiplier", 1.0) * 0.01
        noise = np.random.normal(0, 0.005)
        
        delta_p5 = base_improvement + lambda_boost + k2_boost + noise
        
        return {
            "delta_p5": max(0, delta_p5),
            "delta_p5_std": 0.008,
            "latency_p95_delta": np.random.normal(0.8, 0.5),
            "latency_p95_delta_std": 0.3,
            "kv_prefix_drop": max(0, np.random.normal(0.008, 0.003)),
            "ece_drift": max(0, np.random.normal(0.015, 0.005)),
            "p99_p95_ratio": max(1.0, np.random.normal(1.6, 0.2)),
            "memory_mb": max(0, np.random.normal(520, 30)),
            "ci_lower": max(0, delta_p5 - 0.008),
            "ci_upper": delta_p5 + 0.008
        }
    
    def shutdown(self) -> None:
        """Shutdown orchestrator and all components"""
        logger.info("Shutting down campaign orchestrator...")
        
        self.monitor.stop_monitoring()
        self.campaign_manager.shutdown()
        
        logger.info("Campaign orchestrator shutdown complete")

# CLI interface for the orchestrator
if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Lethe Tuning Campaigns Orchestrator")
    parser.add_argument("command", choices=["plan", "start", "status", "validate", "report"], 
                       help="Command to execute")
    parser.add_argument("--output-dir", default="./tuning_campaigns_output",
                       help="Output directory")
    parser.add_argument("--gap-analysis", help="Path to gap analysis JSON file")
    parser.add_argument("--campaign-type", help="Specific campaign type (for validate command)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = CampaignOrchestrator(
        output_dir=args.output_dir,
        gap_analysis_path=args.gap_analysis
    )
    
    try:
        if args.command == "plan":
            # Create and display execution plan
            plan = orchestrator.create_two_week_execution_plan()
            print(f"Created execution plan:")
            print(f"  Week 1: {plan.week1_campaigns}")
            print(f"  Week 2: {plan.week2_campaigns}")
            print(f"  Start: {plan.week1_start}")
            print(f"  Completion: {plan.completion_deadline}")
            
            # Run priority analysis
            priorities = orchestrator.execute_priority_analysis()
            print(f"\nTop 5 priorities:")
            for i, p in enumerate(priorities[:5]):
                print(f"  {i+1}. {p.slice_candidate.slice_name} (score: {p.priority_score:.4f})")
        
        elif args.command == "start":
            # Start campaign execution
            orchestrator.start_execution(dry_run=args.dry_run)
            
            if not args.dry_run:
                print("Campaigns started. Use 'status' command to monitor progress.")
                print("Press Ctrl+C to stop monitoring...")
                try:
                    while True:
                        time.sleep(60)
                        status = orchestrator.get_overall_status()
                        active = status["summary"]["active_campaigns"]
                        completed = status["summary"]["completed_campaigns"]
                        print(f"Status: {active} active, {completed} completed")
                except KeyboardInterrupt:
                    print("Stopping...")
        
        elif args.command == "status":
            # Show current status
            status = orchestrator.get_overall_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == "validate":
            # Validate specific campaign
            if not args.campaign_type:
                print("--campaign-type required for validate command")
                sys.exit(1)
            
            decision = orchestrator.validate_campaign_for_promotion(args.campaign_type)
            print(f"Validation decision for {args.campaign_type}:")
            print(f"  Decision: {decision.decision}")
            print(f"  Confidence: {decision.confidence_score:.2f}")
            print(f"  Rationale: {decision.rationale}")
        
        elif args.command == "report":
            # Generate final report
            report = orchestrator.generate_final_report()
            print(f"Final report generated:")
            print(f"  Successful campaigns: {len(report['recommendations']['successful_campaigns'])}")
            print(f"  Report saved to: {orchestrator.output_dir}/final_report.json")
    
    finally:
        orchestrator.shutdown()