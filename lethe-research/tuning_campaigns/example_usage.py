#!/usr/bin/env python3
"""
Tuning Campaigns System - Complete Usage Example
================================================

Demonstrates the complete workflow from priority analysis through campaign 
execution to validation and microsite integration.
"""

import logging
import time
import json
from pathlib import Path

# Import the tuning campaigns system
from tuning_campaigns import (
    CampaignOrchestrator,
    PriorityScorer, 
    CampaignManager,
    CampaignFactory,
    CampaignMonitor,
    CampaignReporter,
    MicrositeIntegrator
)
from tuning_campaigns.specific_campaigns import create_demo_slice_candidates
from tuning_campaigns.validation import PromotionPipeline

def main():
    """Complete example of tuning campaigns system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Tuning Campaigns System Example")
    
    # 1. Initialize the complete orchestrator
    print("=" * 60)
    print("1. INITIALIZING ORCHESTRATOR")
    print("=" * 60)
    
    orchestrator = CampaignOrchestrator(
        output_dir="./example_campaign_output"
    )
    
    print(f"✓ Orchestrator initialized with {len(orchestrator.slice_candidates)} slice candidates")
    
    # 2. Run priority analysis
    print("\n" + "=" * 60)
    print("2. PRIORITY ANALYSIS")
    print("=" * 60)
    
    priorities = orchestrator.execute_priority_analysis()
    
    print(f"✓ Priority analysis complete. Top 5 candidates:")
    for i, p in enumerate(priorities[:5]):
        print(f"  {i+1}. {p.slice_candidate.slice_name:20s} Score: {p.priority_score:8.4f} Gap: {p.slice_candidate.gap_p5:6.1%}")
    
    # 3. Create execution plan
    print("\n" + "=" * 60) 
    print("3. EXECUTION PLAN")
    print("=" * 60)
    
    plan = orchestrator.create_two_week_execution_plan()
    
    print(f"✓ Two-week execution plan created:")
    print(f"  Week 1 (fast wins): {plan.week1_campaigns}")
    print(f"  Week 2 (higher ROI): {plan.week2_campaigns}")
    print(f"  Start time: {plan.week1_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Completion: {plan.completion_deadline.strftime('%Y-%m-%d %H:%M')}")
    
    # 4. Demonstrate individual campaign execution
    print("\n" + "=" * 60)
    print("4. INDIVIDUAL CAMPAIGN EXECUTION")  
    print("=" * 60)
    
    # Execute just the first Week 1 campaign as example
    campaign_type = plan.week1_campaigns[0]  # zh_qa_8
    
    candidates = create_demo_slice_candidates()
    spec = CampaignFactory.create_campaign_spec(campaign_type, candidates[campaign_type])
    spec.n_trials = 8  # Reduced for demo
    
    print(f"✓ Creating {campaign_type} campaign:")
    print(f"  Slice: {spec.slice_candidate.slice_name}")
    print(f"  Knobs: {len(spec.knob_spaces)}")
    print(f"  Gates: {list(spec.gates.keys())}")
    print(f"  Trials: {spec.n_trials}")
    
    campaign = orchestrator.campaign_manager.create_campaign(spec)
    
    # Start monitoring
    print(f"\n✓ Starting monitoring and campaign execution...")
    orchestrator.monitor.start_monitoring(orchestrator.campaign_manager)
    orchestrator.campaign_manager.start_campaign(campaign.campaign_id)
    
    # Monitor progress
    print(f"  Campaign ID: {campaign.campaign_id}")
    print(f"  Monitoring campaign progress...")
    
    # Wait for some trials to complete (max 60 seconds)
    start_time = time.time()
    max_wait_time = 60
    
    while time.time() - start_time < max_wait_time:
        status = orchestrator.campaign_manager.get_campaign_status(campaign.campaign_id)
        progress = status['progress']
        
        print(f"    Progress: {progress['trials_completed']}/{progress['trials_total']} trials completed", end='\r')
        
        # Check if campaign is complete or has made significant progress
        if (status['status'] in ['completed', 'failed'] or 
            progress['trials_completed'] >= 5):
            break
            
        time.sleep(3)
    
    print()  # New line after progress updates
    
    # 5. Show campaign health and monitoring
    print("\n" + "=" * 60)
    print("5. MONITORING AND HEALTH STATUS")
    print("=" * 60)
    
    health = orchestrator.monitor.get_campaign_health(campaign.campaign_id)
    
    print(f"✓ Campaign Health Status:")
    print(f"  Overall Health: {health.overall_health}")
    print(f"  Progress: {health.progress_percentage:.1f}%")
    print(f"  Success Rate: {health.success_rate:.1%}")
    print(f"  Resource Usage: {health.resource_utilization:.1%}")
    print(f"  Active Alerts: {health.active_alerts}")
    
    # Generate campaign report
    reporter = CampaignReporter(orchestrator.monitor)
    report = reporter.generate_campaign_summary_report(campaign.campaign_id)
    
    print(f"\n✓ Campaign Report Generated:")
    print(f"  Campaign: {report['campaign_info']['name']}")
    print(f"  Status: {report['campaign_info']['status']}")
    print(f"  Best Improvement: {report['execution_summary']['best_objective'] or 'N/A'}")
    
    # 6. Validation pipeline (if campaign completed)
    print("\n" + "=" * 60)
    print("6. VALIDATION PIPELINE")
    print("=" * 60)
    
    current_campaign = orchestrator.campaign_manager.campaigns[campaign.campaign_id]
    
    if current_campaign.status.value == 'completed' and current_campaign.best_trial:
        print(f"✓ Running validation for completed campaign...")
        
        try:
            promotion_decision = orchestrator.validate_campaign_for_promotion(campaign_type)
            
            print(f"  Validation Decision: {promotion_decision.decision}")
            print(f"  Confidence Score: {promotion_decision.confidence_score:.2f}")
            print(f"  Rationale: {promotion_decision.rationale}")
            print(f"  Gates Passed: {len([g for g in promotion_decision.gate_results if g.passed])}/{len(promotion_decision.gate_results)}")
            print(f"  Guardrail Violations: {len(promotion_decision.guardrail_violations)}")
            
            if promotion_decision.decision == "approve":
                print(f"  🎉 Configuration approved for production!")
            elif promotion_decision.decision == "conditional":
                print(f"  ⚠️  Conditional approval with conditions:")
                for condition in promotion_decision.conditions:
                    print(f"    - {condition}")
            else:
                print(f"  ❌ Configuration rejected")
                
        except Exception as e:
            print(f"  ⚠️  Validation skipped: {str(e)}")
    else:
        print(f"  ⚠️  Campaign not ready for validation (status: {current_campaign.status.value})")
        print(f"      Best trial: {current_campaign.best_trial is not None}")
    
    # 7. Microsite integration
    print("\n" + "=" * 60)
    print("7. MICROSITE INTEGRATION")
    print("=" * 60)
    
    integrator = MicrositeIntegrator()
    
    # Update dashboard
    dashboard_success = integrator.update_campaign_dashboard([current_campaign])
    print(f"✓ Dashboard update: {'SUCCESS' if dashboard_success else 'FAILED'}")
    
    # Create campaign report
    report_path = integrator.create_campaign_report(current_campaign)
    print(f"✓ Campaign report: {report_path}")
    
    # Check integration status
    status = integrator.get_integration_status()
    print(f"✓ Integration status: {status['integration_active']}")
    print(f"  Total integrations: {status['total_integrations']}")
    
    # 8. Overall system status
    print("\n" + "=" * 60)
    print("8. OVERALL SYSTEM STATUS")
    print("=" * 60)
    
    orchestrator.active_campaigns[campaign_type] = current_campaign
    overall_status = orchestrator.get_overall_status()
    
    print(f"✓ System Status:")
    print(f"  Active Campaigns: {overall_status['summary']['active_campaigns']}")
    print(f"  Completed Campaigns: {overall_status['summary']['completed_campaigns']}")
    print(f"  Success Rate: {overall_status['summary']['success_rate']:.1%}")
    print(f"  Validated Campaigns: {overall_status['summary']['validated_campaigns']}")
    
    # 9. Generate final report
    print("\n" + "=" * 60)
    print("9. FINAL REPORT")
    print("=" * 60)
    
    final_report = orchestrator.generate_final_report()
    
    print(f"✓ Final Report Generated:")
    print(f"  Total Campaigns: {final_report['dashboard']['total_campaigns']}")
    print(f"  Successful Campaigns: {len(final_report['recommendations']['successful_campaigns'])}")
    print(f"  Promoted Configurations: {final_report['recommendations']['promoted_configurations']}")
    
    print(f"\n✓ Next Steps:")
    for step in final_report['recommendations']['next_steps'][:3]:
        print(f"  - {step}")
    
    # 10. Cleanup
    print("\n" + "=" * 60)
    print("10. CLEANUP")
    print("=" * 60)
    
    orchestrator.shutdown()
    print(f"✓ System shutdown complete")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    
    output_dir = Path("./example_campaign_output")
    if output_dir.exists():
        print(f"✓ All results saved to: {output_dir.absolute()}")
        print(f"  Campaign artifacts: {list(output_dir.glob('campaigns/*'))[:3]}")
        print(f"  Final report: {output_dir / 'final_report.json'}")
        print(f"  Monitoring database: {output_dir / 'monitoring.db'}")
    
    print(f"\n✓ Example demonstrates:")
    print(f"  - Priority scoring with mathematical formula")
    print(f"  - Campaign management with Bayesian Optimization")
    print(f"  - Real-time monitoring and health checks")
    print(f"  - Comprehensive validation pipeline")
    print(f"  - Guardrails and statistical testing")
    print(f"  - Microsite integration")
    print(f"  - Complete reporting and analytics")
    
    print(f"\n🎉 Tuning Campaigns System ready for production deployment!")

if __name__ == "__main__":
    main()