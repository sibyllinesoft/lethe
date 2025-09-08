#!/usr/bin/env python3
"""
Specific Campaign Implementations
=================================

Implements the four concrete campaigns specified in the TODO:
1. Zh.QA @ 8% (code-switch fragility)
2. JSON/PassKey @ 15% (fact needles) 
3. Code.Debug @ 15% (long closures)
4. Retrieve.KV @ 30% (KV stability)

Each campaign has specific knob grids, gates, and optimization strategies.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from campaign_manager import CampaignSpec, KnobSpace
from priority_scoring import SliceCandidate

logger = logging.getLogger(__name__)

class ZhQACampaign:
    """
    Zh.QA @ 8% Campaign: Code-switch fragility optimization
    
    Strategy: re-isotonic, CE early-exit cap +20%, K2:+25%, r=16, λ:+5%
    Gates: ΔP@5≥+1.5pp with CI>0, p95∆≤+1ms, KV drop≤1pp
    """
    
    @staticmethod
    def create_spec(slice_candidate: SliceCandidate) -> CampaignSpec:
        """Create campaign specification"""
        knob_spaces = [
            KnobSpace(
                name="re_isotonic",
                knob_type="categorical", 
                bounds=[True, False],
                description="Apply isotonic recalibration",
                risk_level="low"
            ),
            KnobSpace(
                name="ce_early_exit_cap",
                knob_type="real",
                bounds=(1.0, 1.4),  # +20% from baseline of ~1.0
                description="Cross-encoder early exit capacity multiplier",
                risk_level="medium"
            ),
            KnobSpace(
                name="K2_multiplier", 
                knob_type="real",
                bounds=(1.0, 1.5),  # +25% from baseline
                description="Rerank K2 parameter multiplier",
                risk_level="low"
            ),
            KnobSpace(
                name="r_dpp",
                knob_type="integer",
                bounds=(12, 20),  # Center around r=16
                description="DPP rank parameter",
                risk_level="medium"
            ),
            KnobSpace(
                name="lambda_hybrid",
                knob_type="real", 
                bounds=(0.0, 0.15),  # +5% adjustment range
                description="Hybrid weighting parameter",
                risk_level="low"
            ),
            KnobSpace(
                name="head_keep_unchanged",
                knob_type="categorical",
                bounds=[True],  # Keep head_keep fixed as specified
                description="Keep head_keep parameter unchanged",
                risk_level="low"
            )
        ]
        
        gates = {
            "min_delta_p5": 0.015,  # ΔP@5≥+1.5pp
            "min_ci_confidence": 0.0,  # CI>0 (positive improvement)
            "max_latency_p95_delta": 1.0,  # p95∆≤+1ms
            "max_kv_drop": 0.01  # KV drop≤1pp
        }
        
        return CampaignSpec(
            name="Zh.QA Code-Switch Fragility",
            slice_candidate=slice_candidate,
            knob_spaces=knob_spaces,
            objective_function="delta_p5_per_ms",  # Optimize quality per latency cost
            n_trials=15,
            gates=gates,
            validator_fences={
                "max_ece_drift": 0.08,
                "min_p95_vs_avg": 1.0,  # p95≥avg
                "max_p99_p95_ratio": 2.5,
                "max_proxy_gap": 0.005,  # ≤0.5%
                "max_kv_prefix_drop": 0.03  # ≤3pp
            },
            description="Optimize Chinese QA performance for code-switch scenarios",
            expected_improvement=0.025,
            risk_assessment="low-medium"
        )
    
    @staticmethod
    def get_specific_config_hints() -> Dict[str, Any]:
        """Get configuration hints for this campaign"""
        return {
            "focus_areas": ["code_switch_handling", "isotonic_calibration", "ce_optimization"],
            "risk_factors": ["ce_capacity_overflow", "kv_fragmentation"],
            "monitoring_priorities": ["p95_latency", "kv_prefix_stability", "calibration_drift"],
            "success_indicators": ["reduced_code_switch_errors", "improved_multilingual_consistency"]
        }

class JSONPassKeyCampaign:
    """
    JSON/PassKey @ 15% Campaign: Fact needles optimization
    
    Strategy: CE early-exit OFF for CE@k≤50, K2:+25%, λ:+5%, 
              head_micro-summaries ON, γ:+0.1, δ:-0.05
    Gates: Same as Zh.QA + ECE×FACT bin ≤0.06
    """
    
    @staticmethod  
    def create_spec(slice_candidate: SliceCandidate) -> CampaignSpec:
        """Create campaign specification"""
        knob_spaces = [
            KnobSpace(
                name="ce_early_exit_threshold",
                knob_type="integer",
                bounds=(30, 70),  # OFF for CE@k≤50, so threshold around 50
                description="CE early exit threshold (OFF below this value)",
                risk_level="medium"
            ),
            KnobSpace(
                name="K2_multiplier",
                knob_type="real", 
                bounds=(1.0, 1.5),  # +25% from baseline
                description="Rerank K2 parameter multiplier",
                risk_level="low"
            ),
            KnobSpace(
                name="lambda_hybrid",
                knob_type="real",
                bounds=(0.0, 0.15),  # +5% adjustment range
                description="Hybrid weighting parameter", 
                risk_level="low"
            ),
            KnobSpace(
                name="head_micro_summaries",
                knob_type="categorical",
                bounds=[True, False],
                description="Enable head micro-summaries",
                risk_level="medium"
            ),
            KnobSpace(
                name="gamma_parameter",
                knob_type="real",
                bounds=(0.0, 0.2),  # γ:+0.1 adjustment
                description="Gamma parameter for fact extraction",
                risk_level="medium"
            ),
            KnobSpace(
                name="delta_parameter", 
                knob_type="real",
                bounds=(-0.1, 0.05),  # δ:-0.05 adjustment 
                description="Delta parameter for fact weighting",
                risk_level="medium"
            )
        ]
        
        gates = {
            "min_delta_p5": 0.015,  # ΔP@5≥+1.5pp
            "min_ci_confidence": 0.0,  # CI>0
            "max_latency_p95_delta": 1.0,  # p95∆≤+1ms
            "max_kv_drop": 0.01,  # KV drop≤1pp  
            "max_ece_fact_bin": 0.06  # ECE×FACT bin ≤0.06 (additional gate)
        }
        
        return CampaignSpec(
            name="JSON/PassKey Fact Needles",
            slice_candidate=slice_candidate,
            knob_spaces=knob_spaces,
            objective_function="delta_p5_per_ms",
            n_trials=16,
            gates=gates,
            validator_fences={
                "max_ece_drift": 0.08,
                "min_p95_vs_avg": 1.0,
                "max_p99_p95_ratio": 2.5,
                "max_proxy_gap": 0.005,
                "max_kv_prefix_drop": 0.03
            },
            description="Optimize fact needle extraction from JSON/structured data",
            expected_improvement=0.030,
            risk_assessment="medium"
        )
    
    @staticmethod
    def get_specific_config_hints() -> Dict[str, Any]:
        """Get configuration hints for this campaign"""
        return {
            "focus_areas": ["fact_extraction", "structured_data_parsing", "needle_in_haystack"],
            "risk_factors": ["ce_early_exit_disable", "micro_summary_overhead"],
            "monitoring_priorities": ["fact_accuracy", "ece_calibration", "structured_parsing_rate"],
            "success_indicators": ["improved_fact_recall", "better_structured_comprehension"]
        }

class CodeDebugCampaign:
    """
    Code.Debug @ 15% Campaign: Long closures optimization
    
    Strategy: stronger closures ON, head_keep +2-3pp, K2:+15%, r=16, τ=0.75, λ:+5%
    Gates: ILP_used≤10% and zero closure breaks
    """
    
    @staticmethod
    def create_spec(slice_candidate: SliceCandidate) -> CampaignSpec:
        """Create campaign specification"""  
        knob_spaces = [
            KnobSpace(
                name="stronger_closures",
                knob_type="categorical",
                bounds=[True, False], 
                description="Enable stronger closure detection",
                risk_level="medium"
            ),
            KnobSpace(
                name="head_keep_boost",
                knob_type="real",
                bounds=(0.02, 0.04),  # +2-3pp from baseline
                description="Head keep parameter boost (percentage points)",
                risk_level="medium"
            ),
            KnobSpace(
                name="K2_multiplier",
                knob_type="real",
                bounds=(1.0, 1.25),  # +15% from baseline  
                description="Rerank K2 parameter multiplier",
                risk_level="low"
            ),
            KnobSpace(
                name="r_dpp",
                knob_type="integer", 
                bounds=(12, 20),  # Center around r=16
                description="DPP rank parameter",
                risk_level="medium"
            ),
            KnobSpace(
                name="tau_group_split",
                knob_type="real",
                bounds=(0.65, 0.85),  # Center around τ=0.75
                description="Group-split tau parameter",
                risk_level="high"  # Can increase ILP incidence
            ),
            KnobSpace(
                name="lambda_hybrid",
                knob_type="real",
                bounds=(0.0, 0.15),  # +5% adjustment range
                description="Hybrid weighting parameter",
                risk_level="low"
            )
        ]
        
        gates = {
            "max_ilp_used": 0.10,  # ILP_used≤10%
            "max_closure_breaks": 0,  # Zero closure breaks
            "min_delta_p5": 0.015,  # Implied from pattern
            "max_latency_p95_delta": 2.0  # Slightly higher tolerance for code tasks
        }
        
        return CampaignSpec(
            name="Code.Debug Long Closures",
            slice_candidate=slice_candidate,
            knob_spaces=knob_spaces,
            objective_function="delta_p5",  # Focus on quality for code tasks
            n_trials=18,  # Higher because of complexity
            gates=gates,
            validator_fences={
                "max_ece_drift": 0.08,
                "min_p95_vs_avg": 1.0,
                "max_p99_p95_ratio": 2.5,
                "max_proxy_gap": 0.005,
                "max_kv_prefix_drop": 0.03,
                "max_tau_move": 0.1,  # Cap τ moves to ±0.1
                "ilp_monitoring": True  # Alert if ILP_used>10%
            },
            description="Optimize debugging for long code closures and complex control flow",
            expected_improvement=0.035,
            risk_assessment="medium-high"
        )
    
    @staticmethod
    def get_specific_config_hints() -> Dict[str, Any]:
        """Get configuration hints for this campaign"""
        return {
            "focus_areas": ["closure_analysis", "control_flow_tracking", "symbol_resolution"],
            "risk_factors": ["ilp_overflow", "tau_instability", "closure_break_cascade"],
            "monitoring_priorities": ["ilp_usage", "closure_integrity", "control_flow_accuracy"],
            "success_indicators": ["reduced_closure_errors", "improved_debug_accuracy"]
        }

class RetrieveKVCampaign:
    """
    Retrieve.KV @ 30% Campaign: KV stability under bigger budgets
    
    Strategy: maintain head anchor, shrink W/stride before touching head, 
              sinks=64-96, μ:+5%
    Gates: KV prefix-reuse ≥ baseline and p99/p95≤2.0  
    """
    
    @staticmethod
    def create_spec(slice_candidate: SliceCandidate) -> CampaignSpec:
        """Create campaign specification"""
        knob_spaces = [
            KnobSpace(
                name="maintain_head_anchor",
                knob_type="categorical",
                bounds=[True],  # Always maintain as specified
                description="Maintain head anchor stability",
                risk_level="low"
            ),
            KnobSpace(
                name="window_shrink_factor",
                knob_type="real",
                bounds=(0.7, 0.95),  # Shrink W before touching head
                description="Window size shrink factor",
                risk_level="medium"
            ),
            KnobSpace(
                name="stride_shrink_factor", 
                knob_type="real",
                bounds=(0.8, 1.0),  # Shrink stride as alternative
                description="Stride shrink factor",
                risk_level="low"
            ),
            KnobSpace(
                name="sinks_count",
                knob_type="integer",
                bounds=(64, 96),  # sinks=64-96
                description="Number of attention sinks",
                risk_level="medium"
            ),
            KnobSpace(
                name="mu_boost",
                knob_type="real", 
                bounds=(0.0, 0.10),  # μ:+5% adjustment
                description="Mu parameter boost",
                risk_level="low"
            ),
            KnobSpace(
                name="head_touch_order",
                knob_type="categorical",
                bounds=["window_first", "stride_first", "combined"],
                description="Order of parameter adjustment (touch head last)",
                risk_level="medium"
            )
        ]
        
        gates = {
            "min_kv_prefix_reuse": 1.0,  # ≥ baseline (relative to baseline)
            "max_p99_p95_ratio": 2.0,  # p99/p95≤2.0
            "min_delta_p5": 0.010,  # Lower bar due to higher budget tier complexity
            "max_memory_inflation": 0.15  # Control memory growth at 30% budget
        }
        
        return CampaignSpec(
            name="Retrieve.KV Stability",
            slice_candidate=slice_candidate,
            knob_spaces=knob_spaces,
            objective_function="delta_p5_per_ms",
            n_trials=14,
            gates=gates,
            validator_fences={
                "max_ece_drift": 0.08,
                "min_p95_vs_avg": 1.0,
                "max_p99_p95_ratio": 2.5,  # Slightly higher overall tolerance
                "max_proxy_gap": 0.005,
                "max_kv_prefix_drop": 0.02,  # Tighter KV control
                "kv_jaccard_penalty": True,  # KV-prefix Jaccard in BO penalty
                "memory_growth_monitor": True
            },
            description="Optimize KV cache stability for high-budget retrieval scenarios", 
            expected_improvement=0.020,
            risk_assessment="medium"
        )
    
    @staticmethod
    def get_specific_config_hints() -> Dict[str, Any]:
        """Get configuration hints for this campaign"""
        return {
            "focus_areas": ["kv_cache_optimization", "memory_efficiency", "attention_stability"],
            "risk_factors": ["memory_overflow", "kv_fragmentation", "attention_drift"],
            "monitoring_priorities": ["kv_prefix_reuse", "p99_p95_ratio", "memory_growth"],
            "success_indicators": ["stable_kv_reuse", "controlled_latency_tail", "efficient_memory_usage"]
        }

class CampaignFactory:
    """Factory for creating specific campaigns"""
    
    CAMPAIGN_SPECS = {
        "zh_qa_8": ZhQACampaign,
        "json_passkey_15": JSONPassKeyCampaign,
        "code_debug_15": CodeDebugCampaign,
        "retrieve_kv_30": RetrieveKVCampaign
    }
    
    @classmethod
    def create_campaign_spec(cls, 
                           campaign_type: str, 
                           slice_candidate: SliceCandidate) -> CampaignSpec:
        """Create campaign spec for given type"""
        if campaign_type not in cls.CAMPAIGN_SPECS:
            available = ", ".join(cls.CAMPAIGN_SPECS.keys())
            raise ValueError(f"Unknown campaign type '{campaign_type}'. Available: {available}")
        
        campaign_class = cls.CAMPAIGN_SPECS[campaign_type]
        return campaign_class.create_spec(slice_candidate)
    
    @classmethod
    def get_all_campaign_types(cls) -> List[str]:
        """Get list of all available campaign types"""
        return list(cls.CAMPAIGN_SPECS.keys())
    
    @classmethod
    def get_campaign_hints(cls, campaign_type: str) -> Dict[str, Any]:
        """Get configuration hints for campaign type"""
        if campaign_type not in cls.CAMPAIGN_SPECS:
            return {}
        
        campaign_class = cls.CAMPAIGN_SPECS[campaign_type]
        if hasattr(campaign_class, 'get_specific_config_hints'):
            return campaign_class.get_specific_config_hints()
        else:
            return {}
    
    @classmethod
    def create_week1_campaigns(cls) -> List[str]:
        """Get Week 1 campaign types (fast wins, low risk)"""
        return ["zh_qa_8", "json_passkey_15"]
    
    @classmethod 
    def create_week2_campaigns(cls) -> List[str]:
        """Get Week 2 campaign types (harder, higher ROI)"""
        return ["code_debug_15", "retrieve_kv_30"]

def create_demo_slice_candidates() -> Dict[str, SliceCandidate]:
    """Create demo slice candidates for testing"""
    candidates = {}
    
    # Zh.QA @ 8%
    candidates["zh_qa_8"] = SliceCandidate(
        slice_name="Zh.QA@8%",
        budget_tier=8,
        domain="Zh", 
        complexity="QA",
        lethe_p5=0.68,
        competitor_p5=0.83,  # 15pp gap
        ci_width=0.018,
        sensitivity_k2=0.12,
        sensitivity_lambda=0.08,
        sensitivity_mu=0.05,
        sensitivity_r=0.10,
        sensitivity_tau=0.06,
        traffic_weight=0.8,  # Lower traffic for specialized use case
        tenant_weight=1.4,   # High business value for multilingual
        kv_prefix_drop_risk=0.02,
        ece_drift_risk=0.015,
        latency_inflation_risk=0.025,
        complexity_risk=0.12,
        sample_size=180,
        last_updated="2025-01-15T10:00:00"
    )
    
    # JSON/PassKey @ 15% 
    candidates["json_passkey_15"] = SliceCandidate(
        slice_name="JSON.PassKey@15%",
        budget_tier=15,
        domain="JSON",
        complexity="PassKey", 
        lethe_p5=0.72,
        competitor_p5=0.87,  # 15pp gap
        ci_width=0.016,
        sensitivity_k2=0.15,
        sensitivity_lambda=0.11,
        sensitivity_mu=0.08,
        sensitivity_r=0.06,
        sensitivity_tau=0.04,
        traffic_weight=1.2,  # High traffic for structured data
        tenant_weight=1.3,   # High business value
        kv_prefix_drop_risk=0.018,
        ece_drift_risk=0.022,
        latency_inflation_risk=0.030,
        complexity_risk=0.15,
        sample_size=220,
        last_updated="2025-01-15T10:00:00"
    )
    
    # Code.Debug @ 15%
    candidates["code_debug_15"] = SliceCandidate(
        slice_name="Code.Debug@15%", 
        budget_tier=15,
        domain="Code",
        complexity="Debug",
        lethe_p5=0.65,
        competitor_p5=0.80,  # 15pp gap
        ci_width=0.020,
        sensitivity_k2=0.13,
        sensitivity_lambda=0.09,
        sensitivity_mu=0.06,
        sensitivity_r=0.08,
        sensitivity_tau=0.12,  # Higher tau sensitivity for closures
        traffic_weight=1.0,
        tenant_weight=1.5,   # Very high value for code assistance
        kv_prefix_drop_risk=0.025,
        ece_drift_risk=0.018,
        latency_inflation_risk=0.035,
        complexity_risk=0.18,  # High complexity
        sample_size=160,
        last_updated="2025-01-15T10:00:00"
    )
    
    # Retrieve.KV @ 30%
    candidates["retrieve_kv_30"] = SliceCandidate(
        slice_name="Retrieve.KV@30%",
        budget_tier=30,
        domain="Retrieve", 
        complexity="KV",
        lethe_p5=0.78,
        competitor_p5=0.88,  # 10pp gap (smaller but still significant)
        ci_width=0.014,
        sensitivity_k2=0.08,  # Lower sensitivity at higher budgets
        sensitivity_lambda=0.06,
        sensitivity_mu=0.09,  # Higher mu sensitivity
        sensitivity_r=0.05,
        sensitivity_tau=0.03,
        traffic_weight=1.1,
        tenant_weight=1.2,
        kv_prefix_drop_risk=0.035,  # Higher risk at large budgets
        ece_drift_risk=0.012,
        latency_inflation_risk=0.040,
        complexity_risk=0.14,
        sample_size=280,
        last_updated="2025-01-15T10:00:00"
    )
    
    return candidates

if __name__ == "__main__":
    # Test campaign creation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create demo candidates
    candidates = create_demo_slice_candidates()
    
    # Test each campaign type
    for campaign_type in CampaignFactory.get_all_campaign_types():
        print(f"\n=== Testing {campaign_type} ===")
        
        candidate = candidates[campaign_type]
        spec = CampaignFactory.create_campaign_spec(campaign_type, candidate)
        hints = CampaignFactory.get_campaign_hints(campaign_type)
        
        print(f"Campaign: {spec.name}")
        print(f"Slice: {spec.slice_candidate.slice_name}")
        print(f"Knobs: {len(spec.knob_spaces)}")
        print(f"Trials: {spec.n_trials}")
        print(f"Gates: {list(spec.gates.keys())}")
        print(f"Expected improvement: {spec.expected_improvement:.1%}")
        print(f"Risk assessment: {spec.risk_assessment}")
        print(f"Focus areas: {hints.get('focus_areas', [])}")
    
    # Test campaign scheduling
    print(f"\n=== Campaign Schedule ===")
    print(f"Week 1 (fast wins): {CampaignFactory.create_week1_campaigns()}")
    print(f"Week 2 (higher ROI): {CampaignFactory.create_week2_campaigns()}")