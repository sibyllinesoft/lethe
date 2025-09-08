#!/usr/bin/env python3
"""
Priority Scoring System for Tuning Campaigns
============================================

Implements the mathematical priority formula:
score = (max(0,ΔP@5)/CI_width)² × S × T - ρ×R

Where:
- ΔP@5: (competitor−Lethe) on paired slice
- CI_width: paired bootstrap 95% width  
- S: counterfactual sensitivity (∂P/∂K2, ∂P/∂λ from IPS replays)
- T: tenant/traffic weight
- R: risk factors (KV-prefix drop, ECE drift, p99/p95 inflation)
- ρ: fixed penalty weight
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SliceCandidate:
    """A failure slice candidate for optimization"""
    slice_name: str
    budget_tier: int  # 8, 15, or 30
    domain: str
    complexity: str
    
    # Performance metrics
    lethe_p5: float  # Current Lethe P@5 on this slice
    competitor_p5: float  # Best competitor P@5 on this slice
    ci_width: float  # Bootstrap 95% CI width from paired evaluation
    
    # Counterfactual sensitivity estimates (from IPS replays)
    sensitivity_k2: float  # ∂P/∂K2
    sensitivity_lambda: float  # ∂P/∂λ  
    sensitivity_mu: float  # ∂P/∂μ
    sensitivity_r: float  # ∂P/∂r (DPP rank)
    sensitivity_tau: float  # ∂P/∂τ (group-split)
    
    # Traffic/business weighting
    traffic_weight: float  # Normalized traffic volume for this slice
    tenant_weight: float  # Business importance weighting
    
    # Risk factors
    kv_prefix_drop_risk: float  # Historical KV prefix drop rate
    ece_drift_risk: float  # ECE calibration drift risk
    latency_inflation_risk: float  # p99/p95 ratio inflation risk
    complexity_risk: float  # Implementation complexity risk
    
    # Metadata
    sample_size: int  # Number of queries in this slice
    last_updated: str
    
    @property
    def gap_p5(self) -> float:
        """Performance gap: competitor - Lethe"""
        return max(0, self.competitor_p5 - self.lethe_p5)
    
    @property
    def total_sensitivity(self) -> float:
        """Combined counterfactual sensitivity"""
        return np.sqrt(
            self.sensitivity_k2**2 + 
            self.sensitivity_lambda**2 + 
            self.sensitivity_mu**2 + 
            self.sensitivity_r**2 + 
            self.sensitivity_tau**2
        )
    
    @property
    def total_weight(self) -> float:
        """Combined traffic and tenant weight"""
        return self.traffic_weight * self.tenant_weight
    
    @property
    def total_risk(self) -> float:
        """Combined risk factor"""
        return (
            self.kv_prefix_drop_risk + 
            self.ece_drift_risk + 
            self.latency_inflation_risk + 
            self.complexity_risk
        )

@dataclass
class CampaignPriority:
    """Priority score and breakdown for a campaign candidate"""
    slice_candidate: SliceCandidate
    
    # Priority score components
    statistical_component: float  # (ΔP@5/CI_width)²
    sensitivity_component: float  # S
    weight_component: float  # T
    risk_penalty: float  # ρ×R
    
    # Final score
    priority_score: float
    
    # Ranking metadata
    rank: Optional[int] = None
    percentile: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "slice_name": self.slice_candidate.slice_name,
            "budget_tier": self.slice_candidate.budget_tier,
            "domain": self.slice_candidate.domain,
            "complexity": self.slice_candidate.complexity,
            "gap_p5": self.slice_candidate.gap_p5,
            "ci_width": self.slice_candidate.ci_width,
            "statistical_component": self.statistical_component,
            "sensitivity_component": self.sensitivity_component,
            "weight_component": self.weight_component,
            "risk_penalty": self.risk_penalty,
            "priority_score": self.priority_score,
            "rank": self.rank,
            "percentile": self.percentile,
            "sample_size": self.slice_candidate.sample_size
        }

class PriorityScorer:
    """Implements priority scoring for campaign selection"""
    
    def __init__(self, 
                 risk_penalty_weight: float = 0.5,
                 min_ci_width: float = 0.001,
                 min_sample_size: int = 50):
        """
        Initialize priority scorer.
        
        Args:
            risk_penalty_weight: ρ parameter for risk penalty
            min_ci_width: Minimum CI width to avoid division by zero
            min_sample_size: Minimum queries required for statistical validity
        """
        self.risk_penalty_weight = risk_penalty_weight
        self.min_ci_width = min_ci_width
        self.min_sample_size = min_sample_size
        
        logger.info(f"Initialized PriorityScorer with ρ={risk_penalty_weight}")
    
    def score_candidate(self, candidate: SliceCandidate) -> CampaignPriority:
        """
        Compute priority score for a single candidate.
        
        Formula: score = (max(0,ΔP@5)/CI_width)² × S × T - ρ×R
        """
        # Validate candidate
        if candidate.sample_size < self.min_sample_size:
            logger.warning(f"Candidate {candidate.slice_name} has insufficient sample size: {candidate.sample_size}")
        
        # Statistical component: (ΔP@5/CI_width)²
        gap = candidate.gap_p5
        ci_width = max(candidate.ci_width, self.min_ci_width)
        statistical_component = (gap / ci_width) ** 2
        
        # Sensitivity component: S
        sensitivity_component = candidate.total_sensitivity
        
        # Weight component: T  
        weight_component = candidate.total_weight
        
        # Risk penalty: ρ×R
        risk_penalty = self.risk_penalty_weight * candidate.total_risk
        
        # Final score
        priority_score = (
            statistical_component * 
            sensitivity_component * 
            weight_component - 
            risk_penalty
        )
        
        logger.debug(
            f"Scored {candidate.slice_name}: "
            f"gap={gap:.4f}, ci_width={ci_width:.4f}, "
            f"stat={statistical_component:.4f}, sens={sensitivity_component:.4f}, "
            f"weight={weight_component:.4f}, risk_penalty={risk_penalty:.4f}, "
            f"final_score={priority_score:.4f}"
        )
        
        return CampaignPriority(
            slice_candidate=candidate,
            statistical_component=statistical_component,
            sensitivity_component=sensitivity_component,
            weight_component=weight_component,
            risk_penalty=risk_penalty,
            priority_score=priority_score
        )
    
    def score_all_candidates(self, candidates: List[SliceCandidate]) -> List[CampaignPriority]:
        """Score all candidates and rank them by priority"""
        # Score each candidate
        priorities = [self.score_candidate(candidate) for candidate in candidates]
        
        # Sort by priority score (descending)
        priorities.sort(key=lambda p: p.priority_score, reverse=True)
        
        # Add ranking information
        for i, priority in enumerate(priorities):
            priority.rank = i + 1
            priority.percentile = (len(priorities) - i) / len(priorities) * 100
        
        logger.info(f"Scored {len(priorities)} candidates. Top score: {priorities[0].priority_score:.4f}")
        
        return priorities
    
    def select_top_candidates(self, 
                            priorities: List[CampaignPriority],
                            max_candidates: int = 4,
                            min_score_threshold: float = 0.0) -> List[CampaignPriority]:
        """Select top candidates for campaign execution"""
        # Filter by minimum score threshold
        filtered = [p for p in priorities if p.priority_score >= min_score_threshold]
        
        # Take top N candidates
        selected = filtered[:max_candidates]
        
        logger.info(f"Selected {len(selected)} candidates from {len(priorities)} total")
        for i, priority in enumerate(selected):
            logger.info(f"  {i+1}. {priority.slice_candidate.slice_name} "
                       f"(score={priority.priority_score:.4f})")
        
        return selected
    
    def analyze_score_distribution(self, priorities: List[CampaignPriority]) -> Dict[str, Any]:
        """Analyze the distribution of priority scores"""
        scores = [p.priority_score for p in priorities]
        
        analysis = {
            "n_candidates": len(scores),
            "score_stats": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75)
            },
            "component_breakdown": self._analyze_components(priorities),
            "budget_tier_breakdown": self._analyze_by_budget_tier(priorities),
            "domain_breakdown": self._analyze_by_domain(priorities)
        }
        
        return analysis
    
    def _analyze_components(self, priorities: List[CampaignPriority]) -> Dict[str, Any]:
        """Analyze contribution of different score components"""
        components = {
            "statistical": [p.statistical_component for p in priorities],
            "sensitivity": [p.sensitivity_component for p in priorities], 
            "weight": [p.weight_component for p in priorities],
            "risk_penalty": [p.risk_penalty for p in priorities]
        }
        
        return {
            name: {
                "mean": np.mean(values),
                "std": np.std(values),
                "correlation_with_score": np.corrcoef(
                    values, [p.priority_score for p in priorities]
                )[0, 1]
            }
            for name, values in components.items()
        }
    
    def _analyze_by_budget_tier(self, priorities: List[CampaignPriority]) -> Dict[int, Any]:
        """Analyze scores by budget tier"""
        by_tier = {}
        for tier in [8, 15, 30]:
            tier_priorities = [p for p in priorities if p.slice_candidate.budget_tier == tier]
            if tier_priorities:
                scores = [p.priority_score for p in tier_priorities]
                by_tier[tier] = {
                    "count": len(tier_priorities),
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "top_candidate": tier_priorities[0].slice_candidate.slice_name
                }
        
        return by_tier
    
    def _analyze_by_domain(self, priorities: List[CampaignPriority]) -> Dict[str, Any]:
        """Analyze scores by domain"""
        by_domain = {}
        domains = set(p.slice_candidate.domain for p in priorities)
        
        for domain in domains:
            domain_priorities = [p for p in priorities if p.slice_candidate.domain == domain]
            scores = [p.priority_score for p in domain_priorities]
            by_domain[domain] = {
                "count": len(domain_priorities),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "top_candidate": max(domain_priorities, 
                                   key=lambda p: p.priority_score).slice_candidate.slice_name
            }
        
        return by_domain
    
    def export_results(self, 
                      priorities: List[CampaignPriority], 
                      output_path: Path,
                      include_analysis: bool = True) -> None:
        """Export scoring results to files"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export priority scores as CSV
        df = pd.DataFrame([p.to_dict() for p in priorities])
        csv_path = output_path / "priority_scores.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported priority scores to {csv_path}")
        
        # Export detailed analysis
        if include_analysis:
            analysis = self.analyze_score_distribution(priorities)
            analysis_path = output_path / "priority_analysis.json"
            
            import json
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Exported analysis to {analysis_path}")
        
        # Export top candidates summary
        top_candidates = self.select_top_candidates(priorities)
        summary_path = output_path / "top_candidates.json"
        
        summary = {
            "selection_timestamp": pd.Timestamp.now().isoformat(),
            "selection_criteria": {
                "risk_penalty_weight": self.risk_penalty_weight,
                "min_sample_size": self.min_sample_size
            },
            "top_candidates": [p.to_dict() for p in top_candidates]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exported top candidates to {summary_path}")

def load_candidates_from_gap_analysis(gap_analysis_path: str) -> List[SliceCandidate]:
    """
    Load slice candidates from existing Gap→Tune→Verify analysis results.
    
    This integrates with the existing framework by parsing gap analysis outputs
    and converting them to SliceCandidate objects.
    """
    import json
    
    with open(gap_analysis_path, 'r') as f:
        gap_data = json.load(f)
    
    candidates = []
    
    for slice_name, slice_data in gap_data.get('slices', {}).items():
        # Parse slice metadata
        parts = slice_name.split('@')
        if len(parts) != 2:
            logger.warning(f"Skipping slice with invalid name format: {slice_name}")
            continue
        
        domain_complexity = parts[0]
        budget_tier = int(parts[1].rstrip('%'))
        
        # Split domain and complexity
        if '.' in domain_complexity:
            domain, complexity = domain_complexity.split('.', 1)
        else:
            domain = domain_complexity
            complexity = "medium"
        
        # Extract performance metrics
        lethe_p5 = slice_data.get('lethe_p5', 0.0)
        competitor_p5 = slice_data.get('competitor_p5', 0.0)
        ci_width = slice_data.get('ci_width', 0.01)
        
        # Extract or estimate sensitivity values
        sensitivity_data = slice_data.get('sensitivity', {})
        sensitivity_k2 = sensitivity_data.get('k2', 0.1)
        sensitivity_lambda = sensitivity_data.get('lambda', 0.05)
        sensitivity_mu = sensitivity_data.get('mu', 0.03)
        sensitivity_r = sensitivity_data.get('r', 0.08)
        sensitivity_tau = sensitivity_data.get('tau', 0.04)
        
        # Extract traffic weighting
        traffic_weight = slice_data.get('traffic_weight', 1.0)
        tenant_weight = slice_data.get('tenant_weight', 1.0)
        
        # Extract or estimate risk factors
        risk_data = slice_data.get('risk_factors', {})
        kv_prefix_drop_risk = risk_data.get('kv_prefix_drop', 0.02)
        ece_drift_risk = risk_data.get('ece_drift', 0.01)
        latency_inflation_risk = risk_data.get('latency_inflation', 0.05)
        complexity_risk = risk_data.get('complexity', 0.1)
        
        # Sample size and metadata
        sample_size = slice_data.get('sample_size', 100)
        last_updated = slice_data.get('last_updated', pd.Timestamp.now().isoformat())
        
        candidate = SliceCandidate(
            slice_name=slice_name,
            budget_tier=budget_tier,
            domain=domain,
            complexity=complexity,
            lethe_p5=lethe_p5,
            competitor_p5=competitor_p5,
            ci_width=ci_width,
            sensitivity_k2=sensitivity_k2,
            sensitivity_lambda=sensitivity_lambda,
            sensitivity_mu=sensitivity_mu,
            sensitivity_r=sensitivity_r,
            sensitivity_tau=sensitivity_tau,
            traffic_weight=traffic_weight,
            tenant_weight=tenant_weight,
            kv_prefix_drop_risk=kv_prefix_drop_risk,
            ece_drift_risk=ece_drift_risk,
            latency_inflation_risk=latency_inflation_risk,
            complexity_risk=complexity_risk,
            sample_size=sample_size,
            last_updated=last_updated
        )
        
        candidates.append(candidate)
    
    logger.info(f"Loaded {len(candidates)} slice candidates from {gap_analysis_path}")
    return candidates

if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example candidates
    candidates = [
        SliceCandidate(
            slice_name="Zh.QA@8%",
            budget_tier=8,
            domain="Zh",
            complexity="QA",
            lethe_p5=0.65,
            competitor_p5=0.80,
            ci_width=0.02,
            sensitivity_k2=0.12,
            sensitivity_lambda=0.08,
            sensitivity_mu=0.05,
            sensitivity_r=0.10,
            sensitivity_tau=0.06,
            traffic_weight=0.8,
            tenant_weight=1.2,
            kv_prefix_drop_risk=0.02,
            ece_drift_risk=0.01,
            latency_inflation_risk=0.03,
            complexity_risk=0.15,
            sample_size=150,
            last_updated="2025-01-15T10:00:00"
        ),
        SliceCandidate(
            slice_name="JSON.PassKey@15%",
            budget_tier=15,
            domain="JSON",
            complexity="PassKey",
            lethe_p5=0.72,
            competitor_p5=0.87,
            ci_width=0.015,
            sensitivity_k2=0.15,
            sensitivity_lambda=0.10,
            sensitivity_mu=0.08,
            sensitivity_r=0.05,
            sensitivity_tau=0.03,
            traffic_weight=1.0,
            tenant_weight=1.5,
            kv_prefix_drop_risk=0.01,
            ece_drift_risk=0.02,
            latency_inflation_risk=0.04,
            complexity_risk=0.12,
            sample_size=200,
            last_updated="2025-01-15T10:00:00"
        )
    ]
    
    # Score candidates
    scorer = PriorityScorer(risk_penalty_weight=0.5)
    priorities = scorer.score_all_candidates(candidates)
    
    # Display results
    print("\nPriority Scoring Results:")
    print("=" * 50)
    for priority in priorities:
        print(f"Rank {priority.rank}: {priority.slice_candidate.slice_name}")
        print(f"  Score: {priority.priority_score:.4f}")
        print(f"  Components: stat={priority.statistical_component:.4f}, "
              f"sens={priority.sensitivity_component:.4f}, "
              f"weight={priority.weight_component:.4f}, "
              f"penalty={priority.risk_penalty:.4f}")
        print(f"  Gap P@5: {priority.slice_candidate.gap_p5:.4f}")
        print()
    
    # Analyze distribution
    analysis = scorer.analyze_score_distribution(priorities)
    print("Score Distribution Analysis:")
    print(f"  Mean: {analysis['score_stats']['mean']:.4f}")
    print(f"  Std:  {analysis['score_stats']['std']:.4f}")
    print(f"  Range: [{analysis['score_stats']['min']:.4f}, {analysis['score_stats']['max']:.4f}]")