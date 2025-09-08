#!/usr/bin/env python3
"""
Campaign Management System
==========================

Core campaign execution system that orchestrates 1-2 slice campaigns per budget tier
with 12-18 Bayesian Optimization trials. Integrates with existing Gap→Tune→Verify 
framework while adding sophisticated BO-based optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

# Bayesian Optimization components
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    logging.warning("scikit-optimize not available. BO functionality limited.")

# Integration with existing framework
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from priority_scoring import SliceCandidate, CampaignPriority
from analysis.metrics import MetricsCalculator, EvaluationMetrics

logger = logging.getLogger(__name__)

class CampaignStatus(Enum):
    """Campaign execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrialStatus(Enum):
    """Individual trial status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class KnobSpace:
    """Defines the optimization space for a parameter knob"""
    name: str
    knob_type: str  # 'real', 'integer', 'categorical'
    bounds: Union[Tuple[float, float], List[Any]]  # (min, max) or [categories]
    description: str
    risk_level: str = "medium"  # low, medium, high
    
    def to_skopt_dimension(self):
        """Convert to scikit-optimize dimension"""
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for BO")
            
        if self.knob_type == "real":
            return Real(self.bounds[0], self.bounds[1], name=self.name)
        elif self.knob_type == "integer":
            return Integer(self.bounds[0], self.bounds[1], name=self.name)
        elif self.knob_type == "categorical":
            return Categorical(self.bounds, name=self.name)
        else:
            raise ValueError(f"Unknown knob type: {self.knob_type}")

@dataclass
class Trial:
    """Single optimization trial"""
    trial_id: str
    campaign_id: str
    trial_number: int
    
    # Parameter configuration
    parameters: Dict[str, Any]
    
    # Evaluation results
    objective_value: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Execution metadata
    status: TrialStatus = TrialStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Validation results
    gates_passed: Dict[str, bool] = field(default_factory=dict)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value
        }

@dataclass
class CampaignSpec:
    """Campaign specification"""
    name: str
    slice_candidate: SliceCandidate
    
    # Optimization configuration
    knob_spaces: List[KnobSpace]
    objective_function: str  # "delta_p5_per_ms", "delta_p5", etc.
    n_trials: int = 15
    
    # Validation gates
    gates: Dict[str, Any] = field(default_factory=dict)
    
    # Risk management
    validator_fences: Dict[str, Any] = field(default_factory=dict)
    safe_knobs_only: bool = True
    
    # Campaign metadata
    description: str = ""
    expected_improvement: float = 0.0
    risk_assessment: str = "medium"
    
    @property
    def budget_tier(self) -> int:
        """Get budget tier from slice candidate"""
        return self.slice_candidate.budget_tier

@dataclass
class Campaign:
    """Campaign execution state"""
    campaign_id: str
    spec: CampaignSpec
    
    # Execution state
    status: CampaignStatus = CampaignStatus.PENDING
    trials: List[Trial] = field(default_factory=list)
    
    # Optimization state
    best_trial: Optional[Trial] = None
    best_objective: Optional[float] = None
    
    # Timing
    created_time: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    promotion_candidate: Optional[Dict[str, Any]] = None
    promotion_approved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "campaign_id": self.campaign_id,
            "spec": {
                "name": self.spec.name,
                "slice_name": self.spec.slice_candidate.slice_name,
                "budget_tier": self.spec.budget_tier,
                "n_trials": self.spec.n_trials,
                "knob_spaces": [asdict(ks) for ks in self.spec.knob_spaces],
                "gates": self.spec.gates,
                "objective_function": self.spec.objective_function
            },
            "status": self.status.value,
            "n_trials_completed": len([t for t in self.trials if t.status == TrialStatus.COMPLETED]),
            "n_trials_total": self.spec.n_trials,
            "best_objective": self.best_objective,
            "created_time": self.created_time.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "promotion_approved": self.promotion_approved
        }

class CampaignManager:
    """Main campaign orchestrator"""
    
    def __init__(self, 
                 output_dir: str = "./campaigns",
                 max_concurrent_trials: int = 3,
                 evaluator_function: Optional[Callable] = None):
        """
        Initialize campaign manager.
        
        Args:
            output_dir: Directory for campaign artifacts
            max_concurrent_trials: Maximum parallel trials
            evaluator_function: Function to evaluate trial configurations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_trials = max_concurrent_trials
        self.evaluator_function = evaluator_function or self._default_evaluator
        
        # Active campaigns
        self.campaigns: Dict[str, Campaign] = {}
        self.campaign_lock = threading.Lock()
        
        # Execution state
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_trials)
        self.running_futures: Dict[str, Future] = {}
        
        # Metrics
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"Initialized CampaignManager with max_concurrent_trials={max_concurrent_trials}")
    
    def create_campaign(self, spec: CampaignSpec) -> Campaign:
        """Create new campaign from specification"""
        campaign_id = f"campaign_{uuid.uuid4().hex[:8]}"
        
        campaign = Campaign(
            campaign_id=campaign_id,
            spec=spec
        )
        
        with self.campaign_lock:
            self.campaigns[campaign_id] = campaign
        
        # Save campaign specification
        self._save_campaign_state(campaign)
        
        logger.info(f"Created campaign {campaign_id} for slice {spec.slice_candidate.slice_name}")
        return campaign
    
    def start_campaign(self, campaign_id: str) -> None:
        """Start campaign execution"""
        with self.campaign_lock:
            if campaign_id not in self.campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.campaigns[campaign_id]
            if campaign.status != CampaignStatus.PENDING:
                raise ValueError(f"Campaign {campaign_id} status is {campaign.status}, cannot start")
            
            campaign.status = CampaignStatus.RUNNING
            campaign.start_time = datetime.now()
        
        # Submit campaign execution as background task
        future = self.executor.submit(self._execute_campaign, campaign_id)
        self.running_futures[campaign_id] = future
        
        logger.info(f"Started campaign {campaign_id}")
    
    def _execute_campaign(self, campaign_id: str) -> None:
        """Execute campaign with Bayesian Optimization"""
        try:
            campaign = self.campaigns[campaign_id]
            spec = campaign.spec
            
            logger.info(f"Executing campaign {campaign_id} with {spec.n_trials} trials")
            
            if not HAS_SKOPT:
                # Fall back to grid search
                self._execute_grid_search(campaign)
            else:
                # Use Bayesian Optimization
                self._execute_bayesian_optimization(campaign)
                
        except Exception as e:
            logger.error(f"Campaign {campaign_id} failed: {str(e)}")
            with self.campaign_lock:
                campaign = self.campaigns[campaign_id]
                campaign.status = CampaignStatus.FAILED
                campaign.end_time = datetime.now()
            
            self._save_campaign_state(campaign)
        finally:
            # Clean up
            if campaign_id in self.running_futures:
                del self.running_futures[campaign_id]
    
    def _execute_bayesian_optimization(self, campaign: Campaign) -> None:
        """Execute campaign using Bayesian Optimization"""
        spec = campaign.spec
        
        # Convert knob spaces to scikit-optimize dimensions
        dimensions = [ks.to_skopt_dimension() for ks in spec.knob_spaces]
        
        # Define objective function for BO
        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_trial_objective(campaign, params)
        
        # Run BO optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=spec.n_trials,
            acq_func=gaussian_ei,
            n_initial_points=max(3, spec.n_trials // 5),  # 20% random exploration
            random_state=42,
            callback=[self._bo_callback(campaign)]
        )
        
        # Update campaign with final results
        with self.campaign_lock:
            campaign.status = CampaignStatus.COMPLETED
            campaign.end_time = datetime.now()
            
            # Find best trial
            if campaign.trials:
                best_trial = min(campaign.trials, 
                               key=lambda t: t.objective_value if t.objective_value is not None else float('inf'))
                campaign.best_trial = best_trial
                campaign.best_objective = best_trial.objective_value
        
        self._save_campaign_state(campaign)
        logger.info(f"Completed BO campaign {campaign.campaign_id}")
    
    def _bo_callback(self, campaign: Campaign):
        """Callback function for BO progress tracking"""
        def callback(result):
            # Save intermediate state
            self._save_campaign_state(campaign)
            
            # Log progress
            n_completed = len(campaign.trials)
            logger.info(f"Campaign {campaign.campaign_id}: trial {n_completed}/{campaign.spec.n_trials} "
                       f"completed, best objective: {campaign.best_objective}")
        
        return callback
    
    def _execute_grid_search(self, campaign: Campaign) -> None:
        """Fallback grid search when BO unavailable"""
        spec = campaign.spec
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(spec.knob_spaces, spec.n_trials)
        
        for i, params in enumerate(param_grid):
            if campaign.status != CampaignStatus.RUNNING:
                break
                
            trial = Trial(
                trial_id=f"{campaign.campaign_id}_trial_{i+1:03d}",
                campaign_id=campaign.campaign_id,
                trial_number=i + 1,
                parameters=params
            )
            
            # Execute trial
            self._execute_trial(campaign, trial)
        
        # Finalize campaign
        with self.campaign_lock:
            campaign.status = CampaignStatus.COMPLETED
            campaign.end_time = datetime.now()
        
        self._save_campaign_state(campaign)
        logger.info(f"Completed grid search campaign {campaign.campaign_id}")
    
    def _evaluate_trial_objective(self, campaign: Campaign, params: Dict[str, Any]) -> float:
        """Evaluate objective function for a trial"""
        trial_id = f"{campaign.campaign_id}_trial_{len(campaign.trials)+1:03d}"
        
        trial = Trial(
            trial_id=trial_id,
            campaign_id=campaign.campaign_id,
            trial_number=len(campaign.trials) + 1,
            parameters=params
        )
        
        # Execute trial
        self._execute_trial(campaign, trial)
        
        # Return objective value (BO minimizes, so negate for maximization)
        objective_value = trial.objective_value if trial.objective_value is not None else float('inf')
        
        # For maximization objectives (like ΔP@5), negate
        if campaign.spec.objective_function in ["delta_p5", "delta_p5_per_ms"]:
            return -objective_value
        else:
            return objective_value
    
    def _execute_trial(self, campaign: Campaign, trial: Trial) -> None:
        """Execute a single trial"""
        trial.status = TrialStatus.RUNNING
        trial.start_time = datetime.now()
        
        try:
            # Add trial to campaign
            with self.campaign_lock:
                campaign.trials.append(trial)
            
            logger.info(f"Executing trial {trial.trial_id} with params: {trial.parameters}")
            
            # Evaluate configuration
            metrics = self.evaluator_function(trial.parameters, campaign.spec)
            trial.metrics = metrics
            
            # Compute objective value
            trial.objective_value = self._compute_objective_value(metrics, campaign.spec)
            
            # Validate against gates
            gates_passed = self._validate_trial_gates(trial, campaign.spec)
            trial.gates_passed = gates_passed
            
            # Update best trial if this one is better
            if all(gates_passed.values()) and trial.objective_value is not None:
                with self.campaign_lock:
                    if (campaign.best_objective is None or 
                        trial.objective_value > campaign.best_objective):
                        campaign.best_trial = trial
                        campaign.best_objective = trial.objective_value
            
            trial.status = TrialStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Trial {trial.trial_id} failed: {str(e)}")
            trial.status = TrialStatus.FAILED
            trial.error_message = str(e)
        
        finally:
            trial.end_time = datetime.now()
            if trial.start_time:
                trial.duration_seconds = (trial.end_time - trial.start_time).total_seconds()
    
    def _compute_objective_value(self, metrics: Dict[str, float], spec: CampaignSpec) -> float:
        """Compute objective value from trial metrics"""
        if spec.objective_function == "delta_p5":
            return metrics.get("delta_p5", 0.0)
        elif spec.objective_function == "delta_p5_per_ms":
            delta_p5 = metrics.get("delta_p5", 0.0)
            latency_ms = metrics.get("latency_p95", 1.0)
            return delta_p5 / latency_ms if latency_ms > 0 else 0.0
        else:
            # Default to delta_p5
            return metrics.get("delta_p5", 0.0)
    
    def _validate_trial_gates(self, trial: Trial, spec: CampaignSpec) -> Dict[str, bool]:
        """Validate trial against campaign gates"""
        gates_passed = {}
        
        for gate_name, gate_spec in spec.gates.items():
            try:
                if gate_name == "min_delta_p5":
                    gates_passed[gate_name] = trial.metrics.get("delta_p5", 0.0) >= gate_spec
                elif gate_name == "max_latency_increase":
                    gates_passed[gate_name] = trial.metrics.get("latency_delta", 0.0) <= gate_spec
                elif gate_name == "max_kv_drop":
                    gates_passed[gate_name] = trial.metrics.get("kv_drop", 0.0) <= gate_spec
                elif gate_name == "max_ece_drift":
                    gates_passed[gate_name] = trial.metrics.get("ece_drift", 0.0) <= gate_spec
                else:
                    # Generic numeric gate
                    metric_value = trial.metrics.get(gate_name, 0.0)
                    if isinstance(gate_spec, dict):
                        min_val = gate_spec.get("min", float("-inf"))
                        max_val = gate_spec.get("max", float("inf"))
                        gates_passed[gate_name] = min_val <= metric_value <= max_val
                    else:
                        gates_passed[gate_name] = metric_value >= gate_spec
                        
            except Exception as e:
                logger.warning(f"Failed to evaluate gate {gate_name}: {str(e)}")
                gates_passed[gate_name] = False
        
        return gates_passed
    
    def _generate_parameter_grid(self, knob_spaces: List[KnobSpace], max_trials: int) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search"""
        # Simple grid generation - for production, use more sophisticated sampling
        grid_points = []
        
        for knob in knob_spaces:
            if knob.knob_type == "real":
                min_val, max_val = knob.bounds
                points = np.linspace(min_val, max_val, min(5, max_trials // len(knob_spaces)))
            elif knob.knob_type == "integer":
                min_val, max_val = knob.bounds
                points = np.linspace(min_val, max_val, min(max_val - min_val + 1, 5), dtype=int)
            elif knob.knob_type == "categorical":
                points = knob.bounds[:min(len(knob.bounds), 5)]
            else:
                points = [0.0]  # Default fallback
            
            grid_points.append((knob.name, points))
        
        # Generate combinations
        param_grid = []
        import itertools
        
        names, point_sets = zip(*grid_points)
        for combination in itertools.product(*point_sets):
            param_dict = dict(zip(names, combination))
            param_grid.append(param_dict)
            
            if len(param_grid) >= max_trials:
                break
        
        return param_grid[:max_trials]
    
    def _default_evaluator(self, parameters: Dict[str, Any], spec: CampaignSpec) -> Dict[str, float]:
        """Default evaluator - returns mock metrics for testing"""
        # This would be replaced with actual model evaluation
        logger.warning(f"Using mock evaluator for parameters: {parameters}")
        
        # Mock performance with some noise
        base_p5 = spec.slice_candidate.lethe_p5
        noise = np.random.normal(0, 0.01)
        delta_p5 = max(0, 0.05 + noise)  # Mock improvement
        
        return {
            "delta_p5": delta_p5,
            "latency_p95": 100 + np.random.normal(0, 5),
            "latency_delta": np.random.normal(0, 2),
            "kv_drop": max(0, np.random.normal(0.01, 0.005)),
            "ece_drift": max(0, np.random.normal(0.01, 0.003)),
            "memory_mb": 512 + np.random.normal(0, 50)
        }
    
    def _save_campaign_state(self, campaign: Campaign) -> None:
        """Save campaign state to disk"""
        campaign_dir = self.output_dir / campaign.campaign_id
        campaign_dir.mkdir(exist_ok=True)
        
        # Save campaign metadata
        metadata_path = campaign_dir / "campaign.json"
        with open(metadata_path, 'w') as f:
            json.dump(campaign.to_dict(), f, indent=2)
        
        # Save trials
        if campaign.trials:
            trials_path = campaign_dir / "trials.json"
            trials_data = [trial.to_dict() for trial in campaign.trials]
            with open(trials_path, 'w') as f:
                json.dump(trials_data, f, indent=2)
    
    def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """Get current campaign status"""
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.campaigns[campaign_id]
        
        return {
            "campaign_id": campaign_id,
            "status": campaign.status.value,
            "slice_name": campaign.spec.slice_candidate.slice_name,
            "progress": {
                "trials_completed": len([t for t in campaign.trials if t.status == TrialStatus.COMPLETED]),
                "trials_total": campaign.spec.n_trials,
                "trials_failed": len([t for t in campaign.trials if t.status == TrialStatus.FAILED])
            },
            "best_objective": campaign.best_objective,
            "runtime_minutes": (
                (datetime.now() - campaign.start_time).total_seconds() / 60 
                if campaign.start_time else 0
            )
        }
    
    def list_campaigns(self) -> List[Dict[str, Any]]:
        """List all campaigns"""
        return [self.get_campaign_status(cid) for cid in self.campaigns.keys()]
    
    def stop_campaign(self, campaign_id: str) -> None:
        """Stop running campaign"""
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        with self.campaign_lock:
            campaign = self.campaigns[campaign_id]
            if campaign.status == CampaignStatus.RUNNING:
                campaign.status = CampaignStatus.CANCELLED
                campaign.end_time = datetime.now()
        
        # Cancel future if running
        if campaign_id in self.running_futures:
            future = self.running_futures[campaign_id]
            future.cancel()
            del self.running_futures[campaign_id]
        
        self._save_campaign_state(campaign)
        logger.info(f"Stopped campaign {campaign_id}")
    
    def get_campaign_results(self, campaign_id: str) -> Dict[str, Any]:
        """Get detailed campaign results"""
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.campaigns[campaign_id]
        
        # Analyze trial results
        completed_trials = [t for t in campaign.trials if t.status == TrialStatus.COMPLETED]
        successful_trials = [t for t in completed_trials if all(t.gates_passed.values())]
        
        results = {
            "campaign_id": campaign_id,
            "campaign": campaign.to_dict(),
            "summary": {
                "total_trials": len(campaign.trials),
                "completed_trials": len(completed_trials),
                "successful_trials": len(successful_trials),
                "success_rate": len(successful_trials) / len(completed_trials) if completed_trials else 0,
                "best_objective": campaign.best_objective
            },
            "trials": [trial.to_dict() for trial in campaign.trials],
            "best_trial": campaign.best_trial.to_dict() if campaign.best_trial else None
        }
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown campaign manager"""
        logger.info("Shutting down campaign manager...")
        
        # Stop all running campaigns
        for campaign_id in list(self.campaigns.keys()):
            if self.campaigns[campaign_id].status == CampaignStatus.RUNNING:
                self.stop_campaign(campaign_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Campaign manager shutdown complete")

if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from priority_scoring import SliceCandidate
    
    # Create example campaign
    candidate = SliceCandidate(
        slice_name="test.campaign@15%",
        budget_tier=15,
        domain="test",
        complexity="campaign",
        lethe_p5=0.70,
        competitor_p5=0.85,
        ci_width=0.02,
        sensitivity_k2=0.10,
        sensitivity_lambda=0.08,
        sensitivity_mu=0.05,
        sensitivity_r=0.06,
        sensitivity_tau=0.04,
        traffic_weight=1.0,
        tenant_weight=1.0,
        kv_prefix_drop_risk=0.02,
        ece_drift_risk=0.01,
        latency_inflation_risk=0.03,
        complexity_risk=0.10,
        sample_size=200,
        last_updated="2025-01-15T10:00:00"
    )
    
    # Define knob spaces
    knob_spaces = [
        KnobSpace("lambda", "real", (0.0, 0.2), "Hybrid weighting parameter"),
        KnobSpace("K2", "integer", (10, 50), "Rerank top-K parameter"),
        KnobSpace("r", "integer", (8, 24), "DPP rank parameter")
    ]
    
    # Create campaign spec
    spec = CampaignSpec(
        name="Test Campaign",
        slice_candidate=candidate,
        knob_spaces=knob_spaces,
        objective_function="delta_p5",
        n_trials=8,
        gates={
            "min_delta_p5": 0.02,
            "max_latency_increase": 5.0,
            "max_kv_drop": 0.03
        }
    )
    
    # Create and run campaign
    manager = CampaignManager(output_dir="./test_campaigns")
    campaign = manager.create_campaign(spec)
    
    print(f"Created campaign: {campaign.campaign_id}")
    print(f"Status: {campaign.status}")
    
    # Start campaign
    manager.start_campaign(campaign.campaign_id)
    
    # Wait for completion (in practice, would monitor asynchronously)
    import time
    while manager.campaigns[campaign.campaign_id].status == CampaignStatus.RUNNING:
        time.sleep(2)
        status = manager.get_campaign_status(campaign.campaign_id)
        print(f"Progress: {status['progress']['trials_completed']}/{status['progress']['trials_total']}")
    
    # Get results
    results = manager.get_campaign_results(campaign.campaign_id)
    print(f"Campaign completed with {results['summary']['successful_trials']} successful trials")
    if results['best_trial']:
        print(f"Best objective: {results['best_trial']['objective_value']}")
        print(f"Best parameters: {results['best_trial']['parameters']}")
    
    manager.shutdown()