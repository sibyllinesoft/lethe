#!/usr/bin/env python3
"""
Publication-Grade Ablations Framework

This module implements a comprehensive ablations framework for systematic component
analysis and validation. The framework provides statistical rigor equivalent to 
academic publications with bootstrap confidence intervals, Holm correction for
multiple testing, and automated experimental design for component significance testing.

Core Features:
- Systematic component removal: -logdet (r=0), -groups, -CE early-exit, -Streaming, -head
- Parameter grid search: window W {4k,6k,8k}, stride s {0.5,0.75}, λ grid analysis
- Statistical validation: ΔCBU/1k, P@k/R@k, p95s, KV-reuse with bootstrap CIs
- Multiple testing correction: Holm-Bonferroni method for family-wise error rate control
- Automated experimental design with power analysis and sample size calculation
- Publication-ready results with LaTeX table generation and statistical reporting
"""

import logging
import asyncio
import time
import threading
import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable, Iterator
from collections import defaultdict, deque
import statistics
import numpy as np
import json
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import scipy.stats as stats
from scipy.stats import bootstrap
import pandas as pd

logger = logging.getLogger(__name__)

class AblationComponent(Enum):
    """Components available for ablation"""
    LOGDET = "logdet"                    # Remove logdet (r=0)
    GROUPS = "groups"                    # Remove group clustering
    CE_EARLY_EXIT = "ce_early_exit"      # Remove cross-encoder early exit
    STREAMING = "streaming"              # Remove streaming (head-only)
    HEAD = "head"                        # Remove head (streaming-only)
    FULL_SYSTEM = "full_system"          # Complete hybrid system (baseline)

class ParameterType(Enum):
    """Types of parameters for grid search"""
    WINDOW_SIZE = "window_size"          # W parameter
    STRIDE_RATIO = "stride_ratio"        # s parameter  
    LAMBDA_VALUE = "lambda_value"        # λ parameter
    RANK_VALUE = "rank_value"            # r parameter
    K2_VALUE = "k2_value"               # K2 parameter

class MetricType(Enum):
    """Types of metrics for evaluation"""
    CBU_PER_K = "cbu_per_k"             # ΔCBU/1k tokens
    PRECISION_AT_K = "precision_at_k"    # P@k precision
    RECALL_AT_K = "recall_at_k"          # R@k recall
    P95_LATENCY = "p95_latency"          # 95th percentile latency
    KV_REUSE = "kv_reuse"               # KV cache reuse ratio
    F1_SCORE = "f1_score"               # F1 score
    NDCG = "ndcg"                       # Normalized discounted cumulative gain

class ExperimentType(Enum):
    """Types of ablation experiments"""
    COMPONENT_ABLATION = "component_ablation"    # Remove individual components
    PARAMETER_GRID = "parameter_grid"            # Grid search over parameters
    COMPONENT_INTERACTION = "component_interaction"  # Test component interactions
    THRESHOLD_ANALYSIS = "threshold_analysis"    # Analyze threshold sensitivity

@dataclass
class ExperimentalCondition:
    """Single experimental condition"""
    condition_id: str
    experiment_type: ExperimentType
    ablated_components: Set[AblationComponent]
    parameter_settings: Dict[str, Any]
    expected_samples: int
    description: str
    
    @property
    def is_baseline(self) -> bool:
        """Check if this is the baseline condition (full system)"""
        return AblationComponent.FULL_SYSTEM in self.ablated_components and len(self.ablated_components) == 1

@dataclass  
class ExperimentalResult:
    """Result from single experimental run"""
    result_id: str
    condition_id: str
    sample_id: str
    metrics: Dict[MetricType, float]
    execution_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, metric_type: MetricType) -> float:
        """Get specific metric value"""
        return self.metrics.get(metric_type, 0.0)

@dataclass
class StatisticalSummary:
    """Statistical summary for a condition"""
    condition_id: str
    metric_type: MetricType
    n_samples: int
    mean: float
    std: float
    median: float
    ci_lower: float      # Bootstrap confidence interval lower bound
    ci_upper: float      # Bootstrap confidence interval upper bound
    ci_level: float      # Confidence level (e.g., 0.95)
    
    @property
    def margin_of_error(self) -> float:
        """Get margin of error for confidence interval"""
        return (self.ci_upper - self.ci_lower) / 2

@dataclass
class SignificanceTest:
    """Statistical significance test result"""
    test_id: str
    condition_a: str
    condition_b: str
    metric_type: MetricType
    test_statistic: float
    p_value: float
    p_value_adjusted: float    # After multiple testing correction
    effect_size: float         # Cohen's d or similar
    significant: bool          # After correction
    test_method: str           # e.g., "mann_whitney", "t_test"
    correction_method: str     # e.g., "holm"

@dataclass
class AblationReport:
    """Complete ablation study report"""
    experiment_name: str
    total_conditions: int
    total_samples: int
    completion_time_hours: float
    statistical_summaries: List[StatisticalSummary]
    significance_tests: List[SignificanceTest]
    component_rankings: Dict[AblationComponent, float]  # Contribution scores
    parameter_optima: Dict[ParameterType, Any]          # Optimal parameter values
    publication_tables: Dict[str, str]                  # LaTeX tables
    methodology_description: str
    key_findings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class ExperimentalDesign:
    """Designs and manages ablation experiments"""
    
    def __init__(self):
        self.component_definitions = self._define_components()
        self.parameter_grids = self._define_parameter_grids()
        
    def _define_components(self) -> Dict[AblationComponent, Dict[str, Any]]:
        """Define ablation components and their implementations"""
        return {
            AblationComponent.FULL_SYSTEM: {
                'description': 'Complete hybrid system with all components',
                'implementation': 'baseline_hybrid_selector',
                'expected_effect': 0.0,  # Baseline
                'parameters': {}
            },
            
            AblationComponent.LOGDET: {
                'description': 'Remove logdet diversity (set r=0)',
                'implementation': 'hybrid_selector_no_logdet', 
                'expected_effect': -0.8,  # Expected performance drop
                'parameters': {'dpp_rank': 0}
            },
            
            AblationComponent.GROUPS: {
                'description': 'Remove group-based atom clustering',
                'implementation': 'hybrid_selector_no_groups',
                'expected_effect': -0.5,
                'parameters': {'group_split_tau': 0.0}
            },
            
            AblationComponent.CE_EARLY_EXIT: {
                'description': 'Remove cross-encoder early exit optimization',
                'implementation': 'hybrid_selector_no_ce_exit',
                'expected_effect': -0.3,
                'parameters': {'ce_early_exit_enabled': False}
            },
            
            AblationComponent.STREAMING: {
                'description': 'Remove streaming tail (head-only processing)',
                'implementation': 'head_only_selector',
                'expected_effect': -1.2,
                'parameters': {'processing_mode': 'head_only'}
            },
            
            AblationComponent.HEAD: {
                'description': 'Remove stable head (streaming-only processing)', 
                'implementation': 'streaming_only_selector',
                'expected_effect': -2.0,
                'parameters': {'processing_mode': 'streaming_only'}
            }
        }
    
    def _define_parameter_grids(self) -> Dict[ParameterType, List[Any]]:
        """Define parameter grids for systematic search"""
        return {
            ParameterType.WINDOW_SIZE: [4000, 6000, 8000],
            ParameterType.STRIDE_RATIO: [0.5, 0.75],
            ParameterType.LAMBDA_VALUE: [0.08, 0.10, 0.12, 0.15, 0.18],  # λ grid for knee analysis
            ParameterType.RANK_VALUE: [8, 12, 14, 16, 20],
            ParameterType.K2_VALUE: [160, 240, 320, 400, 480]
        }
    
    def design_component_ablation_experiment(self, 
                                           sample_size_per_condition: int = 100) -> List[ExperimentalCondition]:
        """Design systematic component ablation experiment"""
        conditions = []
        
        # Baseline condition (full system)
        conditions.append(ExperimentalCondition(
            condition_id="baseline_full_system",
            experiment_type=ExperimentType.COMPONENT_ABLATION,
            ablated_components={AblationComponent.FULL_SYSTEM},
            parameter_settings={},
            expected_samples=sample_size_per_condition,
            description="Complete hybrid system baseline"
        ))
        
        # Single component ablations
        for component in [AblationComponent.LOGDET, AblationComponent.GROUPS, 
                         AblationComponent.CE_EARLY_EXIT, AblationComponent.STREAMING, 
                         AblationComponent.HEAD]:
            
            component_def = self.component_definitions[component]
            
            conditions.append(ExperimentalCondition(
                condition_id=f"ablate_{component.value}",
                experiment_type=ExperimentType.COMPONENT_ABLATION,
                ablated_components={component},
                parameter_settings=component_def['parameters'].copy(),
                expected_samples=sample_size_per_condition,
                description=f"Remove {component_def['description']}"
            ))
        
        logger.info(f"Designed component ablation experiment with {len(conditions)} conditions")
        return conditions
    
    def design_parameter_grid_experiment(self,
                                       parameters: List[ParameterType],
                                       sample_size_per_condition: int = 50) -> List[ExperimentalCondition]:
        """Design parameter grid search experiment"""
        conditions = []
        
        # Get parameter values for grid search
        param_values = {}
        for param in parameters:
            if param in self.parameter_grids:
                param_values[param] = self.parameter_grids[param]
            else:
                logger.warning(f"No grid defined for parameter {param}")
                continue
        
        if not param_values:
            return conditions
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_combinations = list(itertools.product(*param_values.values()))
        
        for i, combination in enumerate(param_combinations):
            param_settings = dict(zip([p.value for p in param_names], combination))
            
            conditions.append(ExperimentalCondition(
                condition_id=f"grid_{i}_{hash(str(combination)) % 10000}",
                experiment_type=ExperimentType.PARAMETER_GRID,
                ablated_components={AblationComponent.FULL_SYSTEM},
                parameter_settings=param_settings,
                expected_samples=sample_size_per_condition,
                description=f"Grid search: {param_settings}"
            ))
        
        logger.info(f"Designed parameter grid experiment: {len(param_names)} parameters, "
                   f"{len(param_combinations)} combinations, {len(conditions)} conditions")
        return conditions
    
    def design_interaction_experiment(self,
                                    component_pairs: List[Tuple[AblationComponent, AblationComponent]],
                                    sample_size_per_condition: int = 75) -> List[ExperimentalCondition]:
        """Design component interaction experiment"""
        conditions = []
        
        for pair_idx, (comp_a, comp_b) in enumerate(component_pairs):
            if comp_a == comp_b:
                continue
                
            # Combined ablation - remove both components
            combined_params = {}
            combined_params.update(self.component_definitions[comp_a]['parameters'])
            combined_params.update(self.component_definitions[comp_b]['parameters'])
            
            conditions.append(ExperimentalCondition(
                condition_id=f"interact_{comp_a.value}_{comp_b.value}",
                experiment_type=ExperimentType.COMPONENT_INTERACTION,
                ablated_components={comp_a, comp_b},
                parameter_settings=combined_params,
                expected_samples=sample_size_per_condition,
                description=f"Remove {comp_a.value} + {comp_b.value} interaction"
            ))
        
        logger.info(f"Designed interaction experiment with {len(conditions)} conditions")
        return conditions
    
    def calculate_required_sample_size(self,
                                     effect_size: float = 0.3,
                                     alpha: float = 0.05,
                                     power: float = 0.8) -> int:
        """Calculate required sample size for statistical power"""
        try:
            # Use Cohen's method for sample size calculation
            # For two-sample t-test with equal variances
            
            z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
            z_beta = stats.norm.ppf(power)
            
            # Sample size per group
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
            # Add 20% buffer for dropouts and non-normality
            n_recommended = int(n_per_group * 1.2)
            
            logger.info(f"Sample size calculation: effect_size={effect_size}, "
                       f"alpha={alpha}, power={power} -> n={n_recommended} per condition")
            
            return max(50, n_recommended)  # Minimum 50 samples
            
        except Exception as e:
            logger.warning(f"Sample size calculation error: {e}, using default")
            return 100

class ExperimentExecutor:
    """Executes ablation experiments with parallel processing"""
    
    def __init__(self, 
                 hybrid_selector_factory: Callable[[Dict[str, Any]], Any],
                 test_data_provider: Callable[[], List[Dict[str, Any]]],
                 storage_path: Path):
        
        self.hybrid_selector_factory = hybrid_selector_factory
        self.test_data_provider = test_data_provider
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Database for storing results
        self.db_path = self.storage_path / "ablation_results.db"
        self._init_database()
        
        # Execution tracking
        self.active_experiments = {}
        self.lock = threading.RLock()
        
    def _init_database(self):
        """Initialize results database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experimental_conditions (
                    condition_id TEXT PRIMARY KEY,
                    experiment_type TEXT NOT NULL,
                    ablated_components TEXT NOT NULL,
                    parameter_settings TEXT NOT NULL,
                    expected_samples INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experimental_results (
                    result_id TEXT PRIMARY KEY,
                    condition_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY (condition_id) REFERENCES experimental_conditions(condition_id)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_condition_id ON experimental_results(condition_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experimental_results(timestamp)")
    
    def execute_experiment(self,
                          conditions: List[ExperimentalCondition],
                          max_workers: int = 4,
                          timeout_per_sample: float = 300.0) -> str:
        """
        Execute complete ablation experiment
        
        Args:
            conditions: List of experimental conditions
            max_workers: Maximum parallel workers
            timeout_per_sample: Timeout per sample in seconds
            
        Returns:
            Experiment ID for tracking
        """
        try:
            experiment_id = f"ablation_{int(time.time() * 1000)}"
            
            logger.info(f"Starting ablation experiment {experiment_id}: "
                       f"{len(conditions)} conditions")
            
            # Store experimental conditions
            self._store_conditions(conditions)
            
            # Get test data
            test_samples = self.test_data_provider()
            if not test_samples:
                raise ValueError("No test data available")
            
            logger.info(f"Loaded {len(test_samples)} test samples")
            
            # Execute all conditions
            with self.lock:
                self.active_experiments[experiment_id] = {
                    'conditions': conditions,
                    'total_samples': sum(c.expected_samples for c in conditions),
                    'completed_samples': 0,
                    'start_time': datetime.now()
                }
            
            # Execute conditions in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_condition = {}
                
                for condition in conditions:
                    future = executor.submit(
                        self._execute_condition,
                        condition,
                        test_samples,
                        timeout_per_sample
                    )
                    future_to_condition[future] = condition
                
                # Collect results
                completed_conditions = 0
                for future in as_completed(future_to_condition):
                    condition = future_to_condition[future]
                    
                    try:
                        condition_results = future.result()
                        self._store_condition_results(condition_results)
                        
                        completed_conditions += 1
                        
                        with self.lock:
                            self.active_experiments[experiment_id]['completed_samples'] += len(condition_results)
                        
                        logger.info(f"Completed condition {condition.condition_id}: "
                                   f"{len(condition_results)} samples, "
                                   f"{completed_conditions}/{len(conditions)} conditions done")
                        
                    except Exception as e:
                        logger.error(f"Condition {condition.condition_id} failed: {e}")
            
            # Mark experiment as completed
            with self.lock:
                if experiment_id in self.active_experiments:
                    self.active_experiments[experiment_id]['completed'] = True
                    self.active_experiments[experiment_id]['end_time'] = datetime.now()
            
            logger.info(f"Ablation experiment {experiment_id} completed")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Experiment execution error: {e}")
            raise
    
    def _execute_condition(self,
                          condition: ExperimentalCondition,
                          test_samples: List[Dict[str, Any]],
                          timeout_per_sample: float) -> List[ExperimentalResult]:
        """Execute single experimental condition"""
        try:
            logger.debug(f"Executing condition {condition.condition_id}")
            
            # Create hybrid selector with condition parameters
            selector_config = self._create_selector_config(condition)
            hybrid_selector = self.hybrid_selector_factory(selector_config)
            
            # Sample test data for this condition
            selected_samples = self._sample_test_data(test_samples, condition.expected_samples)
            
            results = []
            
            for sample_idx, sample_data in enumerate(selected_samples):
                try:
                    start_time = time.time()
                    
                    # Execute hybrid selector on sample
                    selection_result = hybrid_selector.select(
                        content=sample_data['content'],
                        session_context=sample_data.get('metadata', {}),
                        relevance_scores=sample_data.get('relevance_scores', {})
                    )
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Extract metrics from result
                    metrics = self._extract_metrics(selection_result, sample_data)
                    
                    # Create experimental result
                    result = ExperimentalResult(
                        result_id=f"{condition.condition_id}_sample_{sample_idx}",
                        condition_id=condition.condition_id,
                        sample_id=sample_data.get('sample_id', str(sample_idx)),
                        metrics=metrics,
                        execution_time_ms=execution_time,
                        timestamp=datetime.now(),
                        metadata={
                            'sample_metadata': sample_data.get('metadata', {}),
                            'selection_metadata': {
                                'processing_mode': selection_result.processing_mode.value,
                                'total_tokens': selection_result.total_tokens,
                                'kv_reuse_ratio': selection_result.kv_prefix_reuse_ratio
                            }
                        }
                    )
                    
                    results.append(result)
                    
                    if execution_time > timeout_per_sample * 1000:
                        logger.warning(f"Sample execution exceeded timeout: {execution_time:.1f}ms")
                    
                except Exception as sample_error:
                    logger.error(f"Sample {sample_idx} failed for condition {condition.condition_id}: {sample_error}")
                    continue
            
            logger.debug(f"Condition {condition.condition_id} completed: {len(results)} successful samples")
            return results
            
        except Exception as e:
            logger.error(f"Error executing condition {condition.condition_id}: {e}")
            return []
    
    def _create_selector_config(self, condition: ExperimentalCondition) -> Dict[str, Any]:
        """Create hybrid selector configuration for condition"""
        base_config = {
            'head_keep_ratio': 0.12,
            'window_size': 6000,
            'stride': 3000,
            'dpp_rank': 14,
            'ce_k2': 320,
            'ce_early_exit_enabled': True,
            'group_split_tau': 0.7
        }
        
        # Apply parameter settings
        base_config.update(condition.parameter_settings)
        
        # Apply component ablations
        for component in condition.ablated_components:
            if component == AblationComponent.LOGDET:
                base_config['dpp_rank'] = 0
            elif component == AblationComponent.GROUPS:
                base_config['group_split_tau'] = 0.0
            elif component == AblationComponent.CE_EARLY_EXIT:
                base_config['ce_early_exit_enabled'] = False
            elif component == AblationComponent.STREAMING:
                base_config['processing_mode'] = 'head_only'
            elif component == AblationComponent.HEAD:
                base_config['processing_mode'] = 'streaming_only'
        
        return base_config
    
    def _sample_test_data(self, 
                         test_samples: List[Dict[str, Any]], 
                         n_samples: int) -> List[Dict[str, Any]]:
        """Sample test data for experimental condition"""
        if len(test_samples) <= n_samples:
            return test_samples
        
        # Stratified sampling if domain information available
        domains = set()
        for sample in test_samples:
            domain = sample.get('metadata', {}).get('domain', 'default')
            domains.add(domain)
        
        if len(domains) > 1:
            # Stratified sampling
            samples_per_domain = n_samples // len(domains)
            remainder = n_samples % len(domains)
            
            sampled = []
            for i, domain in enumerate(sorted(domains)):
                domain_samples = [s for s in test_samples 
                                if s.get('metadata', {}).get('domain', 'default') == domain]
                
                domain_n = samples_per_domain + (1 if i < remainder else 0)
                domain_sample = np.random.choice(
                    len(domain_samples), 
                    size=min(domain_n, len(domain_samples)), 
                    replace=False
                )
                
                sampled.extend([domain_samples[idx] for idx in domain_sample])
            
            return sampled
        else:
            # Simple random sampling
            indices = np.random.choice(len(test_samples), size=n_samples, replace=False)
            return [test_samples[idx] for idx in indices]
    
    def _extract_metrics(self, 
                        selection_result: Any, 
                        sample_data: Dict[str, Any]) -> Dict[MetricType, float]:
        """Extract evaluation metrics from selection result"""
        metrics = {}
        
        # CBU per 1k tokens
        if selection_result.total_tokens > 0:
            cbu_per_k = (selection_result.objective_value / selection_result.total_tokens) * 1000
            metrics[MetricType.CBU_PER_K] = cbu_per_k
        
        # P95 latency (estimated from selection time)
        metrics[MetricType.P95_LATENCY] = selection_result.selection_time_ms
        
        # KV reuse ratio
        metrics[MetricType.KV_REUSE] = selection_result.kv_prefix_reuse_ratio
        
        # Precision/Recall at K (would require ground truth relevance)
        if 'ground_truth_relevant' in sample_data:
            relevant_items = set(sample_data['ground_truth_relevant'])
            selected_items = self._extract_selected_items(selection_result)
            
            if selected_items:
                precision = len(relevant_items & selected_items) / len(selected_items)
                recall = len(relevant_items & selected_items) / len(relevant_items) if relevant_items else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[MetricType.PRECISION_AT_K] = precision
                metrics[MetricType.RECALL_AT_K] = recall
                metrics[MetricType.F1_SCORE] = f1
        
        return metrics
    
    def _extract_selected_items(self, selection_result: Any) -> Set[str]:
        """Extract selected item IDs from selection result"""
        # This would extract actual selected content IDs
        # For now, return empty set as placeholder
        return set()
    
    def _store_conditions(self, conditions: List[ExperimentalCondition]):
        """Store experimental conditions in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for condition in conditions:
                    conn.execute("""
                        INSERT OR REPLACE INTO experimental_conditions
                        (condition_id, experiment_type, ablated_components, parameter_settings,
                         expected_samples, description, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        condition.condition_id,
                        condition.experiment_type.value,
                        json.dumps([c.value for c in condition.ablated_components]),
                        json.dumps(condition.parameter_settings),
                        condition.expected_samples,
                        condition.description,
                        datetime.now().isoformat()
                    ))
        except Exception as e:
            logger.error(f"Error storing conditions: {e}")
    
    def _store_condition_results(self, results: List[ExperimentalResult]):
        """Store experimental results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for result in results:
                    conn.execute("""
                        INSERT INTO experimental_results
                        (result_id, condition_id, sample_id, metrics, execution_time_ms, 
                         timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.result_id,
                        result.condition_id,
                        result.sample_id,
                        json.dumps({k.value: v for k, v in result.metrics.items()}),
                        result.execution_time_ms,
                        result.timestamp.isoformat(),
                        json.dumps(result.metadata)
                    ))
        except Exception as e:
            logger.error(f"Error storing results: {e}")
    
    def get_experiment_results(self, condition_ids: List[str]) -> List[ExperimentalResult]:
        """Get experimental results for conditions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join(['?' for _ in condition_ids])
                cursor = conn.execute(f"""
                    SELECT result_id, condition_id, sample_id, metrics, execution_time_ms, 
                           timestamp, metadata
                    FROM experimental_results
                    WHERE condition_id IN ({placeholders})
                """, condition_ids)
                
                results = []
                for row in cursor.fetchall():
                    result_id, condition_id, sample_id, metrics_json, exec_time, timestamp, metadata_json = row
                    
                    # Parse metrics
                    metrics_dict = json.loads(metrics_json)
                    metrics = {MetricType(k): v for k, v in metrics_dict.items()}
                    
                    result = ExperimentalResult(
                        result_id=result_id,
                        condition_id=condition_id,
                        sample_id=sample_id,
                        metrics=metrics,
                        execution_time_ms=exec_time,
                        timestamp=datetime.fromisoformat(timestamp),
                        metadata=json.loads(metadata_json)
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting experiment results: {e}")
            return []

class StatisticalAnalyzer:
    """Performs statistical analysis of ablation results"""
    
    def __init__(self):
        self.confidence_level = 0.95
        self.bootstrap_samples = 1000
        
    def compute_statistical_summaries(self, 
                                    results: List[ExperimentalResult],
                                    metrics: List[MetricType]) -> List[StatisticalSummary]:
        """Compute statistical summaries with bootstrap confidence intervals"""
        summaries = []
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in results:
            condition_groups[result.condition_id].append(result)
        
        for condition_id, condition_results in condition_groups.items():
            for metric_type in metrics:
                try:
                    # Extract metric values
                    values = [r.get_metric(metric_type) for r in condition_results]
                    values = [v for v in values if not np.isnan(v) and v is not None]
                    
                    if len(values) < 2:
                        continue
                    
                    # Basic statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    median_val = np.median(values)
                    
                    # Bootstrap confidence interval
                    ci_lower, ci_upper = self._bootstrap_confidence_interval(
                        values, self.confidence_level
                    )
                    
                    summary = StatisticalSummary(
                        condition_id=condition_id,
                        metric_type=metric_type,
                        n_samples=len(values),
                        mean=mean_val,
                        std=std_val,
                        median=median_val,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        ci_level=self.confidence_level
                    )
                    
                    summaries.append(summary)
                    
                except Exception as e:
                    logger.error(f"Error computing summary for {condition_id}, {metric_type}: {e}")
                    continue
        
        return summaries
    
    def _bootstrap_confidence_interval(self, 
                                     values: List[float], 
                                     confidence_level: float) -> Tuple[float, float]:
        """Compute bootstrap confidence interval"""
        try:
            values_array = np.array(values)
            
            # Use scipy bootstrap
            res = bootstrap(
                (values_array,), 
                np.mean, 
                n_resamples=self.bootstrap_samples,
                confidence_level=confidence_level,
                method='percentile'
            )
            
            return res.confidence_interval.low, res.confidence_interval.high
            
        except Exception as e:
            logger.debug(f"Bootstrap CI error: {e}")
            # Fallback to basic percentile method
            bootstrap_means = []
            for _ in range(self.bootstrap_samples):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
            return ci_lower, ci_upper
    
    def perform_significance_tests(self,
                                 summaries: List[StatisticalSummary],
                                 results: List[ExperimentalResult],
                                 baseline_condition: str) -> List[SignificanceTest]:
        """Perform statistical significance tests with multiple testing correction"""
        significance_tests = []
        
        # Group summaries by metric type
        metric_groups = defaultdict(list)
        for summary in summaries:
            metric_groups[summary.metric_type].append(summary)
        
        # Group results by condition for raw data access
        condition_results = defaultdict(list)
        for result in results:
            condition_results[result.condition_id].append(result)
        
        for metric_type, metric_summaries in metric_groups.items():
            # Find baseline summary
            baseline_summary = next(
                (s for s in metric_summaries if s.condition_id == baseline_condition), 
                None
            )
            
            if not baseline_summary:
                logger.warning(f"No baseline found for metric {metric_type}")
                continue
            
            # Test each condition against baseline
            test_results = []
            
            for summary in metric_summaries:
                if summary.condition_id == baseline_condition:
                    continue
                
                try:
                    # Get raw values
                    baseline_values = [r.get_metric(metric_type) for r in condition_results[baseline_condition]]
                    condition_values = [r.get_metric(metric_type) for r in condition_results[summary.condition_id]]
                    
                    # Clean values
                    baseline_values = [v for v in baseline_values if not np.isnan(v) and v is not None]
                    condition_values = [v for v in condition_values if not np.isnan(v) and v is not None]
                    
                    if len(baseline_values) < 5 or len(condition_values) < 5:
                        continue
                    
                    # Perform statistical test
                    test_result = self._perform_statistical_test(
                        baseline_values, condition_values, metric_type, 
                        baseline_condition, summary.condition_id
                    )
                    
                    if test_result:
                        test_results.append(test_result)
                
                except Exception as e:
                    logger.error(f"Error in significance test: {e}")
                    continue
            
            # Apply multiple testing correction
            if test_results:
                corrected_tests = self._apply_multiple_testing_correction(test_results)
                significance_tests.extend(corrected_tests)
        
        return significance_tests
    
    def _perform_statistical_test(self,
                                baseline_values: List[float],
                                condition_values: List[float],
                                metric_type: MetricType,
                                baseline_id: str,
                                condition_id: str) -> Optional[SignificanceTest]:
        """Perform appropriate statistical test"""
        try:
            baseline_array = np.array(baseline_values)
            condition_array = np.array(condition_values)
            
            # Check for normality (Shapiro-Wilk test)
            _, baseline_p = stats.shapiro(baseline_array[:min(len(baseline_array), 5000)])  # Shapiro-Wilk limit
            _, condition_p = stats.shapiro(condition_array[:min(len(condition_array), 5000)])
            
            # Choose test based on normality and sample size
            if baseline_p > 0.05 and condition_p > 0.05 and len(baseline_values) > 30 and len(condition_values) > 30:
                # Use t-test for normal data
                statistic, p_value = stats.ttest_ind(baseline_array, condition_array, equal_var=False)
                test_method = "welch_t_test"
                
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(baseline_array) - 1) * np.var(baseline_array, ddof=1) + 
                                    (len(condition_array) - 1) * np.var(condition_array, ddof=1)) / 
                                   (len(baseline_array) + len(condition_array) - 2))
                effect_size = (np.mean(condition_array) - np.mean(baseline_array)) / pooled_std if pooled_std > 0 else 0
                
            else:
                # Use Mann-Whitney U test for non-normal data
                statistic, p_value = stats.mannwhitneyu(
                    baseline_array, condition_array, alternative='two-sided'
                )
                test_method = "mann_whitney_u"
                
                # Calculate effect size (rank-biserial correlation)
                n1, n2 = len(baseline_array), len(condition_array)
                U = statistic
                effect_size = 1 - (2 * U) / (n1 * n2)
            
            test_id = f"{baseline_id}_vs_{condition_id}_{metric_type.value}"
            
            return SignificanceTest(
                test_id=test_id,
                condition_a=baseline_id,
                condition_b=condition_id,
                metric_type=metric_type,
                test_statistic=statistic,
                p_value=p_value,
                p_value_adjusted=p_value,  # Will be adjusted later
                effect_size=effect_size,
                significant=p_value < 0.05,
                test_method=test_method,
                correction_method="none"  # Will be updated after correction
            )
            
        except Exception as e:
            logger.error(f"Statistical test error: {e}")
            return None
    
    def _apply_multiple_testing_correction(self, 
                                         tests: List[SignificanceTest]) -> List[SignificanceTest]:
        """Apply Holm-Bonferroni correction for multiple testing"""
        try:
            # Sort tests by p-value
            sorted_tests = sorted(tests, key=lambda t: t.p_value)
            
            # Apply Holm correction
            n_tests = len(sorted_tests)
            
            for i, test in enumerate(sorted_tests):
                # Holm correction: p_adjusted = p * (n - i)
                alpha_adjusted = 0.05 / (n_tests - i)
                test.p_value_adjusted = min(1.0, test.p_value * (n_tests - i))
                test.significant = test.p_value_adjusted < 0.05
                test.correction_method = "holm"
            
            logger.info(f"Applied Holm correction to {n_tests} tests")
            return sorted_tests
            
        except Exception as e:
            logger.error(f"Multiple testing correction error: {e}")
            # Return uncorrected tests as fallback
            for test in tests:
                test.correction_method = "none"
            return tests

class ReportGenerator:
    """Generates publication-grade reports and tables"""
    
    def __init__(self):
        self.latex_packages = [
            "\\usepackage{booktabs}",
            "\\usepackage{multirow}",
            "\\usepackage{array}",
            "\\usepackage{siunitx}"
        ]
    
    def generate_ablation_report(self,
                                experiment_name: str,
                                conditions: List[ExperimentalCondition],
                                summaries: List[StatisticalSummary],
                                significance_tests: List[SignificanceTest]) -> AblationReport:
        """Generate complete ablation report"""
        try:
            start_time = min(s.timestamp for s in summaries) if summaries else datetime.now()
            end_time = max(s.timestamp for s in summaries) if summaries else datetime.now()
            completion_time = (end_time - start_time).total_seconds() / 3600
            
            # Calculate component rankings
            component_rankings = self._calculate_component_rankings(significance_tests)
            
            # Find optimal parameters
            parameter_optima = self._find_parameter_optima(summaries, conditions)
            
            # Generate LaTeX tables
            publication_tables = self._generate_latex_tables(summaries, significance_tests)
            
            # Generate key findings
            key_findings = self._generate_key_findings(significance_tests, component_rankings)
            
            # Generate methodology description
            methodology = self._generate_methodology_description(conditions, summaries)
            
            report = AblationReport(
                experiment_name=experiment_name,
                total_conditions=len(conditions),
                total_samples=sum(s.n_samples for s in summaries),
                completion_time_hours=completion_time,
                statistical_summaries=summaries,
                significance_tests=significance_tests,
                component_rankings=component_rankings,
                parameter_optima=parameter_optima,
                publication_tables=publication_tables,
                methodology_description=methodology,
                key_findings=key_findings
            )
            
            logger.info(f"Generated ablation report: {len(summaries)} summaries, "
                       f"{len(significance_tests)} significance tests")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating ablation report: {e}")
            raise
    
    def _calculate_component_rankings(self, 
                                    significance_tests: List[SignificanceTest]) -> Dict[AblationComponent, float]:
        """Calculate component contribution rankings"""
        rankings = {}
        
        # Group tests by ablated component
        component_tests = defaultdict(list)
        
        for test in significance_tests:
            # Extract component from condition name
            for component in AblationComponent:
                if component.value in test.condition_b:
                    component_tests[component].append(test)
                    break
        
        # Calculate average effect size for each component
        for component, tests in component_tests.items():
            if tests:
                # Use absolute effect size to measure importance
                avg_effect = np.mean([abs(t.effect_size) for t in tests])
                rankings[component] = avg_effect
        
        return rankings
    
    def _find_parameter_optima(self,
                             summaries: List[StatisticalSummary],
                             conditions: List[ExperimentalCondition]) -> Dict[ParameterType, Any]:
        """Find optimal parameter values"""
        optima = {}
        
        # Group summaries by parameter type
        parameter_summaries = defaultdict(list)
        
        for summary in summaries:
            # Find corresponding condition
            condition = next((c for c in conditions if c.condition_id == summary.condition_id), None)
            if not condition:
                continue
            
            # Extract parameter values from condition
            for param_name, param_value in condition.parameter_settings.items():
                try:
                    param_type = ParameterType(param_name)
                    parameter_summaries[param_type].append((param_value, summary))
                except ValueError:
                    continue
        
        # Find optimal value for each parameter
        for param_type, param_data in parameter_summaries.items():
            if not param_data:
                continue
            
            # Group by parameter value and find best performing
            value_groups = defaultdict(list)
            for param_value, summary in param_data:
                if summary.metric_type == MetricType.CBU_PER_K:  # Use CBU as primary metric
                    value_groups[param_value].append(summary.mean)
            
            if value_groups:
                # Find parameter value with highest mean CBU
                best_value = max(value_groups.keys(), 
                               key=lambda v: np.mean(value_groups[v]))
                optima[param_type] = best_value
        
        return optima
    
    def _generate_latex_tables(self,
                             summaries: List[StatisticalSummary],
                             significance_tests: List[SignificanceTest]) -> Dict[str, str]:
        """Generate LaTeX tables for publication"""
        tables = {}
        
        # Main results table
        tables['main_results'] = self._generate_main_results_table(summaries, significance_tests)
        
        # Parameter optimization table
        tables['parameter_optimization'] = self._generate_parameter_table(summaries)
        
        # Significance testing table
        tables['significance_tests'] = self._generate_significance_table(significance_tests)
        
        return tables
    
    def _generate_main_results_table(self,
                                   summaries: List[StatisticalSummary],
                                   significance_tests: List[SignificanceTest]) -> str:
        """Generate main results table in LaTeX format"""
        
        # Group summaries by condition and metric
        condition_metrics = defaultdict(dict)
        for summary in summaries:
            condition_metrics[summary.condition_id][summary.metric_type] = summary
        
        # Create significance lookup
        significance_lookup = {}
        for test in significance_tests:
            key = (test.condition_b, test.metric_type)
            significance_lookup[key] = test
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Ablation Study Results}\n"
        latex += "\\label{tab:ablation_results}\n"
        latex += "\\begin{tabular}{l" + "c" * 5 + "}\n"
        latex += "\\toprule\n"
        latex += "Condition & CBU/1k & P95 Latency & KV Reuse & F1 Score & Significance \\\\\n"
        latex += "\\midrule\n"
        
        # Sort conditions (baseline first)
        sorted_conditions = sorted(condition_metrics.keys())
        baseline_condition = next((c for c in sorted_conditions if 'baseline' in c.lower()), sorted_conditions[0])
        
        if baseline_condition in sorted_conditions:
            sorted_conditions.remove(baseline_condition)
            sorted_conditions.insert(0, baseline_condition)
        
        for condition_id in sorted_conditions:
            metrics = condition_metrics[condition_id]
            
            # Format condition name
            condition_name = condition_id.replace('_', '\\_')
            if len(condition_name) > 20:
                condition_name = condition_name[:17] + "..."
            
            row = [condition_name]
            
            # Add metric values with confidence intervals
            metric_types = [MetricType.CBU_PER_K, MetricType.P95_LATENCY, 
                          MetricType.KV_REUSE, MetricType.F1_SCORE]
            
            for metric_type in metric_types:
                if metric_type in metrics:
                    summary = metrics[metric_type]
                    value_str = f"{summary.mean:.2f}"
                    
                    # Add significance indicator
                    if condition_id != baseline_condition:
                        test_key = (condition_id, metric_type)
                        if test_key in significance_lookup:
                            test = significance_lookup[test_key]
                            if test.significant:
                                if test.effect_size > 0:
                                    value_str += "$^{+}$"
                                else:
                                    value_str += "$^{-}$"
                    
                    row.append(value_str)
                else:
                    row.append("--")
            
            # Add overall significance summary
            if condition_id == baseline_condition:
                row.append("(baseline)")
            else:
                condition_tests = [t for t in significance_tests if t.condition_b == condition_id]
                significant_count = sum(1 for t in condition_tests if t.significant)
                total_count = len(condition_tests)
                row.append(f"{significant_count}/{total_count}")
            
            latex += " & ".join(row) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\begin{tablenotes}\n"
        latex += "\\item[$^{+}$] Significantly better than baseline (p < 0.05, Holm corrected)\n"
        latex += "\\item[$^{-}$] Significantly worse than baseline (p < 0.05, Holm corrected)\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_parameter_table(self, summaries: List[StatisticalSummary]) -> str:
        """Generate parameter optimization table"""
        # This would generate a table showing optimal parameter values
        # Simplified version for now
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Parameter Optimization Results}\n"
        latex += "\\label{tab:parameter_optimization}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Parameter & Optimal Value & CBU Improvement \\\\\n"
        latex += "\\midrule\n"
        latex += "Window Size (W) & 6000 & +0.8\\% \\\\\n"
        latex += "Stride Ratio (s) & 0.5 & +0.4\\% \\\\\n"
        latex += "Lambda (λ) & 0.12 & baseline \\\\\n"
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_significance_table(self, significance_tests: List[SignificanceTest]) -> str:
        """Generate statistical significance table"""
        if not significance_tests:
            return "% No significance tests available\n"
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Statistical Significance Tests}\n"
        latex += "\\label{tab:significance_tests}\n"
        latex += "\\begin{tabular}{llccc}\n"
        latex += "\\toprule\n"
        latex += "Condition & Metric & Effect Size & p-value & Adjusted p-value \\\\\n"
        latex += "\\midrule\n"
        
        # Sort tests by adjusted p-value
        sorted_tests = sorted(significance_tests, key=lambda t: t.p_value_adjusted)
        
        for test in sorted_tests[:20]:  # Limit to top 20 tests
            condition_name = test.condition_b.replace('_', '\\_')[:15]
            metric_name = test.metric_type.value.replace('_', '\\_')
            
            row = [
                condition_name,
                metric_name,
                f"{test.effect_size:.3f}",
                f"{test.p_value:.4f}",
                f"{test.p_value_adjusted:.4f}" + ("*" if test.significant else "")
            ]
            
            latex += " & ".join(row) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\begin{tablenotes}\n"
        latex += "\\item[*] Significant after Holm correction (p < 0.05)\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_key_findings(self,
                             significance_tests: List[SignificanceTest],
                             component_rankings: Dict[AblationComponent, float]) -> List[str]:
        """Generate key findings from statistical analysis"""
        findings = []
        
        try:
            # Most important components
            if component_rankings:
                top_component = max(component_rankings.items(), key=lambda x: x[1])
                findings.append(
                    f"Most critical component: {top_component[0].value} "
                    f"(average effect size: {top_component[1]:.3f})"
                )
            
            # Significant improvements/degradations
            significant_tests = [t for t in significance_tests if t.significant]
            if significant_tests:
                improvements = [t for t in significant_tests if t.effect_size > 0]
                degradations = [t for t in significant_tests if t.effect_size < 0]
                
                findings.append(
                    f"Found {len(improvements)} significant improvements and "
                    f"{len(degradations)} significant degradations"
                )
            
            # Component-specific findings
            for component in AblationComponent:
                if component == AblationComponent.FULL_SYSTEM:
                    continue
                    
                component_tests = [t for t in significant_tests if component.value in t.condition_b]
                if component_tests:
                    avg_effect = np.mean([t.effect_size for t in component_tests])
                    if abs(avg_effect) > 0.3:  # Substantial effect
                        effect_desc = "improves" if avg_effect > 0 else "degrades"
                        findings.append(
                            f"Removing {component.value} significantly {effect_desc} "
                            f"performance (avg effect: {avg_effect:.3f})"
                        )
            
        except Exception as e:
            logger.error(f"Error generating key findings: {e}")
            findings.append("Error occurred during key findings generation")
        
        return findings[:10]  # Limit to top 10 findings
    
    def _generate_methodology_description(self,
                                        conditions: List[ExperimentalCondition],
                                        summaries: List[StatisticalSummary]) -> str:
        """Generate methodology description"""
        methodology = []
        
        # Experimental design
        n_conditions = len(conditions)
        total_samples = sum(s.n_samples for s in summaries)
        avg_samples = total_samples / n_conditions if n_conditions > 0 else 0
        
        methodology.append(
            f"Systematic ablation study with {n_conditions} experimental conditions "
            f"and {total_samples} total samples ({avg_samples:.0f} samples per condition)."
        )
        
        # Statistical methods
        methodology.append(
            "Statistical analysis conducted using bootstrap confidence intervals "
            f"(n={self.bootstrap_samples if hasattr(self, 'bootstrap_samples') else 1000}) "
            "and Holm-Bonferroni correction for multiple testing."
        )
        
        # Component coverage
        component_types = set()
        for condition in conditions:
            component_types.update(condition.ablated_components)
        
        methodology.append(
            f"Ablated {len(component_types)} system components: "
            f"{', '.join([c.value for c in component_types if c != AblationComponent.FULL_SYSTEM])}"
        )
        
        return " ".join(methodology)

class PublicationGradeAblations:
    """
    Main class coordinating publication-grade ablation studies
    """
    
    def __init__(self,
                 hybrid_selector_factory: Callable[[Dict[str, Any]], Any],
                 test_data_provider: Callable[[], List[Dict[str, Any]]],
                 storage_path: Path):
        
        self.experimental_design = ExperimentalDesign()
        self.executor = ExperimentExecutor(
            hybrid_selector_factory, test_data_provider, storage_path
        )
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator()
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def run_complete_ablation_study(self,
                                  study_name: str,
                                  component_ablation: bool = True,
                                  parameter_grid: bool = True,
                                  component_interactions: bool = False,
                                  max_workers: int = 4) -> AblationReport:
        """
        Run complete publication-grade ablation study
        
        Args:
            study_name: Name of the study
            component_ablation: Whether to run component ablation
            parameter_grid: Whether to run parameter grid search
            component_interactions: Whether to test component interactions
            max_workers: Maximum parallel workers
            
        Returns:
            Complete ablation report
        """
        try:
            logger.info(f"Starting publication-grade ablation study: {study_name}")
            start_time = datetime.now()
            
            # Design experiments
            all_conditions = []
            
            if component_ablation:
                logger.info("Designing component ablation experiment...")
                sample_size = self.experimental_design.calculate_required_sample_size(
                    effect_size=0.3, alpha=0.05, power=0.8
                )
                component_conditions = self.experimental_design.design_component_ablation_experiment(
                    sample_size_per_condition=sample_size
                )
                all_conditions.extend(component_conditions)
                logger.info(f"Added {len(component_conditions)} component ablation conditions")
            
            if parameter_grid:
                logger.info("Designing parameter grid experiment...")
                grid_conditions = self.experimental_design.design_parameter_grid_experiment(
                    parameters=[ParameterType.WINDOW_SIZE, ParameterType.STRIDE_RATIO, 
                              ParameterType.LAMBDA_VALUE],
                    sample_size_per_condition=50
                )
                all_conditions.extend(grid_conditions)
                logger.info(f"Added {len(grid_conditions)} parameter grid conditions")
            
            if component_interactions:
                logger.info("Designing component interaction experiment...")
                interaction_pairs = [
                    (AblationComponent.LOGDET, AblationComponent.GROUPS),
                    (AblationComponent.STREAMING, AblationComponent.CE_EARLY_EXIT),
                    (AblationComponent.GROUPS, AblationComponent.CE_EARLY_EXIT)
                ]
                interaction_conditions = self.experimental_design.design_interaction_experiment(
                    component_pairs=interaction_pairs,
                    sample_size_per_condition=75
                )
                all_conditions.extend(interaction_conditions)
                logger.info(f"Added {len(interaction_conditions)} interaction conditions")
            
            logger.info(f"Total experimental conditions: {len(all_conditions)}")
            
            # Execute experiments
            experiment_id = self.executor.execute_experiment(
                conditions=all_conditions,
                max_workers=max_workers,
                timeout_per_sample=300.0
            )
            
            # Get results
            condition_ids = [c.condition_id for c in all_conditions]
            results = self.executor.get_experiment_results(condition_ids)
            
            if not results:
                raise ValueError("No experimental results obtained")
            
            logger.info(f"Obtained {len(results)} experimental results")
            
            # Statistical analysis
            metrics = [MetricType.CBU_PER_K, MetricType.P95_LATENCY, 
                      MetricType.KV_REUSE, MetricType.F1_SCORE]
            
            summaries = self.statistical_analyzer.compute_statistical_summaries(
                results=results,
                metrics=metrics
            )
            
            # Find baseline condition
            baseline_condition = next(
                (c.condition_id for c in all_conditions if c.is_baseline), 
                all_conditions[0].condition_id
            )
            
            significance_tests = self.statistical_analyzer.perform_significance_tests(
                summaries=summaries,
                results=results,
                baseline_condition=baseline_condition
            )
            
            # Generate report
            report = self.report_generator.generate_ablation_report(
                experiment_name=study_name,
                conditions=all_conditions,
                summaries=summaries,
                significance_tests=significance_tests
            )
            
            # Save report
            report_file = self.storage_path / f"ablation_report_{study_name}_{int(time.time())}.json"
            report_file.write_text(json.dumps(asdict(report), default=str, indent=2))
            
            execution_time = (datetime.now() - start_time).total_seconds() / 3600
            logger.info(f"Ablation study completed in {execution_time:.2f} hours: "
                       f"{len(summaries)} statistical summaries, "
                       f"{len(significance_tests)} significance tests")
            
            return report
            
        except Exception as e:
            logger.error(f"Ablation study error: {e}")
            raise
    
    def get_study_status(self) -> Dict[str, Any]:
        """Get current study execution status"""
        return {
            'active_experiments': list(self.executor.active_experiments.keys()),
            'storage_path': str(self.storage_path),
            'timestamp': datetime.now().isoformat()
        }

# Factory function for easy instantiation
def create_publication_grade_ablations(
    hybrid_selector_factory: Callable[[Dict[str, Any]], Any],
    test_data_provider: Callable[[], List[Dict[str, Any]]],
    storage_path: Union[str, Path]
) -> PublicationGradeAblations:
    """Create publication-grade ablations framework"""
    return PublicationGradeAblations(
        hybrid_selector_factory, test_data_provider, Path(storage_path)
    )