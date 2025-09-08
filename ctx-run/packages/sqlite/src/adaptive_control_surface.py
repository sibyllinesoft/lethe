#!/usr/bin/env python3
"""
Adaptive Control Surface for Hybrid System Parameter Management

This module implements a 3-dial adaptive control surface that dynamically adjusts
hybrid system parameters based on real-time performance feedback and context bucket
classification. The system maintains optimal performance across diverse workloads
while ensuring production safety and operational stability.

Core Parameters:
- λ (head_keep): Token budget control for head selection
- μ (tail_window/stride): Compute budget control for tail processing  
- r/K2: Diversity/cross-encoder effort control

Key Features:
- Per-turn adaptive adjustment based on context analysis
- Context bucket classification with 6 failure metrics
- Real-time parameter bounds validation with safety guards
- Performance feedback integration with micro-policy rules
- Comprehensive audit logging for operational transparency
"""

import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from collections import defaultdict, deque
import statistics
import numpy as np
import json
import math
from pathlib import Path

logger = logging.getLogger(__name__)

class ContextBucket(Enum):
    """Context bucket classifications for parameter tuning"""
    ENTITY_ENTROPY = "entity_entropy"      # Semantic complexity
    DUP_RATE = "dup_rate"                  # Code duplication level
    SYMBOL_DEPTH = "symbol_depth"         # Symbol table complexity
    REPO_FANOUT = "repo_fanout"           # Repository breadth
    HUNK_LENGTH = "hunk_length"           # Code chunk size
    NL_CODE_RATIO = "nl_code_ratio"       # Natural language vs code balance

class PerformanceLevel(Enum):
    """Performance levels for adaptive control"""
    EXCELLENT = "excellent"    # > target + 20%
    GOOD = "good"             # target to target + 20%
    ACCEPTABLE = "acceptable"  # target - 10% to target
    POOR = "poor"             # target - 20% to target - 10%
    CRITICAL = "critical"     # < target - 20%

@dataclass
class ParameterBounds:
    """Parameter bounds with safety limits"""
    lambda_min: float = 0.08      # Minimum 8% head keep
    lambda_max: float = 0.20      # Maximum 20% head keep
    lambda_default: float = 0.12  # Default 12% head keep
    
    mu_window_min: int = 4000     # Minimum 4k window
    mu_window_max: int = 8000     # Maximum 8k window  
    mu_window_default: int = 6000 # Default 6k window
    
    mu_stride_min: float = 0.25   # Minimum 25% stride
    mu_stride_max: float = 0.75   # Maximum 75% stride
    mu_stride_default: float = 0.50  # Default 50% stride
    
    r_min: int = 8               # Minimum DPP rank
    r_max: int = 20              # Maximum DPP rank 
    r_default: int = 14          # Default DPP rank
    
    k2_min: int = 160            # Minimum CE candidates
    k2_max: int = 480            # Maximum CE candidates
    k2_default: int = 320        # Default CE candidates

@dataclass
class ContextMetrics:
    """Context analysis metrics for bucket classification"""
    entity_entropy: float = 0.0          # Semantic complexity score
    dup_rate: float = 0.0                # Code duplication percentage
    symbol_depth: int = 0                # Symbol nesting level
    repo_fanout: int = 0                 # Number of files/modules
    hunk_length: int = 0                 # Average chunk size
    nl_code_ratio: float = 0.5           # NL to code ratio
    
    # Derived classifications
    bucket_scores: Dict[ContextBucket, float] = field(default_factory=dict)
    dominant_bucket: Optional[ContextBucket] = None
    complexity_score: float = 0.0        # Overall complexity [0,1]

@dataclass
class ControlParameters:
    """Current control parameter state"""
    lambda_value: float = 0.12           # Head keep ratio
    mu_window_size: int = 6000           # Tail window size
    mu_stride: int = 3000                # Tail stride (derived)
    r_value: int = 14                    # DPP rank
    k2_value: int = 320                  # CE candidates
    
    # Metadata
    last_update: datetime = field(default_factory=datetime.now)
    update_reason: str = ""
    safety_bounded: bool = False         # True if parameters hit bounds
    
    def get_stride_ratio(self) -> float:
        """Get stride as ratio of window size"""
        return self.mu_stride / self.mu_window_size if self.mu_window_size > 0 else 0.5
    
    def set_stride_ratio(self, ratio: float):
        """Set stride as ratio of window size"""
        self.mu_stride = int(self.mu_window_size * ratio)

@dataclass
class AdaptationEvent:
    """Record of parameter adaptation"""
    timestamp: datetime = field(default_factory=datetime.now)
    context_metrics: ContextMetrics = field(default_factory=ContextMetrics)
    old_parameters: ControlParameters = field(default_factory=ControlParameters)
    new_parameters: ControlParameters = field(default_factory=ControlParameters)
    adaptation_reason: str = ""
    performance_delta: float = 0.0       # Expected improvement
    micro_policy_applied: str = ""

class ContextAnalyzer:
    """Analyzes context to determine appropriate parameter bucket"""
    
    def __init__(self, bounds: ParameterBounds):
        self.bounds = bounds
        self.analysis_history = deque(maxlen=1000)
        
    def analyze_context(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ContextMetrics:
        """
        Analyze context content to extract bucket classification metrics
        
        Args:
            content: Input content to analyze
            metadata: Optional metadata with pre-computed metrics
            
        Returns:
            ContextMetrics with bucket scores and dominant classification
        """
        try:
            metadata = metadata or {}
            
            # Extract base metrics
            entity_entropy = self._calculate_entity_entropy(content)
            dup_rate = self._calculate_duplication_rate(content)
            symbol_depth = self._calculate_symbol_depth(content)
            repo_fanout = metadata.get('repo_fanout', self._estimate_repo_fanout(content))
            hunk_length = self._calculate_average_hunk_length(content)
            nl_code_ratio = self._calculate_nl_code_ratio(content)
            
            # Create metrics object
            metrics = ContextMetrics(
                entity_entropy=entity_entropy,
                dup_rate=dup_rate,
                symbol_depth=symbol_depth,
                repo_fanout=repo_fanout,
                hunk_length=hunk_length,
                nl_code_ratio=nl_code_ratio
            )
            
            # Calculate bucket scores
            metrics.bucket_scores = self._calculate_bucket_scores(metrics)
            
            # Determine dominant bucket
            if metrics.bucket_scores:
                dominant_bucket, max_score = max(metrics.bucket_scores.items(), key=lambda x: x[1])
                metrics.dominant_bucket = dominant_bucket if max_score > 0.6 else None
            
            # Calculate overall complexity
            metrics.complexity_score = self._calculate_complexity_score(metrics)
            
            # Record analysis
            self.analysis_history.append(metrics)
            
            logger.debug(f"Context analysis: dominant_bucket={metrics.dominant_bucket}, "
                        f"complexity={metrics.complexity_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return ContextMetrics()  # Return default metrics
    
    def _calculate_entity_entropy(self, content: str) -> float:
        """Calculate semantic entity entropy"""
        try:
            # Extract entities (simplified approach)
            words = content.split()
            entities = set()
            
            for word in words:
                # Identify potential entities
                if (word.isupper() or 
                    word.startswith('_') or
                    '()' in word or
                    word.endswith('()') or
                    word.count('_') >= 2):
                    entities.add(word.lower())
            
            if not entities or not words:
                return 0.0
            
            # Calculate entropy based on entity frequency
            entity_counts = defaultdict(int)
            for word in words:
                if word.lower() in entities:
                    entity_counts[word.lower()] += 1
            
            total_entity_refs = sum(entity_counts.values())
            if total_entity_refs == 0:
                return 0.0
            
            entropy = 0.0
            for count in entity_counts.values():
                prob = count / total_entity_refs
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            # Normalize to [0, 1]
            max_entropy = math.log2(len(entity_counts)) if len(entity_counts) > 1 else 1.0
            return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Entity entropy calculation error: {e}")
            return 0.5  # Default moderate entropy
    
    def _calculate_duplication_rate(self, content: str) -> float:
        """Calculate code duplication rate"""
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if len(lines) < 2:
                return 0.0
            
            # Count duplicate lines
            line_counts = defaultdict(int)
            for line in lines:
                if len(line) > 10:  # Only count substantial lines
                    line_counts[line] += 1
            
            duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)
            total_lines = len(lines)
            
            return duplicate_lines / total_lines if total_lines > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Duplication rate calculation error: {e}")
            return 0.0
    
    def _calculate_symbol_depth(self, content: str) -> int:
        """Calculate symbol table nesting depth"""
        try:
            max_depth = 0
            current_depth = 0
            
            # Simple nesting detection
            for char in content:
                if char in '({[':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char in ')}]':
                    current_depth = max(0, current_depth - 1)
            
            return max_depth
            
        except Exception as e:
            logger.debug(f"Symbol depth calculation error: {e}")
            return 0
    
    def _estimate_repo_fanout(self, content: str) -> int:
        """Estimate repository fanout from content"""
        try:
            # Count import statements and file references
            lines = content.split('\n')
            fanout_indicators = set()
            
            for line in lines:
                line_stripped = line.strip().lower()
                
                # Python imports
                if line_stripped.startswith(('import ', 'from ')):
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        fanout_indicators.add(parts[1].split('.')[0])
                
                # File references
                if any(ext in line_stripped for ext in ['.py', '.js', '.ts', '.json', '.yaml']):
                    fanout_indicators.add('file_ref')
                
                # Module references
                if '::' in line or '.' in line and len(line.split('.')) > 2:
                    fanout_indicators.add('module_ref')
            
            return len(fanout_indicators)
            
        except Exception as e:
            logger.debug(f"Repo fanout estimation error: {e}")
            return 1
    
    def _calculate_average_hunk_length(self, content: str) -> int:
        """Calculate average code hunk length"""
        try:
            lines = content.split('\n')
            if not lines:
                return 0
            
            hunks = []
            current_hunk = []
            
            for line in lines:
                if line.strip():
                    current_hunk.append(line)
                else:
                    if current_hunk:
                        hunks.append(len(current_hunk))
                        current_hunk = []
            
            if current_hunk:
                hunks.append(len(current_hunk))
            
            return int(statistics.mean(hunks)) if hunks else 0
            
        except Exception as e:
            logger.debug(f"Hunk length calculation error: {e}")
            return 10  # Default hunk length
    
    def _calculate_nl_code_ratio(self, content: str) -> float:
        """Calculate natural language to code ratio"""
        try:
            lines = content.split('\n')
            nl_lines = 0
            code_lines = 0
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                # Heuristics for natural language vs code
                if (line_stripped.startswith(('#', '//', '"""', "'''", '/*')) or
                    ' ' in line_stripped and not any(op in line_stripped for op in ['=', '(', ')', '{', '}', '[', ']'])):
                    nl_lines += 1
                else:
                    code_lines += 1
            
            total_lines = nl_lines + code_lines
            return nl_lines / total_lines if total_lines > 0 else 0.5
            
        except Exception as e:
            logger.debug(f"NL/code ratio calculation error: {e}")
            return 0.5
    
    def _calculate_bucket_scores(self, metrics: ContextMetrics) -> Dict[ContextBucket, float]:
        """Calculate bucket classification scores"""
        scores = {}
        
        # Entity entropy bucket - high entropy indicates complex semantic content
        scores[ContextBucket.ENTITY_ENTROPY] = min(1.0, metrics.entity_entropy * 1.2)
        
        # Duplication rate bucket - high duplication needs different handling  
        scores[ContextBucket.DUP_RATE] = min(1.0, metrics.dup_rate * 2.0)
        
        # Symbol depth bucket - deep nesting needs careful selection
        depth_norm = min(1.0, metrics.symbol_depth / 20.0)  # Normalize assuming max 20 levels
        scores[ContextBucket.SYMBOL_DEPTH] = depth_norm
        
        # Repo fanout bucket - wide dependencies need broader context
        fanout_norm = min(1.0, metrics.repo_fanout / 50.0)  # Normalize assuming max 50 modules
        scores[ContextBucket.REPO_FANOUT] = fanout_norm
        
        # Hunk length bucket - long chunks need different windowing
        length_norm = min(1.0, metrics.hunk_length / 100.0)  # Normalize assuming max 100 lines
        scores[ContextBucket.HUNK_LENGTH] = length_norm
        
        # NL/code ratio bucket - balanced content needs different processing
        # Score is higher when ratio is closer to 0.5 (balanced)
        balance_score = 1.0 - abs(metrics.nl_code_ratio - 0.5) * 2.0
        scores[ContextBucket.NL_CODE_RATIO] = max(0.0, balance_score)
        
        return scores
    
    def _calculate_complexity_score(self, metrics: ContextMetrics) -> float:
        """Calculate overall complexity score [0,1]"""
        try:
            # Weighted combination of factors
            weights = {
                'entity_entropy': 0.25,
                'dup_rate': 0.15,
                'symbol_depth': 0.20,
                'repo_fanout': 0.15,
                'hunk_length': 0.15,
                'nl_code_ratio': 0.10
            }
            
            score = 0.0
            score += weights['entity_entropy'] * metrics.entity_entropy
            score += weights['dup_rate'] * metrics.dup_rate
            score += weights['symbol_depth'] * min(1.0, metrics.symbol_depth / 20.0)
            score += weights['repo_fanout'] * min(1.0, metrics.repo_fanout / 50.0)
            score += weights['hunk_length'] * min(1.0, metrics.hunk_length / 100.0)
            score += weights['nl_code_ratio'] * abs(metrics.nl_code_ratio - 0.5) * 2.0
            
            return min(1.0, score)
            
        except Exception as e:
            logger.debug(f"Complexity score calculation error: {e}")
            return 0.5

class MicroPolicyEngine:
    """Micro-policy rule engine for automatic parameter adjustment"""
    
    def __init__(self, bounds: ParameterBounds):
        self.bounds = bounds
        self.policy_history = deque(maxlen=500)
        
        # Policy rules mapping bucket -> parameter adjustments
        self.policy_rules = self._initialize_policy_rules()
        
    def _initialize_policy_rules(self) -> Dict[ContextBucket, Dict[str, float]]:
        """Initialize micro-policy adjustment rules"""
        return {
            # High duplication -> increase diversity (r↑, K2↑)
            ContextBucket.DUP_RATE: {
                'lambda_delta': 0.0,      # No change to head keep
                'window_delta': 0.0,      # No change to window
                'stride_ratio_delta': 0.0, # No change to stride
                'r_delta': 2.0,           # Increase DPP rank by 2
                'k2_ratio_delta': 0.20    # Increase K2 by 20%
            },
            
            # Deep symbols -> increase head keep (head_keep↑)
            ContextBucket.SYMBOL_DEPTH: {
                'lambda_delta': 0.02,     # Increase head keep by 2-3pp
                'window_delta': 0.0,
                'stride_ratio_delta': 0.0,
                'r_delta': 0.0,
                'k2_ratio_delta': 0.0
            },
            
            # Low entity entropy -> reduce tail processing (tail_window↓ or stride↑)
            ContextBucket.ENTITY_ENTROPY: {
                'lambda_delta': 0.0,
                'window_delta': -500.0,   # Reduce window by 500 tokens
                'stride_ratio_delta': 0.10, # Increase stride ratio by 10%
                'r_delta': 0.0,
                'k2_ratio_delta': 0.0
            },
            
            # High repo fanout -> increase window for broader context
            ContextBucket.REPO_FANOUT: {
                'lambda_delta': 0.01,     # Slight head increase
                'window_delta': 500.0,    # Increase window by 500 tokens
                'stride_ratio_delta': -0.05, # Decrease stride for more overlap
                'r_delta': 1.0,           # Slight rank increase
                'k2_ratio_delta': 0.10    # 10% more CE candidates
            },
            
            # Long hunks -> adjust windowing strategy
            ContextBucket.HUNK_LENGTH: {
                'lambda_delta': -0.01,    # Reduce head slightly
                'window_delta': 1000.0,   # Increase window for long chunks
                'stride_ratio_delta': -0.10, # Reduce stride for better coverage
                'r_delta': 0.0,
                'k2_ratio_delta': 0.0
            },
            
            # Balanced NL/code -> optimize for mixed content
            ContextBucket.NL_CODE_RATIO: {
                'lambda_delta': 0.005,    # Slight head increase
                'window_delta': 0.0,      # Keep window stable
                'stride_ratio_delta': 0.0,
                'r_delta': 1.0,           # Increase diversity slightly
                'k2_ratio_delta': 0.15    # 15% more candidates for mixed content
            }
        }
    
    def apply_micro_policy(self, 
                          current_params: ControlParameters, 
                          context_metrics: ContextMetrics,
                          performance_level: PerformanceLevel) -> Tuple[ControlParameters, str]:
        """
        Apply micro-policy rules to adjust parameters
        
        Args:
            current_params: Current parameter state
            context_metrics: Context analysis results
            performance_level: Current performance assessment
            
        Returns:
            Tuple of (new_parameters, policy_description)
        """
        try:
            new_params = ControlParameters(
                lambda_value=current_params.lambda_value,
                mu_window_size=current_params.mu_window_size,
                mu_stride=current_params.mu_stride,
                r_value=current_params.r_value,
                k2_value=current_params.k2_value,
                last_update=datetime.now(),
                update_reason="micro_policy_adjustment"
            )
            
            applied_policies = []
            
            # Apply primary bucket policy
            if context_metrics.dominant_bucket and context_metrics.dominant_bucket in self.policy_rules:
                bucket = context_metrics.dominant_bucket
                rules = self.policy_rules[bucket]
                
                # Apply adjustments with performance-based scaling
                performance_multiplier = self._get_performance_multiplier(performance_level)
                
                # Lambda adjustment
                lambda_delta = rules.get('lambda_delta', 0.0) * performance_multiplier
                new_params.lambda_value += lambda_delta
                
                # Window size adjustment  
                window_delta = rules.get('window_delta', 0.0) * performance_multiplier
                new_params.mu_window_size += int(window_delta)
                
                # Stride ratio adjustment
                stride_delta = rules.get('stride_ratio_delta', 0.0) * performance_multiplier
                current_ratio = new_params.get_stride_ratio()
                new_ratio = current_ratio + stride_delta
                new_params.set_stride_ratio(new_ratio)
                
                # R value adjustment
                r_delta = rules.get('r_delta', 0.0) * performance_multiplier
                new_params.r_value += int(r_delta)
                
                # K2 ratio adjustment
                k2_ratio_delta = rules.get('k2_ratio_delta', 0.0) * performance_multiplier
                new_params.k2_value = int(new_params.k2_value * (1.0 + k2_ratio_delta))
                
                applied_policies.append(f"{bucket.value}_primary")
            
            # Apply secondary policies based on significant bucket scores
            for bucket, score in context_metrics.bucket_scores.items():
                if bucket != context_metrics.dominant_bucket and score > 0.7:
                    # Apply reduced secondary policy
                    if bucket in self.policy_rules:
                        rules = self.policy_rules[bucket]
                        secondary_multiplier = 0.3 * self._get_performance_multiplier(performance_level)
                        
                        new_params.lambda_value += rules.get('lambda_delta', 0.0) * secondary_multiplier
                        new_params.mu_window_size += int(rules.get('window_delta', 0.0) * secondary_multiplier)
                        
                        applied_policies.append(f"{bucket.value}_secondary")
            
            # Apply safety bounds
            new_params, was_bounded = self._apply_safety_bounds(new_params)
            if was_bounded:
                applied_policies.append("safety_bounded")
            
            policy_description = " + ".join(applied_policies) if applied_policies else "no_policy_applied"
            
            # Record policy application
            event = AdaptationEvent(
                context_metrics=context_metrics,
                old_parameters=current_params,
                new_parameters=new_params,
                adaptation_reason=policy_description,
                micro_policy_applied=policy_description
            )
            self.policy_history.append(event)
            
            logger.info(f"Micro-policy applied: {policy_description}, "
                       f"λ: {current_params.lambda_value:.3f} → {new_params.lambda_value:.3f}, "
                       f"W: {current_params.mu_window_size} → {new_params.mu_window_size}")
            
            return new_params, policy_description
            
        except Exception as e:
            logger.error(f"Micro-policy application error: {e}")
            return current_params, f"error: {e}"
    
    def _get_performance_multiplier(self, performance_level: PerformanceLevel) -> float:
        """Get multiplier based on performance level"""
        multipliers = {
            PerformanceLevel.EXCELLENT: 0.2,   # Small adjustments when excellent
            PerformanceLevel.GOOD: 0.5,       # Moderate adjustments when good  
            PerformanceLevel.ACCEPTABLE: 1.0,  # Full adjustments when acceptable
            PerformanceLevel.POOR: 1.5,       # Aggressive adjustments when poor
            PerformanceLevel.CRITICAL: 2.0    # Maximum adjustments when critical
        }
        return multipliers.get(performance_level, 1.0)
    
    def _apply_safety_bounds(self, params: ControlParameters) -> Tuple[ControlParameters, bool]:
        """Apply safety bounds to parameters"""
        bounded = False
        
        # Lambda bounds
        if params.lambda_value < self.bounds.lambda_min:
            params.lambda_value = self.bounds.lambda_min
            bounded = True
        elif params.lambda_value > self.bounds.lambda_max:
            params.lambda_value = self.bounds.lambda_max
            bounded = True
        
        # Window size bounds
        if params.mu_window_size < self.bounds.mu_window_min:
            params.mu_window_size = self.bounds.mu_window_min
            bounded = True
        elif params.mu_window_size > self.bounds.mu_window_max:
            params.mu_window_size = self.bounds.mu_window_max  
            bounded = True
        
        # Stride bounds (as ratio)
        current_ratio = params.get_stride_ratio()
        if current_ratio < self.bounds.mu_stride_min:
            params.set_stride_ratio(self.bounds.mu_stride_min)
            bounded = True
        elif current_ratio > self.bounds.mu_stride_max:
            params.set_stride_ratio(self.bounds.mu_stride_max)
            bounded = True
        
        # R value bounds
        if params.r_value < self.bounds.r_min:
            params.r_value = self.bounds.r_min
            bounded = True
        elif params.r_value > self.bounds.r_max:
            params.r_value = self.bounds.r_max
            bounded = True
        
        # K2 bounds
        if params.k2_value < self.bounds.k2_min:
            params.k2_value = self.bounds.k2_min
            bounded = True
        elif params.k2_value > self.bounds.k2_max:
            params.k2_value = self.bounds.k2_max
            bounded = True
        
        params.safety_bounded = bounded
        return params, bounded

class AdaptiveControlSurface:
    """
    Main adaptive control surface implementing 3-dial parameter management
    """
    
    def __init__(self, bounds: Optional[ParameterBounds] = None):
        self.bounds = bounds or ParameterBounds()
        
        # Core components
        self.context_analyzer = ContextAnalyzer(self.bounds)
        self.micro_policy_engine = MicroPolicyEngine(self.bounds)
        
        # Current state
        self.current_parameters = ControlParameters(
            lambda_value=self.bounds.lambda_default,
            mu_window_size=self.bounds.mu_window_default,
            mu_stride=int(self.bounds.mu_window_default * self.bounds.mu_stride_default),
            r_value=self.bounds.r_default,
            k2_value=self.bounds.k2_default
        )
        
        # Monitoring
        self.adaptation_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Adaptive control surface initialized with bounds: "
                   f"λ=[{self.bounds.lambda_min:.3f}, {self.bounds.lambda_max:.3f}], "
                   f"W=[{self.bounds.mu_window_min}, {self.bounds.mu_window_max}], "
                   f"r=[{self.bounds.r_min}, {self.bounds.r_max}]")
    
    def adapt_parameters(self, 
                        content: str,
                        performance_metrics: Dict[str, float],
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapt parameters based on context and performance feedback
        
        Args:
            content: Input content for context analysis
            performance_metrics: Current performance measurements
            metadata: Optional metadata for enhanced analysis
            
        Returns:
            Dictionary with adapted parameters and adaptation details
        """
        try:
            with self.lock:
                start_time = time.time()
                
                # Analyze context
                context_metrics = self.context_analyzer.analyze_context(content, metadata)
                
                # Assess performance level
                performance_level = self._assess_performance_level(performance_metrics)
                
                # Record performance
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'metrics': performance_metrics.copy(),
                    'level': performance_level,
                    'context_complexity': context_metrics.complexity_score
                })
                
                # Apply micro-policies if performance indicates need for adjustment
                if performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                    new_parameters, policy_description = self.micro_policy_engine.apply_micro_policy(
                        self.current_parameters, context_metrics, performance_level
                    )
                    
                    # Update current parameters
                    old_params = self.current_parameters
                    self.current_parameters = new_parameters
                    
                    # Record adaptation
                    event = AdaptationEvent(
                        context_metrics=context_metrics,
                        old_parameters=old_params,
                        new_parameters=new_parameters,
                        adaptation_reason=f"performance_{performance_level.value}",
                        performance_delta=self._estimate_performance_delta(old_params, new_parameters),
                        micro_policy_applied=policy_description
                    )
                    self.adaptation_history.append(event)
                    
                    adapted = True
                else:
                    # No adaptation needed
                    policy_description = "no_adaptation_needed"
                    adapted = False
                
                adaptation_time = (time.time() - start_time) * 1000
                
                result = {
                    'adapted': adapted,
                    'parameters': {
                        'lambda': self.current_parameters.lambda_value,
                        'mu_window': self.current_parameters.mu_window_size,
                        'mu_stride': self.current_parameters.mu_stride,
                        'r': self.current_parameters.r_value,
                        'k2': self.current_parameters.k2_value
                    },
                    'context_analysis': {
                        'dominant_bucket': context_metrics.dominant_bucket.value if context_metrics.dominant_bucket else None,
                        'complexity_score': context_metrics.complexity_score,
                        'bucket_scores': {k.value: v for k, v in context_metrics.bucket_scores.items()}
                    },
                    'performance_assessment': {
                        'level': performance_level.value,
                        'metrics': performance_metrics
                    },
                    'adaptation_details': {
                        'policy_applied': policy_description,
                        'safety_bounded': self.current_parameters.safety_bounded,
                        'adaptation_time_ms': adaptation_time
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.debug(f"Parameter adaptation completed: adapted={adapted}, "
                            f"dominant_bucket={context_metrics.dominant_bucket}, "
                            f"performance={performance_level.value}")
                
                return result
                
        except Exception as e:
            logger.error(f"Parameter adaptation error: {e}")
            return {
                'adapted': False,
                'error': str(e),
                'parameters': self._get_current_parameters_dict(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_performance_level(self, metrics: Dict[str, float]) -> PerformanceLevel:
        """Assess current performance level"""
        try:
            # Key performance indicators with targets
            cbu_per_ms = metrics.get('cbu_per_ms', 0.0)
            p95_latency = metrics.get('p95_latency', float('inf'))
            kv_reuse = metrics.get('kv_reuse_ratio', 0.0)
            
            # Performance targets (from TODO.md requirements)
            cbu_target = 12.5
            latency_target = 1.0  # ms
            kv_reuse_target = 0.6
            
            # Calculate performance scores [0,1] where 1.0 is perfect
            cbu_score = min(1.0, cbu_per_ms / cbu_target) if cbu_target > 0 else 0.0
            latency_score = min(1.0, latency_target / p95_latency) if p95_latency > 0 else 0.0
            kv_reuse_score = min(1.0, kv_reuse / kv_reuse_target) if kv_reuse_target > 0 else 0.0
            
            # Weighted overall score
            overall_score = 0.5 * cbu_score + 0.3 * latency_score + 0.2 * kv_reuse_score
            
            # Map to performance levels
            if overall_score >= 1.2:  # Exceeds target by 20%
                return PerformanceLevel.EXCELLENT
            elif overall_score >= 1.0:  # Meets target
                return PerformanceLevel.GOOD
            elif overall_score >= 0.9:  # Within 10% of target
                return PerformanceLevel.ACCEPTABLE
            elif overall_score >= 0.8:  # Within 20% of target
                return PerformanceLevel.POOR
            else:  # Below 80% of target
                return PerformanceLevel.CRITICAL
                
        except Exception as e:
            logger.debug(f"Performance assessment error: {e}")
            return PerformanceLevel.ACCEPTABLE  # Default fallback
    
    def _estimate_performance_delta(self, 
                                  old_params: ControlParameters, 
                                  new_params: ControlParameters) -> float:
        """Estimate expected performance improvement from parameter changes"""
        try:
            # Simple heuristic-based estimation
            delta = 0.0
            
            # Lambda changes affect head quality
            lambda_delta = new_params.lambda_value - old_params.lambda_value
            delta += lambda_delta * 10.0  # Rough coefficient
            
            # Window size changes affect tail coverage  
            window_delta = new_params.mu_window_size - old_params.mu_window_size
            delta += (window_delta / 1000.0) * 2.0  # Rough coefficient
            
            # R value changes affect diversity
            r_delta = new_params.r_value - old_params.r_value
            delta += r_delta * 0.5  # Rough coefficient
            
            return delta
            
        except Exception as e:
            logger.debug(f"Performance delta estimation error: {e}")
            return 0.0
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter state"""
        with self.lock:
            return self._get_current_parameters_dict()
    
    def _get_current_parameters_dict(self) -> Dict[str, Any]:
        """Get current parameters as dictionary"""
        return {
            'lambda': self.current_parameters.lambda_value,
            'mu_window': self.current_parameters.mu_window_size,
            'mu_stride': self.current_parameters.mu_stride,
            'mu_stride_ratio': self.current_parameters.get_stride_ratio(),
            'r': self.current_parameters.r_value,
            'k2': self.current_parameters.k2_value,
            'last_update': self.current_parameters.last_update.isoformat(),
            'update_reason': self.current_parameters.update_reason,
            'safety_bounded': self.current_parameters.safety_bounded
        }
    
    def manual_override(self, parameter_updates: Dict[str, float]) -> Dict[str, Any]:
        """Manual parameter override with safety validation"""
        try:
            with self.lock:
                old_params = ControlParameters(
                    lambda_value=self.current_parameters.lambda_value,
                    mu_window_size=self.current_parameters.mu_window_size,
                    mu_stride=self.current_parameters.mu_stride,
                    r_value=self.current_parameters.r_value,
                    k2_value=self.current_parameters.k2_value
                )
                
                # Apply updates
                if 'lambda' in parameter_updates:
                    self.current_parameters.lambda_value = parameter_updates['lambda']
                if 'mu_window' in parameter_updates:
                    self.current_parameters.mu_window_size = int(parameter_updates['mu_window'])
                if 'mu_stride' in parameter_updates:
                    self.current_parameters.mu_stride = int(parameter_updates['mu_stride'])
                if 'mu_stride_ratio' in parameter_updates:
                    self.current_parameters.set_stride_ratio(parameter_updates['mu_stride_ratio'])
                if 'r' in parameter_updates:
                    self.current_parameters.r_value = int(parameter_updates['r'])
                if 'k2' in parameter_updates:
                    self.current_parameters.k2_value = int(parameter_updates['k2'])
                
                # Apply safety bounds
                self.current_parameters, was_bounded = self.micro_policy_engine._apply_safety_bounds(
                    self.current_parameters
                )
                
                # Update metadata
                self.current_parameters.last_update = datetime.now()
                self.current_parameters.update_reason = "manual_override"
                
                # Record adaptation
                event = AdaptationEvent(
                    old_parameters=old_params,
                    new_parameters=self.current_parameters,
                    adaptation_reason="manual_override",
                    micro_policy_applied="manual_override"
                )
                self.adaptation_history.append(event)
                
                logger.info(f"Manual parameter override applied: {parameter_updates}, "
                           f"safety_bounded={was_bounded}")
                
                return {
                    'success': True,
                    'parameters': self._get_current_parameters_dict(),
                    'safety_bounded': was_bounded,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Manual override error: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_adaptation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent adaptation history"""
        with self.lock:
            recent_events = list(self.adaptation_history)[-limit:]
            return [asdict(event) for event in recent_events]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        with self.lock:
            if not self.performance_history:
                return {'status': 'no_data'}
            
            recent_perf = list(self.performance_history)[-50:]  # Last 50 measurements
            
            # Extract metrics
            cbu_values = [p['metrics'].get('cbu_per_ms', 0) for p in recent_perf]
            latency_values = [p['metrics'].get('p95_latency', 0) for p in recent_perf]
            complexity_values = [p['context_complexity'] for p in recent_perf]
            
            # Calculate statistics
            summary = {
                'measurements_count': len(recent_perf),
                'time_range_minutes': (recent_perf[-1]['timestamp'] - recent_perf[0]['timestamp']).total_seconds() / 60,
                'cbu_per_ms': {
                    'mean': statistics.mean(cbu_values) if cbu_values else 0,
                    'std': statistics.stdev(cbu_values) if len(cbu_values) > 1 else 0,
                    'min': min(cbu_values) if cbu_values else 0,
                    'max': max(cbu_values) if cbu_values else 0
                },
                'p95_latency': {
                    'mean': statistics.mean(latency_values) if latency_values else 0,
                    'std': statistics.stdev(latency_values) if len(latency_values) > 1 else 0,
                    'min': min(latency_values) if latency_values else 0,
                    'max': max(latency_values) if latency_values else 0
                },
                'context_complexity': {
                    'mean': statistics.mean(complexity_values) if complexity_values else 0,
                    'std': statistics.stdev(complexity_values) if len(complexity_values) > 1 else 0
                },
                'adaptations_count': len(self.adaptation_history),
                'current_parameters': self._get_current_parameters_dict()
            }
            
            return summary

# Factory function for easy instantiation
def create_adaptive_control_surface(config: Optional[Dict[str, Any]] = None) -> AdaptiveControlSurface:
    """Create adaptive control surface with optional configuration"""
    bounds = ParameterBounds()
    
    if config:
        # Override bounds from config
        for attr in ['lambda_min', 'lambda_max', 'lambda_default',
                     'mu_window_min', 'mu_window_max', 'mu_window_default',
                     'r_min', 'r_max', 'r_default',
                     'k2_min', 'k2_max', 'k2_default']:
            if attr in config:
                setattr(bounds, attr, config[attr])
    
    return AdaptiveControlSurface(bounds)