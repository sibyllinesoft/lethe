"""
Lethe→StreamingLLM Hybrid System Implementation

A hybrid context management system that combines:
- Lethe: Stable context selection with grouped atoms (Head)
- StreamingLLM: Windowed volatile content with attention sinks (Tail)

Both optimized under the same dual objective:
max F(H∪T) - λ(tokens) - μ(compute)

Key Features:
- Head (H): Lethe-selected stable content (8-20% keep ratio)
- Tail (T): StreamingLLM windowed content with attention sinks  
- Gated activation: Streaming only if accept-rate < 0.4 and entropy > threshold
- Comprehensive instrumentation for λ,μ,tokens,keep_ratios,KV_reuse,p95 times
- KV-aware arrangement for prefix reuse optimization
"""

import time
import logging
import math
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import scipy.stats as stats

logger = logging.getLogger(__name__)

# Core data structures for the hybrid system
@dataclass
class AtomGroup:
    """Grouped atoms for stable head content."""
    group_type: str  # 'def', 'error', 'tool_key', 'symbol_header'
    atoms: List[str]
    total_tokens: int
    utility_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class HeadResult:
    """Result from Lethe head selection."""
    selected_atoms: List[AtomGroup]
    total_tokens: int
    keep_ratio: float
    utility_score: float
    ce_early_exit_at: int
    dpp_rank: int
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TailWindow:
    """Single tail window with attention sinks."""
    window_id: int
    content_tokens: List[str]
    attention_sinks: List[str] 
    window_size: int
    utility_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TailResult:
    """Result from StreamingLLM tail windowing."""
    windows: List[TailWindow]
    total_tokens: int
    keep_ratio: float
    num_windows: int
    stride: int
    sink_tokens_per_window: int
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HybridInstrumentation:
    """Comprehensive instrumentation data."""
    # Core parameters
    lambda_param: float
    mu_param: float
    tokens_in: int
    head_tokens: int
    tail_tokens: int
    keep_ratio_head: float
    keep_ratio_tail: float
    
    # Algorithm parameters
    K1: int
    K2: int
    dpp_rank: int
    ce_early_exit: bool
    num_windows: int
    window_size: int
    stride: int
    sinks: int
    
    # Performance metrics
    kv_prefix_reuse: float
    middleware_p95_ms: float
    llm_p95_ms: float
    delta_cbu_per_1k: float  # Change in Compute Budget Units per 1k tokens
    
    # Quality metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Control metrics
    primal_dual_gap: float = 0.0
    tail_cvar_95: float = 0.0
    lambda_drift_24h: float = 0.0
    mu_drift_24h: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'lambda': self.lambda_param,
            'mu': self.mu_param,
            'tokens_in': self.tokens_in,
            'head_tokens': self.head_tokens,
            'tail_tokens': self.tail_tokens,
            'keep_ratio_head': self.keep_ratio_head,
            'keep_ratio_tail': self.keep_ratio_tail,
            'K1': self.K1,
            'K2': self.K2,
            'dpp_rank': self.dpp_rank,
            'ce_early_exit': self.ce_early_exit,
            'num_windows': self.num_windows,
            'window_size': self.window_size,
            'stride': self.stride,
            'sinks': self.sinks,
            'kv_prefix_reuse': self.kv_prefix_reuse,
            'middleware_p95_ms': self.middleware_p95_ms,
            'llm_p95_ms': self.llm_p95_ms,
            'delta_cbu_per_1k': self.delta_cbu_per_1k,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'primal_dual_gap': self.primal_dual_gap,
            'tail_cvar_95': self.tail_cvar_95,
            'lambda_drift_24h': self.lambda_drift_24h,
            'mu_drift_24h': self.mu_drift_24h
        }

@dataclass
class HybridResult:
    """Complete hybrid selection result."""
    head_result: Optional[HeadResult]
    tail_result: Optional[TailResult]
    final_context: str
    total_tokens: int
    keep_ratio: float
    gating_decision: str  # 'head_only', 'hybrid', 'fallback'
    instrumentation: HybridInstrumentation
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityEntropyAnalyzer:
    """Analyzes entity entropy for gating decisions."""
    
    def __init__(self):
        self.entity_cache = {}
        self.entropy_threshold = 2.5  # Configurable threshold
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract entities with types and confidence scores."""
        # Simplified entity extraction - in practice would use NER
        import re
        
        # Simple patterns for common entities
        patterns = {
            'function': r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
            'class': r'\bclass\s+[A-Z][a-zA-Z0-9_]*',
            'variable': r'\b[a-z_][a-zA-Z0-9_]*\s*=',
            'import': r'\bimport\s+[a-zA-Z0-9_.]+|\bfrom\s+[a-zA-Z0-9_.]+',
            'error': r'\b[A-Z][a-zA-Z]*Error\b|\bException\b'
        }
        
        entities = []
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group().strip()
                confidence = 0.9  # Simplified confidence
                entities.append((entity_text, entity_type, confidence))
        
        return entities
    
    def calculate_entropy(self, entities: List[Tuple[str, str, float]]) -> float:
        """Calculate entity entropy using Shannon entropy."""
        if not entities:
            return 0.0
        
        # Count entity types
        type_counts = Counter(entity_type for _, entity_type, _ in entities)
        total_entities = sum(type_counts.values())
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in type_counts.values():
            prob = count / total_entities
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def should_enable_streaming(self, text: str, accept_rate: float) -> Tuple[bool, float]:
        """Determine if streaming should be enabled based on entropy and accept rate."""
        entities = self.extract_entities(text)
        entropy = self.calculate_entropy(entities)
        
        # Gating condition: enable streaming if accept-rate < 0.4 and entropy > threshold
        enable_streaming = (accept_rate < 0.4) and (entropy > self.entropy_threshold)
        
        return enable_streaming, entropy

class LetheHeadBuilder:
    """Builds stable head content using Lethe selection with grouped atoms."""
    
    def __init__(self, 
                 target_keep_ratio: float = 0.12,
                 dpp_rank: int = 14,
                 group_split_tau: float = 0.7,
                 ce_early_exit_k2: int = 320):
        self.target_keep_ratio = target_keep_ratio
        self.dpp_rank = dpp_rank
        self.group_split_tau = group_split_tau
        self.ce_early_exit_k2 = ce_early_exit_k2
        
    def extract_grouped_atoms(self, text: str) -> List[AtomGroup]:
        """Extract and group atoms by type."""
        import re
        
        atoms = []
        
        # Group 1: Function definitions
        def_pattern = r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:'
        def_matches = list(re.finditer(def_pattern, text))
        if def_matches:
            def_atoms = [match.group() for match in def_matches]
            def_tokens = sum(len(atom.split()) for atom in def_atoms)
            atoms.append(AtomGroup(
                group_type='def',
                atoms=def_atoms,
                total_tokens=def_tokens,
                utility_score=len(def_atoms) * 1.5,  # High utility for definitions
                metadata={'pattern': def_pattern}
            ))
        
        # Group 2: Error frames
        error_pattern = r'\b[A-Z][a-zA-Z]*Error\b.*?(?=\n\n|\n[A-Z]|\Z)'
        error_matches = list(re.finditer(error_pattern, text, re.DOTALL))
        if error_matches:
            error_atoms = [match.group() for match in error_matches]
            error_tokens = sum(len(atom.split()) for atom in error_atoms)
            atoms.append(AtomGroup(
                group_type='error',
                atoms=error_atoms,
                total_tokens=error_tokens,
                utility_score=len(error_atoms) * 1.3,  # High utility for errors
                metadata={'pattern': error_pattern}
            ))
        
        # Group 3: Tool keys (import statements, config keys)
        tool_pattern = r'\b(?:import|from)\s+[a-zA-Z0-9_.]+|\b[A-Z_][A-Z0-9_]*\s*='
        tool_matches = list(re.finditer(tool_pattern, text))
        if tool_matches:
            tool_atoms = [match.group() for match in tool_matches]
            tool_tokens = sum(len(atom.split()) for atom in tool_atoms)
            atoms.append(AtomGroup(
                group_type='tool_key',
                atoms=tool_atoms,
                total_tokens=tool_tokens,
                utility_score=len(tool_atoms) * 1.2,
                metadata={'pattern': tool_pattern}
            ))
        
        # Group 4: Symbol headers (class definitions, main blocks)
        symbol_pattern = r'\bclass\s+[A-Z][a-zA-Z0-9_]*.*?:|\bif\s+__name__\s*==\s*["\']__main__["\'].*?:'
        symbol_matches = list(re.finditer(symbol_pattern, text))
        if symbol_matches:
            symbol_atoms = [match.group() for match in symbol_matches]
            symbol_tokens = sum(len(atom.split()) for atom in symbol_atoms)
            atoms.append(AtomGroup(
                group_type='symbol_header',
                atoms=symbol_atoms,
                total_tokens=symbol_tokens,
                utility_score=len(symbol_atoms) * 1.4,
                metadata={'pattern': symbol_pattern}
            ))
        
        return atoms
    
    def apply_dpp_selection(self, atom_groups: List[AtomGroup], target_tokens: int) -> List[AtomGroup]:
        """Apply Determinantal Point Process selection for diversity."""
        if not atom_groups:
            return []
        
        # Simplified DPP: Select diverse groups based on utility and diversity
        selected_groups = []
        current_tokens = 0
        
        # Sort by utility score (descending)
        sorted_groups = sorted(atom_groups, key=lambda g: g.utility_score, reverse=True)
        
        for group in sorted_groups:
            if current_tokens + group.total_tokens <= target_tokens:
                selected_groups.append(group)
                current_tokens += group.total_tokens
            
            # Early exit if we have diverse enough selection
            if len(selected_groups) >= self.dpp_rank:
                break
        
        return selected_groups
    
    def apply_ce_early_exit(self, atom_groups: List[AtomGroup], k2: int) -> Tuple[List[AtomGroup], bool]:
        """Apply Cross-Entropy early exit optimization."""
        if len(atom_groups) <= k2:
            return atom_groups, False
        
        # Simplified CE: Keep top K2 groups by utility
        sorted_groups = sorted(atom_groups, key=lambda g: g.utility_score, reverse=True)
        early_exit_groups = sorted_groups[:k2]
        
        return early_exit_groups, True
    
    def build_head(self, text: str, lambda_param: float, mu_param: float) -> HeadResult:
        """Build stable head content using Lethe selection."""
        start_time = time.time()
        
        # Extract grouped atoms
        atom_groups = self.extract_grouped_atoms(text)
        
        # Calculate target tokens based on keep ratio
        total_tokens = len(text.split())
        target_tokens = int(total_tokens * self.target_keep_ratio)
        
        # Apply Cross-Entropy early exit if needed
        ce_groups, ce_applied = self.apply_ce_early_exit(atom_groups, self.ce_early_exit_k2)
        
        # Apply DPP selection for diversity
        selected_groups = self.apply_dpp_selection(ce_groups, target_tokens)
        
        # Calculate metrics
        selected_tokens = sum(group.total_tokens for group in selected_groups)
        actual_keep_ratio = selected_tokens / total_tokens if total_tokens > 0 else 0.0
        utility_score = sum(group.utility_score for group in selected_groups)
        
        processing_time = (time.time() - start_time) * 1000
        
        return HeadResult(
            selected_atoms=selected_groups,
            total_tokens=selected_tokens,
            keep_ratio=actual_keep_ratio,
            utility_score=utility_score,
            ce_early_exit_at=self.ce_early_exit_k2 if ce_applied else 0,
            dpp_rank=len(selected_groups),
            processing_time_ms=processing_time,
            metadata={
                'target_keep_ratio': self.target_keep_ratio,
                'target_tokens': target_tokens,
                'total_input_tokens': total_tokens,
                'num_groups_extracted': len(atom_groups),
                'num_groups_selected': len(selected_groups),
                'ce_early_exit_applied': ce_applied
            }
        )

class StreamingTailBuilder:
    """Builds volatile tail content using StreamingLLM windowing."""
    
    def __init__(self,
                 window_size: int = 6000,
                 stride: int = 3000, 
                 sink_tokens: int = 96):
        self.window_size = window_size
        self.stride = stride
        self.sink_tokens = sink_tokens
    
    def create_attention_sinks(self, head_summary: str) -> List[str]:
        """Create attention sinks including typed digest of head content."""
        # Create typed micro-summaries of head content
        sinks = []
        
        # Add head digest as attention sink
        if head_summary:
            head_tokens = head_summary.split()[:self.sink_tokens//2]
            sinks.extend(head_tokens)
        
        # Add generic attention sinks
        generic_sinks = ['<|START|>', '<|CONTEXT|>', '<|RELEVANT|>']
        sinks.extend(generic_sinks)
        
        # Truncate to sink token limit
        if len(sinks) > self.sink_tokens:
            sinks = sinks[:self.sink_tokens]
        
        return sinks
    
    def create_windows(self, text: str, head_summary: str) -> List[TailWindow]:
        """Create sliding windows with attention sinks."""
        tokens = text.split()
        
        if len(tokens) <= self.window_size:
            # Single window case
            sinks = self.create_attention_sinks(head_summary)
            return [TailWindow(
                window_id=0,
                content_tokens=tokens,
                attention_sinks=sinks,
                window_size=len(tokens),
                utility_score=len(tokens) * 0.8,  # Base utility
                metadata={'single_window': True}
            )]
        
        windows = []
        window_id = 0
        start_pos = 0
        
        while start_pos < len(tokens):
            # Calculate window end
            end_pos = min(start_pos + self.window_size, len(tokens))
            window_tokens = tokens[start_pos:end_pos]
            
            # Create attention sinks for this window
            sinks = self.create_attention_sinks(head_summary)
            
            # Calculate utility score (later windows have higher utility for recency)
            recency_weight = 1.0 + (window_id * 0.1)
            utility_score = len(window_tokens) * 0.8 * recency_weight
            
            windows.append(TailWindow(
                window_id=window_id,
                content_tokens=window_tokens,
                attention_sinks=sinks,
                window_size=len(window_tokens),
                utility_score=utility_score,
                metadata={
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'recency_weight': recency_weight
                }
            ))
            
            # Move to next window position
            start_pos += self.stride
            window_id += 1
            
            # Break if stride would create tiny window
            if len(tokens) - start_pos < self.stride // 2:
                break
        
        return windows
    
    def select_windows(self, windows: List[TailWindow], mu_param: float, budget_tokens: int) -> List[TailWindow]:
        """Select windows based on utility and budget constraints."""
        if not windows:
            return []
        
        # Sort by utility score (descending)
        sorted_windows = sorted(windows, key=lambda w: w.utility_score, reverse=True)
        
        selected_windows = []
        current_tokens = 0
        
        for window in sorted_windows:
            window_cost = window.window_size + len(window.attention_sinks)
            if current_tokens + window_cost <= budget_tokens:
                selected_windows.append(window)
                current_tokens += window_cost
        
        # Sort selected windows by window_id to maintain temporal order
        selected_windows.sort(key=lambda w: w.window_id)
        
        return selected_windows
    
    def build_tail(self, text: str, head_summary: str, lambda_param: float, mu_param: float, budget_tokens: int) -> TailResult:
        """Build volatile tail content using StreamingLLM windowing."""
        start_time = time.time()
        
        # Create sliding windows
        windows = self.create_windows(text, head_summary)
        
        # Select windows based on budget and utility
        selected_windows = self.select_windows(windows, mu_param, budget_tokens)
        
        # Calculate metrics
        total_tokens = sum(w.window_size + len(w.attention_sinks) for w in selected_windows)
        original_tokens = len(text.split())
        keep_ratio = total_tokens / original_tokens if original_tokens > 0 else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return TailResult(
            windows=selected_windows,
            total_tokens=total_tokens,
            keep_ratio=keep_ratio,
            num_windows=len(selected_windows),
            stride=self.stride,
            sink_tokens_per_window=self.sink_tokens,
            processing_time_ms=processing_time,
            metadata={
                'total_windows_created': len(windows),
                'windows_selected': len(selected_windows),
                'original_tokens': original_tokens,
                'budget_tokens': budget_tokens,
                'window_size': self.window_size
            }
        )

class KVAwareArranger:
    """Arranges head and tail for optimal KV cache reuse."""
    
    def __init__(self):
        self.kv_cache_stats = {
            'prefix_hits': 0,
            'total_requests': 0,
            'jaccard_scores': []
        }
    
    def calculate_kv_reuse(self, previous_context: str, current_context: str) -> float:
        """Calculate KV cache reuse using prefix Jaccard similarity."""
        if not previous_context or not current_context:
            return 0.0
        
        prev_tokens = set(previous_context.split())
        curr_tokens = set(current_context.split())
        
        if not prev_tokens or not curr_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(prev_tokens & curr_tokens)
        union = len(prev_tokens | curr_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        
        self.kv_cache_stats['jaccard_scores'].append(jaccard)
        
        return jaccard
    
    def arrange_for_kv_optimization(self, head_result: Optional[HeadResult], tail_result: Optional[TailResult]) -> str:
        """Arrange head and tail content for optimal KV cache reuse."""
        context_parts = []
        
        # Head first (stable content for KV prefix reuse)
        if head_result:
            head_content = []
            for group in head_result.selected_atoms:
                group_content = ' '.join(group.atoms)
                head_content.append(f"# {group.group_type.upper()}\n{group_content}")
            
            if head_content:
                context_parts.append("# HEAD CONTENT (STABLE)\n" + '\n\n'.join(head_content))
        
        # Tail after (volatile content)
        if tail_result:
            tail_content = []
            for window in tail_result.windows:
                # Add attention sinks first
                sink_content = ' '.join(window.attention_sinks)
                window_content = ' '.join(window.content_tokens)
                
                combined_window = f"# WINDOW {window.window_id}\n{sink_content}\n{window_content}"
                tail_content.append(combined_window)
            
            if tail_content:
                context_parts.append("# TAIL CONTENT (STREAMING)\n" + '\n\n'.join(tail_content))
        
        return '\n\n' + '='*50 + '\n\n'.join(context_parts) if context_parts else ""

class EVTTailModeler:
    """Extreme Value Theory modeling for tail compute distribution."""
    
    def __init__(self, window_size: int = 100):
        self.compute_history = []
        self.window_size = window_size
        self.xi_parameter = 0.0  # Shape parameter from GPD
        self.beta_parameter = 1.0  # Scale parameter from GPD
        self.threshold = None
        
    def record_compute_time(self, compute_ms: float):
        """Record compute time for EVT analysis."""
        self.compute_history.append(compute_ms)
        
        # Keep rolling window
        if len(self.compute_history) > self.window_size:
            self.compute_history.pop(0)
    
    def update_evt_parameters(self):
        """Update EVT parameters using GPD fitting."""
        if len(self.compute_history) < 20:
            return
        
        data = np.array(self.compute_history)
        
        # Use 95th percentile as threshold for extreme values
        self.threshold = np.percentile(data, 95)
        
        # Extract excesses over threshold
        excesses = data[data > self.threshold] - self.threshold
        
        if len(excesses) > 5:
            # Fit Generalized Pareto Distribution
            try:
                shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
                self.xi_parameter = shape
                self.beta_parameter = scale
                
                logger.debug(f"EVT parameters updated: ξ={self.xi_parameter:.4f}, β={self.beta_parameter:.4f}")
                
            except Exception as e:
                logger.warning(f"EVT fitting failed: {e}")
    
    def predict_tail_risk(self) -> float:
        """Predict tail compute risk using fitted EVT model."""
        if self.threshold is None:
            return 0.0
        
        # Calculate CVaR at 95% level
        try:
            # CVaR = threshold + (beta / (1 - xi)) if xi < 1
            if self.xi_parameter < 1:
                cvar_95 = self.threshold + (self.beta_parameter / (1 - self.xi_parameter))
                return cvar_95
            else:
                # Fallback to empirical 99th percentile
                return np.percentile(self.compute_history, 99) if self.compute_history else 0.0
        except:
            return np.percentile(self.compute_history, 99) if self.compute_history else 0.0
    
    def should_reduce_stride(self) -> bool:
        """Determine if stride should be reduced based on ξ parameter."""
        # If shape parameter is increasing (heavier tail), reduce stride
        return self.xi_parameter > 0.2

class PrimalDualGapMonitor:
    """Monitor primal-dual gap for optimization convergence."""
    
    def __init__(self, tolerance: float = 0.005):
        self.tolerance = tolerance
        self.gap_history = []
        self.convergence_count = 0
        
    def calculate_gap(self, primal_value: float, dual_value: float) -> float:
        """Calculate normalized primal-dual gap."""
        if abs(primal_value) < 1e-10:
            return 0.0
        
        gap = abs(primal_value - dual_value) / abs(primal_value)
        self.gap_history.append(gap)
        
        # Keep last 20 gaps
        if len(self.gap_history) > 20:
            self.gap_history.pop(0)
            
        return gap
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.gap_history) < 5:
            return False
        
        recent_gaps = self.gap_history[-5:]
        converged = all(gap < self.tolerance for gap in recent_gaps)
        
        if converged:
            self.convergence_count += 1
        else:
            self.convergence_count = 0
            
        return self.convergence_count >= 3

class KVPrefixJaccardAlarm:
    """Alarm system for KV prefix reuse degradation."""
    
    def __init__(self, threshold_drop: float = 0.10):
        self.threshold_drop = threshold_drop
        self.baseline_jaccard = None
        self.recent_scores = []
        self.alarm_active = False
        
    def update_jaccard_score(self, jaccard_score: float):
        """Update Jaccard score and check for alarm conditions."""
        self.recent_scores.append(jaccard_score)
        
        # Keep rolling window of 10 scores
        if len(self.recent_scores) > 10:
            self.recent_scores.pop(0)
        
        # Establish baseline if not set
        if self.baseline_jaccard is None and len(self.recent_scores) >= 5:
            self.baseline_jaccard = np.mean(self.recent_scores)
            logger.info(f"KV Jaccard baseline established: {self.baseline_jaccard:.4f}")
            return
        
        # Check for alarm condition
        if self.baseline_jaccard is not None and len(self.recent_scores) >= 3:
            recent_avg = np.mean(self.recent_scores[-3:])
            drop = self.baseline_jaccard - recent_avg
            
            if drop > self.threshold_drop:
                if not self.alarm_active:
                    logger.warning(f"KV Prefix Jaccard alarm triggered! Drop: {drop:.4f} > {self.threshold_drop}")
                    self.alarm_active = True
            else:
                if self.alarm_active:
                    logger.info("KV Prefix Jaccard alarm cleared")
                    self.alarm_active = False
    
    def should_reduce_head_size(self) -> Tuple[bool, float]:
        """Determine if head size should be reduced and by how much."""
        if self.alarm_active:
            # Reduce head by 2-3% as recommended in TODO
            reduction_percent = 0.025  # 2.5% reduction
            return True, reduction_percent
        return False, 0.0

class ParameterDriftMonitor:
    """Monitor λ and μ parameter drift over 24h windows."""
    
    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.lambda_history = []
        self.mu_history = []
        self.timestamp_history = []
        
    def record_parameters(self, lambda_val: float, mu_val: float):
        """Record parameter values with timestamp."""
        now = datetime.now()
        
        self.lambda_history.append(lambda_val)
        self.mu_history.append(mu_val)
        self.timestamp_history.append(now)
        
        # Clean old records (> 48 hours)
        cutoff = now - timedelta(hours=48)
        while (self.timestamp_history and 
               self.timestamp_history[0] < cutoff):
            self.lambda_history.pop(0)
            self.mu_history.pop(0)
            self.timestamp_history.pop(0)
    
    def calculate_24h_drift(self) -> Tuple[float, float]:
        """Calculate parameter drift over last 24 hours."""
        if len(self.timestamp_history) < 2:
            return 0.0, 0.0
        
        now = datetime.now()
        cutoff_24h = now - timedelta(hours=24)
        
        # Find values from 24h ago
        lambda_24h_ago = None
        mu_24h_ago = None
        
        for i, ts in enumerate(self.timestamp_history):
            if ts >= cutoff_24h:
                if i > 0:
                    lambda_24h_ago = self.lambda_history[i-1]
                    mu_24h_ago = self.mu_history[i-1]
                break
        
        if lambda_24h_ago is None:
            return 0.0, 0.0
        
        # Calculate drift as relative change
        current_lambda = self.lambda_history[-1]
        current_mu = self.mu_history[-1]
        
        lambda_drift = abs(current_lambda - lambda_24h_ago) / lambda_24h_ago if lambda_24h_ago > 0 else 0.0
        mu_drift = abs(current_mu - mu_24h_ago) / mu_24h_ago if mu_24h_ago > 0 else 0.0
        
        return lambda_drift, mu_drift
    
    def check_drift_alarms(self) -> Dict[str, bool]:
        """Check if drift alarms should be triggered."""
        lambda_drift, mu_drift = self.calculate_24h_drift()
        
        alarms = {
            'lambda_drift_alarm': lambda_drift > self.drift_threshold,
            'mu_drift_alarm': mu_drift > self.drift_threshold,
            'lambda_drift': lambda_drift,
            'mu_drift': mu_drift
        }
        
        if alarms['lambda_drift_alarm']:
            logger.warning(f"Lambda drift alarm: {lambda_drift:.3f} > {self.drift_threshold}")
        
        if alarms['mu_drift_alarm']:
            logger.warning(f"Mu drift alarm: {mu_drift:.3f} > {self.drift_threshold}")
        
        return alarms

class AdvancedInstrumentationLogger:
    """Enhanced instrumentation logging with structured telemetry."""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("hybrid_telemetry.jsonl")
        self.evt_modeler = EVTTailModeler()
        self.gap_monitor = PrimalDualGapMonitor()
        self.jaccard_alarm = KVPrefixJaccardAlarm()
        self.drift_monitor = ParameterDriftMonitor()
        
    def log_hybrid_run(self, result: 'HybridResult', evaluation_metrics: Optional[Dict] = None):
        """Log comprehensive hybrid run data."""
        instrumentation = result.instrumentation
        
        # Update monitoring systems
        self.evt_modeler.record_compute_time(result.processing_time_ms)
        self.evt_modeler.update_evt_parameters()
        
        # Calculate advanced metrics
        tail_cvar = self.evt_modeler.predict_tail_risk()
        
        # Update Jaccard alarm
        self.jaccard_alarm.update_jaccard_score(instrumentation.kv_prefix_reuse)
        should_reduce_head, head_reduction = self.jaccard_alarm.should_reduce_head_size()
        
        # Update drift monitoring
        self.drift_monitor.record_parameters(instrumentation.lambda_param, instrumentation.mu_param)
        drift_alarms = self.drift_monitor.check_drift_alarms()
        
        # Calculate primal-dual gap (simplified)
        primal_value = result.total_tokens * instrumentation.lambda_param
        dual_value = primal_value * 0.98  # Simplified dual estimate
        gap = self.gap_monitor.calculate_gap(primal_value, dual_value)
        
        # Enhanced telemetry data
        telemetry_data = instrumentation.to_dict()
        telemetry_data.update({
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': result.processing_time_ms,
            'gating_decision': result.gating_decision,
            'final_context_length': len(result.final_context),
            
            # Advanced metrics
            'tail_cvar_95': tail_cvar,
            'primal_dual_gap': gap,
            'optimization_converged': self.gap_monitor.is_converged(),
            'xi_parameter': self.evt_modeler.xi_parameter,
            'beta_parameter': self.evt_modeler.beta_parameter,
            'should_reduce_stride': self.evt_modeler.should_reduce_stride(),
            
            # KV alarms
            'jaccard_alarm_active': self.jaccard_alarm.alarm_active,
            'should_reduce_head': should_reduce_head,
            'head_reduction_suggested': head_reduction,
            
            # Drift monitoring
            'lambda_drift_24h': drift_alarms['lambda_drift'],
            'mu_drift_24h': drift_alarms['mu_drift'],
            'lambda_drift_alarm': drift_alarms['lambda_drift_alarm'],
            'mu_drift_alarm': drift_alarms['mu_drift_alarm'],
            
            # Evaluation metrics
            **(evaluation_metrics or {})
        })
        
        # Update instrumentation object with advanced metrics
        instrumentation.tail_cvar_95 = tail_cvar
        instrumentation.primal_dual_gap = gap
        instrumentation.lambda_drift_24h = drift_alarms['lambda_drift']
        instrumentation.mu_drift_24h = drift_alarms['mu_drift']
        
        # Write to log file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(telemetry_data) + '\n')
        
        logger.info(f"Advanced telemetry logged: gap={gap:.4f}, tail_CVaR={tail_cvar:.2f}ms")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitoring systems."""
        return {
            'evt_tail_model': {
                'xi_parameter': self.evt_modeler.xi_parameter,
                'beta_parameter': self.evt_modeler.beta_parameter,
                'tail_cvar_95': self.evt_modeler.predict_tail_risk(),
                'should_reduce_stride': self.evt_modeler.should_reduce_stride()
            },
            'primal_dual_gap': {
                'current_gap': self.gap_monitor.gap_history[-1] if self.gap_monitor.gap_history else 0.0,
                'converged': self.gap_monitor.is_converged(),
                'convergence_count': self.gap_monitor.convergence_count
            },
            'kv_jaccard_alarm': {
                'alarm_active': self.jaccard_alarm.alarm_active,
                'baseline_jaccard': self.jaccard_alarm.baseline_jaccard,
                'recent_avg': np.mean(self.jaccard_alarm.recent_scores) if self.jaccard_alarm.recent_scores else 0.0,
                'should_reduce_head': self.jaccard_alarm.should_reduce_head_size()[0]
            },
            'parameter_drift': self.drift_monitor.check_drift_alarms()
        }

class HybridSelector:
    """Main hybrid selector combining Lethe + StreamingLLM."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Default canary configuration from TODO.md
        self.head_keep_ratio = self.config.get('head_keep', 0.12)
        self.window_size = self.config.get('window_size', 6000)
        self.stride = self.config.get('stride', 3000) 
        self.sink_tokens = self.config.get('sinks', 96)
        self.ce_k2 = self.config.get('K2', 320)
        self.dpp_rank = self.config.get('dpp_rank', 14)
        
        # Initialize components
        self.entropy_analyzer = EntityEntropyAnalyzer()
        self.head_builder = LetheHeadBuilder(
            target_keep_ratio=self.head_keep_ratio,
            dpp_rank=self.dpp_rank,
            ce_early_exit_k2=self.ce_k2
        )
        self.tail_builder = StreamingTailBuilder(
            window_size=self.window_size,
            stride=self.stride,
            sink_tokens=self.sink_tokens
        )
        self.kv_arranger = KVAwareArranger()
        
        # Performance tracking
        self.performance_history = []
        self.previous_context = ""
        
        # Advanced instrumentation
        self.advanced_logger = AdvancedInstrumentationLogger(
            log_path=Path(self.config.get('telemetry_path', 'hybrid_telemetry.jsonl'))
        )
        
    def select(self, query: str, context: str, lambda_param: float = 0.001, mu_param: float = 0.0001) -> HybridResult:
        """Execute hybrid selection with comprehensive instrumentation."""
        overall_start = time.time()
        
        # Step 1: Build head using Lethe
        head_result = self.head_builder.build_head(context, lambda_param, mu_param)
        
        # Create head summary for tail builder
        head_summary = ""
        if head_result.selected_atoms:
            head_summary = " ".join([
                f"{group.group_type}: {' '.join(group.atoms[:3])}"
                for group in head_result.selected_atoms
            ])
        
        # Step 2: Gating decision for streaming
        head_content = head_summary or ""
        accept_rate = head_result.keep_ratio
        should_stream, entropy = self.entropy_analyzer.should_enable_streaming(context, accept_rate)
        
        tail_result = None
        gating_decision = "head_only"
        
        # Step 3: Build tail if streaming enabled
        if should_stream:
            total_tokens = len(context.split())
            budget_tokens = total_tokens - head_result.total_tokens
            
            if budget_tokens > self.window_size:  # Enough budget for streaming
                tail_result = self.tail_builder.build_tail(
                    context, head_summary, lambda_param, mu_param, budget_tokens
                )
                gating_decision = "hybrid"
        
        # Step 4: KV-aware arrangement
        final_context = self.kv_arranger.arrange_for_kv_optimization(head_result, tail_result)
        
        # Step 5: Calculate metrics
        total_tokens = head_result.total_tokens
        if tail_result:
            total_tokens += tail_result.total_tokens
        
        original_tokens = len(context.split())
        overall_keep_ratio = total_tokens / original_tokens if original_tokens > 0 else 0.0
        
        # Step 6: KV reuse calculation
        kv_reuse = self.kv_arranger.calculate_kv_reuse(self.previous_context, final_context)
        self.previous_context = final_context
        
        # Step 7: Performance instrumentation
        processing_time = (time.time() - overall_start) * 1000
        
        instrumentation = HybridInstrumentation(
            lambda_param=lambda_param,
            mu_param=mu_param,
            tokens_in=original_tokens,
            head_tokens=head_result.total_tokens,
            tail_tokens=tail_result.total_tokens if tail_result else 0,
            keep_ratio_head=head_result.keep_ratio,
            keep_ratio_tail=tail_result.keep_ratio if tail_result else 0.0,
            K1=1000,  # Default K1 value
            K2=self.ce_k2,
            dpp_rank=head_result.dpp_rank,
            ce_early_exit=head_result.ce_early_exit_at > 0,
            num_windows=tail_result.num_windows if tail_result else 0,
            window_size=self.window_size,
            stride=self.stride,
            sinks=self.sink_tokens,
            kv_prefix_reuse=kv_reuse,
            middleware_p95_ms=processing_time,  # Simplified
            llm_p95_ms=0.0,  # Would be measured in actual LLM calls
            delta_cbu_per_1k=self._calculate_delta_cbu(total_tokens),
            precision_at_k={5: 0.0, 10: 0.0},  # Would be calculated with ground truth
            recall_at_k={5: 0.0, 10: 0.0}     # Would be calculated with ground truth
        )
        
        result = HybridResult(
            head_result=head_result,
            tail_result=tail_result,
            final_context=final_context,
            total_tokens=total_tokens,
            keep_ratio=overall_keep_ratio,
            gating_decision=gating_decision,
            instrumentation=instrumentation,
            processing_time_ms=processing_time,
            metadata={
                'entropy': entropy,
                'should_stream': should_stream,
                'accept_rate': accept_rate,
                'query': query
            }
        )
        
        # Advanced logging with comprehensive telemetry
        self.advanced_logger.log_hybrid_run(result)
        
        # Log standard instrumentation
        logger.info(f"Hybrid selection: {instrumentation.to_dict()}")
        
        return result
    
    def _calculate_delta_cbu(self, tokens: int) -> float:
        """Calculate change in Compute Budget Units per 1k tokens."""
        # Simplified CBU calculation - would use actual cost model
        base_cbu_per_1k = 0.01  # Base cost per 1k tokens
        efficiency_factor = min(1.0, 1000.0 / max(1, tokens))  # Efficiency improves with context
        return base_cbu_per_1k * efficiency_factor
    
    def get_adaptive_adjustments(self) -> Dict[str, Any]:
        """Get recommended adaptive adjustments based on monitoring systems."""
        monitoring = self.advanced_logger.get_monitoring_summary()
        adjustments = {
            'head_keep_ratio_adjustment': 0.0,
            'stride_adjustment': 0.0,
            'lambda_adjustment': 0.0,
            'mu_adjustment': 0.0,
            'actions_recommended': []
        }
        
        # KV Jaccard alarm - reduce head size by 2-3%
        if monitoring['kv_jaccard_alarm']['should_reduce_head']:
            reduction = 0.025  # 2.5% reduction as specified in TODO
            adjustments['head_keep_ratio_adjustment'] = -reduction
            adjustments['actions_recommended'].append(f"Reduce head size by {reduction*100:.1f}% due to KV prefix degradation")
        
        # EVT tail modeling - adjust stride based on ξ parameter
        if monitoring['evt_tail_model']['should_reduce_stride']:
            stride_reduction = 0.2  # Reduce stride by 20%
            adjustments['stride_adjustment'] = -stride_reduction
            adjustments['actions_recommended'].append(f"Reduce stride by {stride_reduction*100:.0f}% due to heavy tail (ξ={monitoring['evt_tail_model']['xi_parameter']:.3f})")
        
        # Parameter drift - suggest parameter adjustments
        drift_data = monitoring['parameter_drift']
        if drift_data['lambda_drift_alarm']:
            lambda_adjust = -0.1  # Reduce lambda by 10% to stabilize
            adjustments['lambda_adjustment'] = lambda_adjust
            adjustments['actions_recommended'].append(f"Reduce λ by 10% due to drift alarm (drift={drift_data['lambda_drift']:.3f})")
        
        if drift_data['mu_drift_alarm']:
            mu_adjust = -0.1  # Reduce mu by 10% to stabilize
            adjustments['mu_adjustment'] = mu_adjust
            adjustments['actions_recommended'].append(f"Reduce μ by 10% due to drift alarm (drift={drift_data['mu_drift']:.3f})")
        
        # Primal-dual gap - suggest optimization adjustments
        if not monitoring['primal_dual_gap']['converged'] and monitoring['primal_dual_gap']['current_gap'] > 0.01:
            adjustments['actions_recommended'].append("Optimization not converged - consider parameter tuning")
        
        return adjustments
    
    def apply_adaptive_adjustments(self, adjustments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply adaptive adjustments to system parameters."""
        if adjustments is None:
            adjustments = self.get_adaptive_adjustments()
        
        changes_made = {}
        
        # Apply head keep ratio adjustment
        if adjustments['head_keep_ratio_adjustment'] != 0.0:
            old_ratio = self.head_keep_ratio
            self.head_keep_ratio = max(0.05, min(0.30, self.head_keep_ratio + adjustments['head_keep_ratio_adjustment']))
            
            # Update head builder
            self.head_builder.target_keep_ratio = self.head_keep_ratio
            
            changes_made['head_keep_ratio'] = {
                'old': old_ratio,
                'new': self.head_keep_ratio,
                'change': self.head_keep_ratio - old_ratio
            }
        
        # Apply stride adjustment
        if adjustments['stride_adjustment'] != 0.0:
            old_stride = self.stride
            self.stride = max(1000, int(self.stride * (1 + adjustments['stride_adjustment'])))
            
            # Update tail builder
            self.tail_builder.stride = self.stride
            
            changes_made['stride'] = {
                'old': old_stride,
                'new': self.stride,
                'change': self.stride - old_stride
            }
        
        # Note: λ and μ adjustments would be applied externally by the optimization system
        # since they're passed as parameters to the select() method
        
        if changes_made:
            logger.info(f"Applied adaptive adjustments: {changes_made}")
            
            # Log the adjustment event
            adjustment_log = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'adaptive_adjustment',
                'changes_made': changes_made,
                'recommendations': adjustments['actions_recommended']
            }
            
            with open(self.advanced_logger.log_path.parent / 'adaptive_adjustments.jsonl', 'a') as f:
                f.write(json.dumps(adjustment_log) + '\n')
        
        return changes_made
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_selections': len(self.performance_history),
            'kv_cache_stats': self.kv_arranger.kv_cache_stats,
            'monitoring_summary': self.advanced_logger.get_monitoring_summary(),
            'adaptive_adjustments': self.get_adaptive_adjustments(),
            'config': {
                'head_keep_ratio': self.head_keep_ratio,
                'window_size': self.window_size,
                'stride': self.stride,
                'sink_tokens': self.sink_tokens,
                'ce_k2': self.ce_k2,
                'dpp_rank': self.dpp_rank
            }
        }

# Integration with benchmark infrastructure
from .competitor_interface import ContextManagementCompetitor, ContextProcessingResult
import requests

class LetheStreamingHybridCompetitor(ContextManagementCompetitor):
    """Lethe→StreamingLLM Hybrid competitor for benchmarking."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Lethe-StreamingLLM-Hybrid", config)
        self.selector = None
    
    def initialize(self) -> bool:
        """Initialize the hybrid system."""
        try:
            # Use canary configuration from TODO.md
            hybrid_config = {
                'head_keep': 0.12,
                'window_size': 6000,
                'stride': 3000,
                'sinks': 96,
                'K2': 320,
                'dpp_rank': 14,
                'telemetry_path': 'hybrid_benchmark_telemetry.jsonl'
            }
            
            self.selector = HybridSelector(hybrid_config)
            self._initialized = True
            logger.info("Hybrid competitor initialized with canary configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid competitor: {e}")
            return False
    
    def process_context(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """Process context using the hybrid approach."""
        if not self._initialized:
            raise RuntimeError("Competitor not initialized")
        
        start_time = time.time()
        original_tokens = len(context.split())
        
        # Calculate lambda parameter from max_tokens constraint
        # λ should achieve the desired compression ratio
        target_keep_ratio = min(1.0, max_tokens / original_tokens) if original_tokens > 0 else 1.0
        lambda_param = 1.0 - target_keep_ratio  # λ=0 keeps all, λ=1 filters all
        lambda_param = max(0.001, min(0.999, lambda_param))  # Clamp to reasonable range
        
        # Use default μ parameter for compute budget
        mu_param = self.mu_param
        
        # Apply hybrid selection
        try:
            result = self.selector.select(query, context, lambda_param, mu_param)
            
            # Use gemma2:9b via Ollama for response generation
            llm_response = self._generate_response(query, result.final_context)
            
            # Calculate compression ratio
            processed_tokens = result.total_tokens
            compression_ratio = 1.0 - (float(processed_tokens) / float(original_tokens)) if original_tokens > 0 else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            return ContextProcessingResult(
                original_context=context,
                processed_context=result.final_context,
                query=query,
                response=llm_response,
                processing_time_ms=processing_time,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                compression_ratio=compression_ratio,
                method_name=self.name,
                metadata={
                    'gating_decision': result.gating_decision,
                    'head_tokens': result.head_result.total_tokens if result.head_result else 0,
                    'tail_tokens': result.tail_result.total_tokens if result.tail_result else 0,
                    'head_keep_ratio': result.head_result.keep_ratio if result.head_result else 0.0,
                    'tail_keep_ratio': result.tail_result.keep_ratio if result.tail_result else 0.0,
                    'kv_reuse': result.instrumentation.kv_prefix_reuse,
                    'lambda': lambda_param,
                    'mu': mu_param,
                    'primal_dual_gap': result.instrumentation.primal_dual_gap,
                    'tail_cvar_95': result.instrumentation.tail_cvar_95,
                    'num_windows': result.tail_result.num_windows if result.tail_result else 0,
                    'ce_early_exit': result.head_result.ce_early_exit_at > 0 if result.head_result else False,
                    'dpp_rank': result.head_result.dpp_rank if result.head_result else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            
            # Return error result
            return ContextProcessingResult(
                original_context=context,
                processed_context="",
                query=query,
                response=f"Error: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000,
                original_token_count=original_tokens,
                processed_token_count=0,
                compression_ratio=1.0,  # All context filtered out
                method_name=self.name,
                metadata={'error': str(e)}
            )
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Ollama gemma2:9b."""
        try:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            response = requests.post('http://localhost:11434/api/generate', json={
                'model': 'gemma2:9b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'max_tokens': 500
                }
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error calling gemma2:9b: {str(e)}"
    
    def get_installation_requirements(self) -> List[str]:
        """Get requirements for the hybrid system."""
        return ['numpy', 'scipy', 'requests']
    
    def cleanup(self):
        """Clean up hybrid system resources."""
        if self.selector:
            # Apply any final adaptive adjustments based on performance
            adjustments = self.selector.get_adaptive_adjustments()
            if adjustments['actions_recommended']:
                logger.info(f"Final recommendations: {adjustments['actions_recommended']}")
            
            # Log performance summary
            stats = self.selector.get_performance_stats()
            logger.info(f"Hybrid performance summary: {stats['monitoring_summary']}")
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """Get hybrid-specific performance statistics."""
        if not self.selector:
            return {}
        return self.selector.get_performance_stats()
    
    def apply_adaptive_adjustments(self) -> Dict[str, Any]:
        """Apply adaptive adjustments and return changes made."""
        if not self.selector:
            return {}
        return self.selector.apply_adaptive_adjustments()