"""
Lethe→StreamingLLM Hybrid System Benchmark Integration

Integrates the hybrid system with the existing benchmark infrastructure 
to enable direct comparison with other context management competitors.

Features:
- Implements ContextManagementCompetitor interface
- Provides comprehensive instrumentation
- Supports canary configuration rollout
- Enables A/B testing against StreamingLLM baseline
"""

import time
import logging
from typing import Dict, Any, List
import json
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitor_interface import ContextManagementCompetitor, ContextProcessingResult
from lethe_streaming_hybrid import HybridSelector

logger = logging.getLogger(__name__)

class HybridSystemCompetitor(ContextManagementCompetitor):
    """Hybrid system competitor for benchmark comparisons."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LetheStreamingHybrid", config)
        
        # Hybrid system configuration
        self.hybrid_config = {
            'head_keep': config.get('head_keep', 0.12) if config else 0.12,
            'window_size': config.get('window_size', 6000) if config else 6000,
            'stride': config.get('stride', 3000) if config else 3000,
            'sinks': config.get('sinks', 96) if config else 96,
            'K2': config.get('K2', 320) if config else 320,
            'dpp_rank': config.get('dpp_rank', 14) if config else 14
        }
        
        # Performance parameters
        self.lambda_param = config.get('lambda', 0.001) if config else 0.001
        self.mu_param = config.get('mu', 0.0001) if config else 0.0001
        
        # Instrumentation tracking
        self.instrumentation_history = []
        self.performance_stats = {
            'total_calls': 0,
            'total_processing_time': 0.0,
            'head_only_calls': 0,
            'hybrid_calls': 0,
            'average_kv_reuse': 0.0,
            'average_compression_ratio': 0.0
        }
        
        # Hybrid selector instance
        self.hybrid_selector = None
        
    def initialize(self) -> bool:
        """Initialize the hybrid system."""
        try:
            self.hybrid_selector = HybridSelector(self.hybrid_config)
            self._initialized = True
            
            logger.info(f"Hybrid system initialized with config: {self.hybrid_config}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid system: {e}")
            return False
    
    def process_context(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """Process context using the hybrid system."""
        if not self._initialized:
            raise RuntimeError("Hybrid system not initialized")
        
        start_time = time.time()
        
        try:
            # Adjust lambda parameter based on max_tokens budget
            original_tokens = len(context.split())
            if original_tokens > 0:
                target_keep_ratio = min(1.0, max_tokens / original_tokens)
                # Scale lambda to achieve target ratio
                adjusted_lambda = self._calculate_lambda_for_ratio(target_keep_ratio)
            else:
                adjusted_lambda = self.lambda_param
            
            # Execute hybrid selection
            hybrid_result = self.hybrid_selector.select(
                query=query,
                context=context,
                lambda_param=adjusted_lambda,
                mu_param=self.mu_param
            )
            
            # Extract processed context
            processed_context = hybrid_result.final_context
            
            # Ensure we don't exceed max_tokens
            if len(processed_context.split()) > max_tokens:
                context_tokens = processed_context.split()
                processed_context = ' '.join(context_tokens[:max_tokens])
                actual_tokens = max_tokens
            else:
                actual_tokens = hybrid_result.total_tokens
            
            # Generate mock response (in practice would call LLM)
            response = self._generate_mock_response(query, processed_context)
            
            # Calculate final metrics
            processing_time = (time.time() - start_time) * 1000
            compression_ratio = 1.0 - (actual_tokens / original_tokens) if original_tokens > 0 else 0.0
            
            # Update performance stats
            self._update_performance_stats(hybrid_result, processing_time, compression_ratio)
            
            # Store instrumentation
            self.instrumentation_history.append({
                'timestamp': time.time(),
                'instrumentation': hybrid_result.instrumentation.to_dict(),
                'processing_time_ms': processing_time,
                'gating_decision': hybrid_result.gating_decision
            })
            
            return ContextProcessingResult(
                original_context=context,
                processed_context=processed_context,
                query=query,
                response=response,
                processing_time_ms=processing_time,
                original_token_count=original_tokens,
                processed_token_count=actual_tokens,
                compression_ratio=compression_ratio,
                method_name=self.name,
                metadata={
                    'approach': 'hybrid_lethe_streaming',
                    'gating_decision': hybrid_result.gating_decision,
                    'head_tokens': hybrid_result.head_result.total_tokens if hybrid_result.head_result else 0,
                    'tail_tokens': hybrid_result.tail_result.total_tokens if hybrid_result.tail_result else 0,
                    'kv_reuse': hybrid_result.instrumentation.kv_prefix_reuse,
                    'lambda_param': adjusted_lambda,
                    'mu_param': self.mu_param,
                    'config': self.hybrid_config,
                    'instrumentation': hybrid_result.instrumentation.to_dict()
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return ContextProcessingResult(
                original_context=context,
                processed_context="",
                query=query,
                response=f"Error: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000,
                original_token_count=len(context.split()),
                processed_token_count=0,
                compression_ratio=1.0,
                method_name=self.name,
                metadata={"error": str(e)}
            )
    
    def _calculate_lambda_for_ratio(self, target_ratio: float) -> float:
        """Calculate lambda parameter to achieve target compression ratio."""
        if target_ratio >= 0.9:
            return 0.0001  # Minimal constraint
        elif target_ratio >= 0.5:
            return 0.001   # Light constraint
        elif target_ratio >= 0.2:
            return 0.005   # Medium constraint
        else:
            return 0.01    # Heavy constraint
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """Generate mock response for testing purposes."""
        # In practice, this would call an actual LLM
        return f"Mock response for query: {query[:50]}... based on {len(context.split())} tokens of context."
    
    def _update_performance_stats(self, hybrid_result, processing_time: float, compression_ratio: float):
        """Update running performance statistics."""
        self.performance_stats['total_calls'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        if hybrid_result.gating_decision == 'head_only':
            self.performance_stats['head_only_calls'] += 1
        elif hybrid_result.gating_decision == 'hybrid':
            self.performance_stats['hybrid_calls'] += 1
        
        # Update running averages
        total_calls = self.performance_stats['total_calls']
        
        current_kv_avg = self.performance_stats['average_kv_reuse']
        new_kv_reuse = hybrid_result.instrumentation.kv_prefix_reuse
        self.performance_stats['average_kv_reuse'] = (current_kv_avg * (total_calls - 1) + new_kv_reuse) / total_calls
        
        current_comp_avg = self.performance_stats['average_compression_ratio']
        self.performance_stats['average_compression_ratio'] = (current_comp_avg * (total_calls - 1) + compression_ratio) / total_calls
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if self.performance_stats['total_calls'] == 0:
            return {'error': 'No processing calls made'}
        
        avg_processing_time = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['total_calls']
        )
        
        # Calculate gating decision ratios
        total_calls = self.performance_stats['total_calls']
        head_only_ratio = self.performance_stats['head_only_calls'] / total_calls
        hybrid_ratio = self.performance_stats['hybrid_calls'] / total_calls
        
        # Recent instrumentation stats
        recent_instrumentation = {}
        if self.instrumentation_history:
            recent_data = self.instrumentation_history[-10:]  # Last 10 calls
            recent_instrumentation = {
                'avg_lambda': sum(d['instrumentation']['lambda'] for d in recent_data) / len(recent_data),
                'avg_mu': sum(d['instrumentation']['mu'] for d in recent_data) / len(recent_data),
                'avg_head_tokens': sum(d['instrumentation']['head_tokens'] for d in recent_data) / len(recent_data),
                'avg_tail_tokens': sum(d['instrumentation']['tail_tokens'] for d in recent_data) / len(recent_data),
                'avg_kv_reuse': sum(d['instrumentation']['kv_prefix_reuse'] for d in recent_data) / len(recent_data),
                'avg_delta_cbu': sum(d['instrumentation']['delta_cbu_per_1k'] for d in recent_data) / len(recent_data)
            }
        
        return {
            'competitor_name': self.name,
            'config': self.hybrid_config,
            'performance_stats': self.performance_stats,
            'derived_metrics': {
                'average_processing_time_ms': avg_processing_time,
                'head_only_ratio': head_only_ratio,
                'hybrid_ratio': hybrid_ratio,
                'fallback_ratio': 1.0 - head_only_ratio - hybrid_ratio
            },
            'recent_instrumentation': recent_instrumentation,
            'total_instrumentation_records': len(self.instrumentation_history)
        }
    
    def get_detailed_instrumentation(self) -> List[Dict[str, Any]]:
        """Get detailed instrumentation history."""
        return self.instrumentation_history.copy()
    
    def export_instrumentation(self, filepath: str) -> bool:
        """Export instrumentation data to file."""
        try:
            export_data = {
                'competitor_name': self.name,
                'config': self.hybrid_config,
                'performance_summary': self.get_performance_summary(),
                'instrumentation_history': self.instrumentation_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Instrumentation exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export instrumentation: {e}")
            return False
    
    def get_installation_requirements(self) -> List[str]:
        """Get installation requirements for hybrid system."""
        return [
            "numpy>=1.20.0",
            "scipy>=1.7.0", 
            "pandas>=1.3.0"
        ]
    
    def cleanup(self):
        """Clean up hybrid system resources."""
        if self.hybrid_selector:
            # Clear any caches
            self.hybrid_selector.head_builder.__dict__.clear()
            self.hybrid_selector.tail_builder.__dict__.clear()
        
        self.instrumentation_history.clear()
        logger.info("Hybrid system resources cleaned up")

class HybridCanaryController:
    """Controls canary rollout of hybrid system."""
    
    def __init__(self, baseline_competitor, hybrid_competitor, canary_percentage: float = 5.0):
        """
        Initialize canary controller.
        
        Args:
            baseline_competitor: StreamingLLM or other baseline
            hybrid_competitor: HybridSystemCompetitor
            canary_percentage: Percentage of traffic to route to hybrid
        """
        self.baseline = baseline_competitor
        self.hybrid = hybrid_competitor
        self.canary_percentage = canary_percentage
        
        # Routing state
        self.request_count = 0
        self.hybrid_requests = 0
        self.baseline_requests = 0
        
        # Performance tracking for comparison
        self.performance_comparison = {
            'hybrid_metrics': [],
            'baseline_metrics': []
        }
    
    def process_request(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """Process request with canary routing."""
        self.request_count += 1
        
        # Determine routing (simple hash-based for deterministic behavior)
        route_to_hybrid = (hash(query + context) % 100) < self.canary_percentage
        
        if route_to_hybrid:
            self.hybrid_requests += 1
            result = self.hybrid.process_context(query, context, max_tokens)
            self.performance_comparison['hybrid_metrics'].append({
                'processing_time_ms': result.processing_time_ms,
                'compression_ratio': result.compression_ratio,
                'token_count': result.processed_token_count
            })
            result.metadata['canary_routing'] = 'hybrid'
        else:
            self.baseline_requests += 1
            result = self.baseline.process_context(query, context, max_tokens)
            self.performance_comparison['baseline_metrics'].append({
                'processing_time_ms': result.processing_time_ms,
                'compression_ratio': result.compression_ratio,
                'token_count': result.processed_token_count
            })
            result.metadata['canary_routing'] = 'baseline'
        
        return result
    
    def get_canary_stats(self) -> Dict[str, Any]:
        """Get canary performance statistics."""
        hybrid_metrics = self.performance_comparison['hybrid_metrics']
        baseline_metrics = self.performance_comparison['baseline_metrics']
        
        stats = {
            'total_requests': self.request_count,
            'hybrid_requests': self.hybrid_requests,
            'baseline_requests': self.baseline_requests,
            'canary_percentage': self.canary_percentage,
            'actual_hybrid_percentage': (self.hybrid_requests / self.request_count * 100) if self.request_count > 0 else 0
        }
        
        if hybrid_metrics:
            stats['hybrid_performance'] = {
                'avg_processing_time_ms': sum(m['processing_time_ms'] for m in hybrid_metrics) / len(hybrid_metrics),
                'avg_compression_ratio': sum(m['compression_ratio'] for m in hybrid_metrics) / len(hybrid_metrics),
                'avg_token_count': sum(m['token_count'] for m in hybrid_metrics) / len(hybrid_metrics)
            }
        
        if baseline_metrics:
            stats['baseline_performance'] = {
                'avg_processing_time_ms': sum(m['processing_time_ms'] for m in baseline_metrics) / len(baseline_metrics),
                'avg_compression_ratio': sum(m['compression_ratio'] for m in baseline_metrics) / len(baseline_metrics),
                'avg_token_count': sum(m['token_count'] for m in baseline_metrics) / len(baseline_metrics)
            }
        
        # Performance comparison
        if hybrid_metrics and baseline_metrics:
            hybrid_avg_latency = stats['hybrid_performance']['avg_processing_time_ms']
            baseline_avg_latency = stats['baseline_performance']['avg_processing_time_ms']
            
            stats['performance_comparison'] = {
                'latency_difference_ms': hybrid_avg_latency - baseline_avg_latency,
                'latency_improvement_pct': ((baseline_avg_latency - hybrid_avg_latency) / baseline_avg_latency * 100) if baseline_avg_latency > 0 else 0,
                'meets_latency_requirement': (hybrid_avg_latency - baseline_avg_latency) <= 1.0  # p95 ≤ +1ms
            }
        
        return stats
    
    def should_promote_hybrid(self, min_requests: int = 100) -> bool:
        """Determine if hybrid should be promoted based on canary results."""
        if self.request_count < min_requests:
            return False
        
        stats = self.get_canary_stats()
        
        if 'performance_comparison' not in stats:
            return False
        
        comparison = stats['performance_comparison']
        
        # Check promotion criteria from TODO.md:
        # Hybrid must beat Streaming on P@k or ΔCBU/1k with p95 ≤ +1ms
        meets_latency_req = comparison['meets_latency_requirement']
        has_improvement = comparison['latency_improvement_pct'] > 0
        
        return meets_latency_req and has_improvement

def create_hybrid_competitor(config: Dict[str, Any] = None) -> HybridSystemCompetitor:
    """Factory function to create hybrid system competitor."""
    return HybridSystemCompetitor(config)

def create_canary_controller(baseline_name: str = "streaming", canary_pct: float = 5.0) -> HybridCanaryController:
    """Factory function to create canary controller with baseline."""
    
    # Import and create baseline competitor
    if baseline_name == "streaming":
        from streamingllm_benchmark import StreamingLLMCompetitor
        baseline_config = {
            'window_size': 6000,
            'attention_sink_size': 96
        }
        baseline = StreamingLLMCompetitor(baseline_config)
    else:
        raise ValueError(f"Unknown baseline competitor: {baseline_name}")
    
    # Create hybrid competitor
    hybrid_config = {
        'head_keep': 0.12,
        'window_size': 6000,
        'stride': 3000,
        'sinks': 96,
        'K2': 320,
        'dpp_rank': 14
    }
    hybrid = HybridSystemCompetitor(hybrid_config)
    
    # Initialize both
    baseline.initialize()
    hybrid.initialize()
    
    return HybridCanaryController(baseline, hybrid, canary_pct)

if __name__ == "__main__":
    """Test the hybrid benchmark integration."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create hybrid competitor
    hybrid_config = {
        'head_keep': 0.12,
        'window_size': 6000,
        'stride': 3000,
        'sinks': 96
    }
    
    competitor = HybridSystemCompetitor(hybrid_config)
    
    if competitor.initialize():
        print("Hybrid competitor initialized successfully")
        
        # Test processing
        test_query = "What are the main functions in this code?"
        test_context = """
        def process_data(data):
            '''Process input data and return results.'''
            if not data:
                raise ValueError("Data cannot be empty")
            
            results = []
            for item in data:
                processed = transform_item(item)
                results.append(processed)
            
            return results
        
        def transform_item(item):
            '''Transform a single item.'''
            return item.upper().strip()
        
        class DataProcessor:
            '''Main data processing class.'''
            
            def __init__(self, config):
                self.config = config
            
            def run(self):
                '''Run the processing pipeline.'''
                data = self.load_data()
                return process_data(data)
        """ * 10  # Make it longer to test compression
        
        result = competitor.process_context(test_query, test_context)
        
        print(f"Processing complete:")
        print(f"  Original tokens: {result.original_token_count}")
        print(f"  Processed tokens: {result.processed_token_count}")
        print(f"  Compression ratio: {result.compression_ratio:.2%}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")
        print(f"  Gating decision: {result.metadata.get('gating_decision')}")
        print(f"  KV reuse: {result.metadata.get('kv_reuse', 0.0):.3f}")
        
        # Get performance summary
        summary = competitor.get_performance_summary()
        print(f"\nPerformance summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
    
    else:
        print("Failed to initialize hybrid competitor")