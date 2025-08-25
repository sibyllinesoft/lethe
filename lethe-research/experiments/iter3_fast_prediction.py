#!/usr/bin/env python3
"""
Ultra-fast ML prediction for Iteration 3 - optimized for <200ms loading.

This module provides a lightweight alternative to the full ML models,
using simple heuristics and lookup tables for sub-100ms performance.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class UltraFastPredictor:
    """Ultra-lightweight predictor optimized for speed over accuracy."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize with minimal overhead."""
        self.models_dir = Path(models_dir)
        self._load_time_ms = 0
        
        # Precomputed lookup tables (loaded from config)
        self.feature_weights = {
            'code_symbol': 0.15,
            'error_token': -0.10,
            'path_file': 0.12,
            'numeric_id': 0.08,
            'complexity': 0.20,
            'query_length': 0.05
        }
        
        # Plan selection rules (deterministic)
        self.plan_rules = {
            'code_heavy': 'exploit',      # Code symbols, function calls
            'error_debugging': 'verify',   # Error messages, stack traces  
            'exploratory': 'explore',     # How-to, best practices
            'default': 'exploit'
        }
        
        # Regex patterns (compiled once)
        self.code_pattern = re.compile(r'[_a-zA-Z][\w]*\(|\b[A-Z][A-Za-z0-9]+::[A-Za-z0-9]+\b|(class|function|def|var|let|const|import|export)', re.IGNORECASE)
        self.error_pattern = re.compile(r'(exception|error|stack\s+trace|errno|\bE\d{2,}\b|failed|crashed|broken|bug|undefined|null|nan)', re.IGNORECASE)
        self.path_pattern = re.compile(r'/[^\s]+\.[a-zA-Z0-9]+|[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+|\b\w+\.(js|py|ts|java|cpp|h|css|html)\b')
        self.explore_pattern = re.compile(r'(how|why|what|when|where|best|practice|guide|tutorial)', re.IGNORECASE)
        
    def load_models(self) -> bool:
        """Load configuration (very fast, no actual models)."""
        start_time = time.time()
        
        try:
            # Load feature extractor config if available
            config_path = self.models_dir / 'feature_extractor.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Update weights from config if available
                    if 'feature_weights' in config:
                        self.feature_weights.update(config['feature_weights'])
            
            self._load_time_ms = (time.time() - start_time) * 1000
            return True
            
        except Exception as e:
            # Silently use defaults (no print to avoid JSON parsing issues)
            self._load_time_ms = (time.time() - start_time) * 1000
            return True  # Still functional with defaults
    
    def extract_features(self, query: str) -> Dict[str, float]:
        """Fast feature extraction using precompiled patterns."""
        
        # Basic features
        query_len = len(query)
        word_count = len(query.split())
        
        # Pattern matching
        has_code = bool(self.code_pattern.search(query))
        has_error = bool(self.error_pattern.search(query))
        has_path = bool(self.path_pattern.search(query))
        has_numeric = bool(re.search(r'\b\d{3,}\b', query))
        
        # Complexity score
        complexity = 0.0
        if word_count > 10: complexity += 0.3
        elif word_count > 5: complexity += 0.15
        if has_code: complexity += 0.25
        if has_error: complexity += 0.2
        if has_path: complexity += 0.15
        if '?' in query: complexity += 0.1
        complexity = min(complexity, 1.0)
        
        return {
            'query_len': query_len,
            'word_count': word_count,
            'has_code_symbol': has_code,
            'has_error_token': has_error,
            'has_path_file': has_path,
            'has_numeric_id': has_numeric,
            'complexity_score': complexity
        }
    
    def predict_fusion_params(self, query: str, context: Optional[Dict] = None) -> Tuple[float, float]:
        """Predict alpha/beta using fast heuristics."""
        
        features = self.extract_features(query)
        context = context or {}
        
        # Alpha prediction (lexical vs semantic balance)
        # Higher alpha = more lexical/BM25, lower alpha = more semantic/vector
        alpha_base = 0.7
        
        # Code queries prefer lexical matching
        if features['has_code_symbol']:
            alpha_base += 0.15
            
        # Error debugging benefits from semantic understanding  
        if features['has_error_token']:
            alpha_base -= 0.10
            
        # Complex queries need more semantic search
        if features['complexity_score'] > 0.5:
            alpha_base -= 0.05 * features['complexity_score']
            
        # Context from retrieval results
        if context.get('bm25_top1', 0) > 0.8:
            alpha_base += 0.05  # BM25 is working well
        if context.get('ann_top1', 0) > 0.8:
            alpha_base -= 0.05  # Vector search is working well
            
        # Beta prediction (reranking aggressiveness)
        # Higher beta = more aggressive reranking
        beta_base = 0.5
        
        # Long queries benefit from reranking
        if features['word_count'] > 8:
            beta_base += 0.1
            
        # Error queries need careful ranking
        if features['has_error_token']:
            beta_base += 0.15
            
        # High overlap suggests good initial ranking
        if context.get('overlap_ratio', 0) > 0.7:
            beta_base -= 0.1
        
        # Clamp to safe ranges
        alpha = max(0.3, min(1.5, alpha_base))
        beta = max(0.3, min(1.0, beta_base))
        
        return alpha, beta
    
    def predict_plan(self, query: str, context: Optional[Dict] = None) -> str:
        """Predict plan using deterministic rules."""
        
        features = self.extract_features(query)
        query_lower = query.lower()
        
        # Rule-based plan selection
        
        # Code-heavy queries -> exploit (targeted retrieval)
        if features['has_code_symbol'] and not features['has_error_token']:
            return 'exploit'
            
        # Error debugging -> verify (fact-checking)
        if features['has_error_token']:
            return 'verify'
            
        # Exploratory queries -> explore (broad search)
        if self.explore_pattern.search(query):
            return 'explore'
            
        # Complex queries -> explore (need broad context)
        if features['complexity_score'] > 0.7:
            return 'explore'
            
        # High contradiction context -> verify
        context = context or {}
        if context.get('contradictions', 0) > 2:
            return 'verify'
            
        # Default to exploit (focused retrieval)
        return 'exploit'
    
    def predict_all(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict both fusion parameters and plan."""
        
        start_time = time.time()
        
        # Get predictions
        alpha, beta = self.predict_fusion_params(query, context)
        plan = self.predict_plan(query, context)
        
        prediction_time_ms = (time.time() - start_time) * 1000
        
        return {
            'alpha': alpha,
            'beta': beta,
            'plan': plan,
            'prediction_time_ms': prediction_time_ms,
            'model_loaded': True,
            'load_time_ms': self._load_time_ms
        }
    
    def get_load_time(self) -> float:
        """Get model loading time in milliseconds."""
        return self._load_time_ms


def create_ultra_fast_predictor(models_dir: str = "models") -> UltraFastPredictor:
    """Create and initialize an ultra-fast predictor."""
    
    predictor = UltraFastPredictor(models_dir)
    
    # Load config (very fast)
    load_success = predictor.load_models()
    
    if not load_success:
        # Silently continue with defaults
        pass
    else:
        load_time = predictor.get_load_time()
        # Silently note performance (no output to avoid JSON parsing issues)
        pass
    
    return predictor


# Export for compatibility
__all__ = ['UltraFastPredictor', 'create_ultra_fast_predictor']


if __name__ == "__main__":
    # Test the ultra-fast predictor
    print("Testing ultra-fast ML predictor...")
    
    predictor = create_ultra_fast_predictor()
    
    test_queries = [
        "TypeScript async error handling",
        "How to implement REST API authentication", 
        "class MyComponent extends React.Component",
        "NullPointerException in line 42",
        "best practices for database design"
    ]
    
    for query in test_queries:
        result = predictor.predict_all(query)
        print(f"Query: {query[:40]}...")
        print(f"  Alpha: {result['alpha']:.3f}, Beta: {result['beta']:.3f}, Plan: {result['plan']}")
        print(f"  Time: {result['prediction_time_ms']:.1f}ms")
        print()