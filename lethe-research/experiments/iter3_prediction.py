#!/usr/bin/env python3
"""
Iteration 3: Prediction utilities for fast model inference

This module provides lightweight, fast-loading prediction utilities
for production use in the TypeScript retrieval system.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)


class FastQueryPredictor:
    """Fast, lightweight predictor for production use."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize with model directory path."""
        self.models_dir = Path(models_dir)
        
        # Model components
        self.fusion_alpha_model = None
        self.fusion_beta_model = None
        self.fusion_scaler = None
        
        self.plan_model = None
        self.plan_scaler = None
        self.plan_classes = ['explore', 'verify', 'exploit']
        
        # Feature extraction patterns (loaded from config)
        self.code_patterns = []
        self.error_patterns = []
        self.file_patterns = []
        
        self._models_loaded = False
        self._load_time_ms = 0
    
    def load_models(self) -> bool:
        """Load models with timing for performance validation."""
        
        if self._models_loaded:
            return True
            
        start_time = time.time()
        
        try:
            # Load fusion model
            fusion_path = self.models_dir / 'dynamic_fusion_model.joblib'
            if fusion_path.exists():
                fusion_data = joblib.load(fusion_path)
                self.fusion_alpha_model = fusion_data['alpha_model']
                self.fusion_beta_model = fusion_data['beta_model'] 
                self.fusion_scaler = fusion_data['scaler']
                logger.debug("Fusion model loaded")
            else:
                logger.warning(f"Fusion model not found at {fusion_path}")
                return False
            
            # Load plan selector
            plan_path = self.models_dir / 'learned_plan_selector.joblib'
            if plan_path.exists():
                plan_data = joblib.load(plan_path)
                self.plan_model = plan_data['model']
                self.plan_scaler = plan_data['scaler']
                self.plan_classes = plan_data['plan_classes']
                logger.debug("Plan selector loaded")
            else:
                logger.warning(f"Plan selector not found at {plan_path}")
                return False
            
            # Load feature extractor patterns
            feature_path = self.models_dir / 'feature_extractor.json'
            if feature_path.exists():
                with open(feature_path) as f:
                    config = json.load(f)
                    self.code_patterns = config['code_patterns']
                    self.error_patterns = config['error_patterns']
                    self.file_patterns = config['file_patterns']
                logger.debug("Feature patterns loaded")
            else:
                logger.warning(f"Feature config not found at {feature_path}")
                # Use default patterns
                self._setup_default_patterns()
            
            self._load_time_ms = (time.time() - start_time) * 1000
            self._models_loaded = True
            
            logger.info(f"Models loaded successfully in {self._load_time_ms:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self._models_loaded = False
            return False
    
    def _setup_default_patterns(self):
        """Setup default feature extraction patterns."""
        
        self.code_patterns = [
            r'[_a-zA-Z][\w]*\(',
            r'\b[A-Z][A-Za-z0-9]+::[A-Za-z0-9]+\b',
            r'\b(class|function|def|var|let|const|import|export)\b',
            r'\{[^}]*\}',
            r'\[[^\]]*\]',
        ]
        
        self.error_patterns = [
            r'(Exception|Error|stack trace|errno)',
            r'\bE\d{2,}\b',
            r'(failed|crashed|broken|bug)',
            r'(undefined|null|NaN)',
        ]
        
        self.file_patterns = [
            r'\/[^\s]+\.[a-zA-Z0-9]+',
            r'[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+',
            r'\b\w+\.(js|py|ts|java|cpp|h|css|html)\b',
        ]
    
    def predict_fusion_params(self, query: str, context: Optional[Dict] = None) -> Tuple[float, float]:
        """Predict optimal alpha and beta parameters.
        
        Args:
            query: Query string
            context: Optional context with retrieval metadata
            
        Returns:
            Tuple of (alpha, beta) values clamped to [0.3, 1.5]
        """
        
        if not self._models_loaded:
            if not self.load_models():
                # Fallback to static parameters
                logger.warning("Models not loaded, using static fallback")
                return 0.7, 0.5
        
        try:
            # Extract features efficiently
            features = self._extract_features_fast(query, context)
            
            # Convert to array and normalize
            X = np.array(features).reshape(1, -1)
            X_scaled = self.fusion_scaler.transform(X)
            
            # Predict
            alpha_pred = float(self.fusion_alpha_model.predict(X_scaled)[0])
            beta_pred = float(self.fusion_beta_model.predict(X_scaled)[0])
            
            # Clamp to safe ranges [0.3, 1.5]
            alpha = np.clip(alpha_pred, 0.3, 1.5)
            beta = np.clip(beta_pred, 0.3, 1.5)
            
            return alpha, beta
            
        except Exception as e:
            logger.warning(f"Fusion prediction failed: {e}, using fallback")
            return 0.7, 0.5  # Safe fallback
    
    def predict_plan(self, query: str, context: Optional[Dict] = None) -> str:
        """Predict optimal plan selection.
        
        Args:
            query: Query string  
            context: Optional context with session state
            
        Returns:
            Plan name: 'explore', 'verify', or 'exploit'
        """
        
        if not self._models_loaded:
            if not self.load_models():
                # Fallback to heuristic
                logger.warning("Models not loaded, using heuristic fallback")
                return self._heuristic_plan_fallback(query, context)
        
        try:
            # Extract features
            features = self._extract_features_fast(query, context)
            
            # Convert and normalize
            X = np.array(features).reshape(1, -1)
            X_scaled = self.plan_scaler.transform(X)
            
            # Predict
            prediction_idx = self.plan_model.predict(X_scaled)[0]
            
            return self.plan_classes[prediction_idx]
            
        except Exception as e:
            logger.warning(f"Plan prediction failed: {e}, using fallback")
            return self._heuristic_plan_fallback(query, context)
    
    def predict_all(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict both fusion parameters and plan in one call."""
        
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
            'model_loaded': self._models_loaded
        }
    
    def _extract_features_fast(self, query: str, context: Optional[Dict] = None) -> list:
        """Fast feature extraction optimized for production."""
        
        query_lower = query.lower()
        
        # Basic features
        query_len = len(query)
        word_count = len(query.split())
        char_count = len(query)
        
        # Pattern matching (optimized)
        has_code_symbol = self._fast_pattern_check(query, self.code_patterns)
        has_error_token = self._fast_pattern_check(query_lower, self.error_patterns)  
        has_path_or_file = self._fast_pattern_check(query, self.file_patterns)
        has_numeric_id = bool(self._simple_regex_search(r'\b\d{3,}\b', query))
        
        # Complexity score (simplified)
        complexity_score = 0.0
        if word_count > 10:
            complexity_score += 0.3
        elif word_count > 5:
            complexity_score += 0.15
        if has_code_symbol:
            complexity_score += 0.25
        if has_error_token:
            complexity_score += 0.2
        if has_path_or_file:
            complexity_score += 0.15
        complexity_score = min(complexity_score, 1.0)
        
        # Context features
        bm25_top1 = context.get('bm25_top1', 0.5) if context else 0.5
        ann_top1 = context.get('ann_top1', 0.5) if context else 0.5
        overlap_ratio = context.get('overlap_ratio', 0.1) if context else 0.1
        hyde_k = context.get('hyde_k', 3) if context else 3
        
        return [
            query_len,
            float(has_code_symbol),
            float(has_error_token),
            float(has_path_or_file),
            float(has_numeric_id),
            bm25_top1,
            ann_top1,
            overlap_ratio,
            hyde_k,
            word_count,
            char_count,
            complexity_score
        ]
    
    def _fast_pattern_check(self, text: str, patterns: list) -> bool:
        """Fast pattern checking without full regex compilation."""
        
        # Use simple string operations for performance
        for pattern in patterns:
            if self._simple_pattern_match(pattern, text):
                return True
        return False
    
    def _simple_pattern_match(self, pattern: str, text: str) -> bool:
        """Simplified pattern matching for common cases."""
        
        # Convert regex patterns to simple string checks for speed
        import re
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False
    
    def _simple_regex_search(self, pattern: str, text: str) -> bool:
        """Simple regex search with caching."""
        
        import re
        try:
            return bool(re.search(pattern, text))
        except re.error:
            return False
    
    def _heuristic_plan_fallback(self, query: str, context: Optional[Dict] = None) -> str:
        """Heuristic fallback when ML model unavailable."""
        
        query_lower = query.lower()
        
        # Check for error-related terms
        error_terms = ['error', 'exception', 'failed', 'broken', 'bug', 'crash']
        if any(term in query_lower for term in error_terms):
            return 'verify'
        
        # Check for exploration indicators
        exploration_terms = ['how', 'what', 'why', 'best practices', 'tutorial', 'guide']
        if any(term in query_lower for term in exploration_terms):
            return 'explore'
        
        # Check context for contradictions
        if context and context.get('contradictions', 0) > 0:
            return 'verify'
        
        # Default to exploit
        return 'exploit'
    
    def get_load_time(self) -> float:
        """Get model loading time in milliseconds."""
        return self._load_time_ms
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded


class Iteration3Config:
    """Configuration for Iteration 3 dynamic features."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
    def is_fusion_dynamic(self) -> bool:
        """Check if dynamic fusion is enabled."""
        return self.config.get('fusion', {}).get('dynamic', False)
    
    def is_plan_learned(self) -> bool:
        """Check if learned planning is enabled.""" 
        return self.config.get('plan', {}).get('learned', False)
    
    def get_models_dir(self) -> str:
        """Get models directory path."""
        return self.config.get('models_dir', 'models')
    
    def get_fallback_alpha(self) -> float:
        """Get fallback alpha value."""
        return self.config.get('fallback', {}).get('alpha', 0.7)
    
    def get_fallback_beta(self) -> float:
        """Get fallback beta value."""
        return self.config.get('fallback', {}).get('beta', 0.5)
    
    def get_fallback_plan(self) -> str:
        """Get fallback plan."""
        return self.config.get('fallback', {}).get('plan', 'exploit')


# Global predictor instance for reuse across subprocess calls
_global_predictor = None

def create_production_predictor(models_dir: str = "models") -> FastQueryPredictor:
    """Create or reuse a production-ready predictor instance."""
    global _global_predictor
    
    if _global_predictor is None or not _global_predictor._models_loaded:
        _global_predictor = FastQueryPredictor(models_dir)
        # Pre-load models
        load_success = _global_predictor.load_models()
        
        if not load_success:
            logger.warning("Model loading failed, predictor will use fallbacks")
        else:
            load_time = _global_predictor.get_load_time()
            if load_time > 200:
                logger.warning(f"Model loading took {load_time:.1f}ms (>200ms budget)")
            else:
                logger.info(f"Models loaded in {load_time:.1f}ms (within budget)")
    
    return _global_predictor


# Export for easy import
__all__ = [
    'FastQueryPredictor',
    'Iteration3Config', 
    'create_production_predictor'
]