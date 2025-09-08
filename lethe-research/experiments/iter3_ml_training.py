#!/usr/bin/env python3
"""
Iteration 3: Dynamic Fusion & Learned Planning - ML Training Pipeline

This module implements the machine learning training pipeline for predicting
optimal alpha/beta fusion parameters and plan selection based on query characteristics.

Key components:
1. Query feature extraction
2. Dynamic alpha/beta regression models (LightGBM/Ridge)
3. Learned plan selection classifier (Logistic Regression)
4. Model validation and serialization
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)

@dataclass
class QueryFeatures:
    """Query feature extraction results."""
    query_len: int
    has_code_symbol: bool
    has_error_token: bool
    has_path_or_file: bool
    has_numeric_id: bool
    bm25_top1: float
    ann_top1: float
    overlap_ratio: float
    hyde_k: int
    word_count: int
    char_count: int
    complexity_score: float


@dataclass
class TrainingExample:
    """Single training example for ML models."""
    features: QueryFeatures
    optimal_alpha: float
    optimal_beta: float
    best_plan: str
    ndcg_score: float
    query_text: str


class QueryFeatureExtractor:
    """Extract features from queries for ML prediction."""
    
    def __init__(self):
        self.code_patterns = [
            r'[_a-zA-Z][\w]*\(',  # Function calls
            r'\b[A-Z][A-Za-z0-9]+::[A-Za-z0-9]+\b',  # Namespaced identifiers
            r'\b(class|function|def|var|let|const|import|export)\b',  # Keywords
            r'\{[^}]*\}',  # Code blocks
            r'\[[^\]]*\]',  # Array syntax
        ]
        
        self.error_patterns = [
            r'(Exception|Error|stack trace|errno)',
            r'\bE\d{2,}\b',  # Error codes
            r'(failed|crashed|broken|bug)',
            r'(undefined|null|NaN)',
        ]
        
        self.file_patterns = [
            r'\/[^\s]+\.[a-zA-Z0-9]+',  # Unix paths
            r'[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+',  # Windows paths
            r'\b\w+\.(js|py|ts|java|cpp|h|css|html)\b',  # File extensions
        ]
        
    def extract_features(self, query: str, context: Optional[Dict] = None) -> QueryFeatures:
        """Extract comprehensive features from a query string."""
        
        query_lower = query.lower()
        
        # Basic text features
        query_len = len(query)
        word_count = len(query.split())
        char_count = len(query)
        
        # Pattern-based features
        has_code_symbol = any(
            self._regex_search(pattern, query) 
            for pattern in self.code_patterns
        )
        
        has_error_token = any(
            self._regex_search(pattern, query_lower) 
            for pattern in self.error_patterns
        )
        
        has_path_or_file = any(
            self._regex_search(pattern, query) 
            for pattern in self.file_patterns
        )
        
        has_numeric_id = bool(self._regex_search(r'\b\d{3,}\b', query))
        
        # Complexity score based on multiple factors
        complexity_score = self._calculate_complexity_score(query, {
            'has_code': has_code_symbol,
            'has_error': has_error_token,
            'has_file': has_path_or_file,
            'word_count': word_count
        })
        
        # Context-dependent features (from retrieval results if available)
        bm25_top1 = context.get('bm25_top1', 0.5) if context else 0.5
        ann_top1 = context.get('ann_top1', 0.5) if context else 0.5
        overlap_ratio = context.get('overlap_ratio', 0.1) if context else 0.1
        hyde_k = context.get('hyde_k', 3) if context else 3
        
        return QueryFeatures(
            query_len=query_len,
            has_code_symbol=has_code_symbol,
            has_error_token=has_error_token,
            has_path_or_file=has_path_or_file,
            has_numeric_id=has_numeric_id,
            bm25_top1=bm25_top1,
            ann_top1=ann_top1,
            overlap_ratio=overlap_ratio,
            hyde_k=hyde_k,
            word_count=word_count,
            char_count=char_count,
            complexity_score=complexity_score
        )
    
    def _regex_search(self, pattern: str, text: str) -> bool:
        """Check if regex pattern matches text."""
        import re
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False
    
    def _calculate_complexity_score(self, query: str, features: Dict) -> float:
        """Calculate query complexity score (0-1 scale)."""
        
        complexity = 0.0
        
        # Length-based complexity
        if features['word_count'] > 10:
            complexity += 0.3
        elif features['word_count'] > 5:
            complexity += 0.15
            
        # Pattern-based complexity
        if features['has_code']:
            complexity += 0.25
        if features['has_error']:
            complexity += 0.2
        if features['has_file']:
            complexity += 0.15
            
        # Structural complexity
        if '?' in query:
            complexity += 0.1
        if any(word in query.lower() for word in ['how', 'why', 'what', 'when', 'where']):
            complexity += 0.1
            
        return min(complexity, 1.0)


class DynamicFusionModel:
    """ML model for predicting optimal alpha/beta parameters."""
    
    def __init__(self, model_type: str = 'ridge'):
        """Initialize fusion model.
        
        Args:
            model_type: 'ridge', 'random_forest', or 'lightgbm'
        """
        self.model_type = model_type
        self.alpha_model = None
        self.beta_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def train(self, training_data: List[TrainingExample]) -> Dict[str, Any]:
        """Train the fusion parameter prediction models."""
        
        logger.info(f"Training dynamic fusion model with {len(training_data)} examples")
        
        # Prepare feature matrix
        X, y_alpha, y_beta = self._prepare_training_data(training_data)
        
        # Split data for validation
        X_train, X_test, y_alpha_train, y_alpha_test, y_beta_train, y_beta_test = (
            train_test_split(X, y_alpha, y_beta, test_size=0.2, random_state=42)
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train alpha model
        if self.model_type == 'ridge':
            self.alpha_model = Ridge(alpha=1.0, random_state=42)
            self.beta_model = Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'random_forest':
            self.alpha_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.beta_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Fit models
        self.alpha_model.fit(X_train_scaled, y_alpha_train)
        self.beta_model.fit(X_train_scaled, y_beta_train)
        
        # Evaluate
        alpha_pred = self.alpha_model.predict(X_test_scaled)
        beta_pred = self.beta_model.predict(X_test_scaled)
        
        alpha_mse = mean_squared_error(y_alpha_test, alpha_pred)
        beta_mse = mean_squared_error(y_beta_test, beta_pred)
        
        # Cross-validation scores
        alpha_cv = cross_val_score(self.alpha_model, X_train_scaled, y_alpha_train, cv=5, scoring='neg_mean_squared_error')
        beta_cv = cross_val_score(self.beta_model, X_train_scaled, y_beta_train, cv=5, scoring='neg_mean_squared_error')
        
        results = {
            'alpha_mse': alpha_mse,
            'beta_mse': beta_mse,
            'alpha_cv_mean': -alpha_cv.mean(),
            'alpha_cv_std': alpha_cv.std(),
            'beta_cv_mean': -beta_cv.mean(),
            'beta_cv_std': beta_cv.std(),
            'training_samples': len(training_data),
            'model_type': self.model_type
        }
        
        logger.info(f"Training complete: Alpha MSE={alpha_mse:.4f}, Beta MSE={beta_mse:.4f}")
        
        return results
    
    def predict(self, features: QueryFeatures) -> Tuple[float, float]:
        """Predict optimal alpha and beta values for query features."""
        
        if self.alpha_model is None or self.beta_model is None:
            raise ValueError("Model not trained yet")
            
        # Convert features to array
        X = self._features_to_array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict and clamp to safe ranges
        alpha_pred = float(self.alpha_model.predict(X_scaled)[0])
        beta_pred = float(self.beta_model.predict(X_scaled)[0])
        
        # Clamp to safe ranges [0.3, 1.5] as specified
        alpha_clamped = np.clip(alpha_pred, 0.3, 1.5)
        beta_clamped = np.clip(beta_pred, 0.3, 1.5)
        
        return alpha_clamped, beta_clamped
    
    def _prepare_training_data(self, training_data: List[TrainingExample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert training examples to feature matrices."""
        
        # Extract feature names from first example
        sample_features = training_data[0].features
        self.feature_names = [
            'query_len', 'has_code_symbol', 'has_error_token', 'has_path_or_file',
            'has_numeric_id', 'bm25_top1', 'ann_top1', 'overlap_ratio', 'hyde_k',
            'word_count', 'char_count', 'complexity_score'
        ]
        
        # Build matrices
        X = np.array([self._features_to_array(ex.features) for ex in training_data])
        y_alpha = np.array([ex.optimal_alpha for ex in training_data])
        y_beta = np.array([ex.optimal_beta for ex in training_data])
        
        return X, y_alpha, y_beta
    
    def _features_to_array(self, features: QueryFeatures) -> np.ndarray:
        """Convert QueryFeatures to numpy array."""
        
        return np.array([
            features.query_len,
            float(features.has_code_symbol),
            float(features.has_error_token),
            float(features.has_path_or_file),
            float(features.has_numeric_id),
            features.bm25_top1,
            features.ann_top1,
            features.overlap_ratio,
            features.hyde_k,
            features.word_count,
            features.char_count,
            features.complexity_score
        ])


class LearnedPlanSelector:
    """ML classifier for optimal plan selection."""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.plan_classes = ['explore', 'verify', 'exploit']
        
    def train(self, training_data: List[TrainingExample]) -> Dict[str, Any]:
        """Train the plan selection classifier."""
        
        logger.info(f"Training learned plan selector with {len(training_data)} examples")
        
        # Prepare data
        X, y = self._prepare_training_data(training_data)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train with hyperparameter tuning
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs']
        }
        
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'training_samples': len(training_data),
            'class_distribution': np.bincount(y) / len(y)
        }
        
        logger.info(f"Plan selector trained: Accuracy={accuracy:.4f}")
        
        return results
    
    def predict(self, features: QueryFeatures, context: Optional[Dict] = None) -> str:
        """Predict optimal plan for query features."""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Add context features for enhanced prediction
        enhanced_features = self._enhance_features_with_context(features, context)
        
        X = self._features_to_array(enhanced_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        
        return self.plan_classes[prediction]
    
    def predict_proba(self, features: QueryFeatures, context: Optional[Dict] = None) -> Dict[str, float]:
        """Get prediction probabilities for all plans."""
        
        enhanced_features = self._enhance_features_with_context(features, context)
        X = self._features_to_array(enhanced_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)[0]
        
        return {
            plan: float(prob) 
            for plan, prob in zip(self.plan_classes, proba)
        }
    
    def _prepare_training_data(self, training_data: List[TrainingExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert training data to feature matrix and labels."""
        
        # Extract features with enhanced context
        X = np.array([
            self._features_to_array(
                self._enhance_features_with_context(ex.features, None)
            ) 
            for ex in training_data
        ])
        
        # Convert plan names to indices
        y = np.array([
            self.plan_classes.index(ex.best_plan) 
            for ex in training_data
        ])
        
        return X, y
    
    def _enhance_features_with_context(self, features: QueryFeatures, context: Optional[Dict]) -> QueryFeatures:
        """Enhance features with context information for plan selection."""
        
        # Add previous pack statistics if available
        if context:
            # Create enhanced feature set for plan selection
            enhanced = QueryFeatures(
                query_len=features.query_len,
                has_code_symbol=features.has_code_symbol,
                has_error_token=features.has_error_token,
                has_path_or_file=features.has_path_or_file,
                has_numeric_id=features.has_numeric_id,
                bm25_top1=features.bm25_top1,
                ann_top1=features.ann_top1,
                overlap_ratio=features.overlap_ratio,
                hyde_k=features.hyde_k,
                word_count=features.word_count,
                char_count=features.char_count,
                complexity_score=features.complexity_score
            )
            
            return enhanced
        
        return features
    
    def _features_to_array(self, features: QueryFeatures) -> np.ndarray:
        """Convert enhanced features to array for plan selection."""
        
        return np.array([
            features.query_len,
            float(features.has_code_symbol),
            float(features.has_error_token),
            float(features.has_path_or_file),
            float(features.has_numeric_id),
            features.bm25_top1,
            features.ann_top1,
            features.overlap_ratio,
            features.hyde_k,
            features.word_count,
            features.char_count,
            features.complexity_score
        ])


class Iteration3MLPipeline:
    """Complete ML pipeline for Iteration 3."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = QueryFeatureExtractor()
        self.fusion_model = DynamicFusionModel(model_type='ridge')
        self.plan_selector = LearnedPlanSelector()
        
    def generate_synthetic_training_data(self, n_samples: int = 1000) -> List[TrainingExample]:
        """Generate synthetic training data based on Iteration 1 & 2 patterns."""
        
        logger.info(f"Generating {n_samples} synthetic training examples")
        
        training_examples = []
        
        # Query templates with different characteristics
        query_templates = [
            # Code-focused queries
            ("async error handling in TypeScript", "code"),
            ("React component optimization", "code"),  
            ("database indexing performance", "code"),
            ("authentication middleware", "code"),
            ("microservices patterns", "code"),
            
            # Error-focused queries
            ("undefined reference error", "error"),
            ("memory leak investigation", "error"),
            ("stack overflow exception", "error"),
            ("build failure analysis", "error"),
            
            # General queries
            ("machine learning algorithms", "general"),
            ("data visualization techniques", "general"),
            ("API design principles", "general"),
            ("cloud deployment strategies", "general"),
        ]
        
        np.random.seed(42)  # Reproducible results
        
        for i in range(n_samples):
            # Select random query template
            query_template, category = query_templates[i % len(query_templates)]
            
            # Add variation to query
            variations = [
                f"how to {query_template}",
                f"best practices for {query_template}",
                f"troubleshooting {query_template}",
                f"{query_template} examples",
                f"advanced {query_template}",
                query_template  # original
            ]
            
            query = variations[i % len(variations)]
            
            # Extract features
            context = {
                'bm25_top1': np.random.uniform(0.1, 0.9),
                'ann_top1': np.random.uniform(0.1, 0.9),
                'overlap_ratio': np.random.uniform(0.05, 0.5),
                'hyde_k': np.random.choice([2, 3, 4, 5])
            }
            
            features = self.feature_extractor.extract_features(query, context)
            
            # Generate optimal parameters based on query characteristics
            optimal_alpha, optimal_beta = self._generate_optimal_params(features, category)
            best_plan = self._generate_optimal_plan(features, category)
            
            # Simulate nDCG score based on parameters
            ndcg_score = self._simulate_ndcg(features, optimal_alpha, optimal_beta, best_plan)
            
            training_examples.append(TrainingExample(
                features=features,
                optimal_alpha=optimal_alpha,
                optimal_beta=optimal_beta,
                best_plan=best_plan,
                ndcg_score=ndcg_score,
                query_text=query
            ))
        
        logger.info(f"Generated {len(training_examples)} synthetic training examples")
        return training_examples
    
    def _generate_optimal_params(self, features: QueryFeatures, category: str) -> Tuple[float, float]:
        """Generate realistic optimal alpha/beta parameters based on query characteristics."""
        
        base_alpha = 0.7
        base_beta = 0.5
        
        # Code queries favor lexical matching
        if category == "code" or features.has_code_symbol:
            alpha = base_alpha + np.random.normal(0.2, 0.1)
            beta = base_beta + np.random.normal(-0.1, 0.1)
        
        # Error queries need balanced approach
        elif category == "error" or features.has_error_token:
            alpha = base_alpha + np.random.normal(0.0, 0.15)
            beta = base_beta + np.random.normal(0.1, 0.1)
        
        # General queries favor semantic
        else:
            alpha = base_alpha + np.random.normal(-0.1, 0.1)
            beta = base_beta + np.random.normal(0.2, 0.1)
            
        # Add complexity-based adjustment
        if features.complexity_score > 0.5:
            beta += 0.1  # Complex queries benefit from semantic understanding
            
        # Clamp to valid ranges
        alpha = np.clip(alpha, 0.3, 1.5)
        beta = np.clip(beta, 0.3, 1.5)
        
        return alpha, beta
    
    def _generate_optimal_plan(self, features: QueryFeatures, category: str) -> str:
        """Generate realistic optimal plan based on query characteristics."""
        
        # Simple heuristic-based assignment with some randomness
        if features.has_error_token or category == "error":
            return np.random.choice(['verify', 'explore'], p=[0.7, 0.3])
        
        elif features.complexity_score > 0.6:
            return np.random.choice(['explore', 'exploit'], p=[0.6, 0.4])
            
        elif features.has_code_symbol:
            return np.random.choice(['exploit', 'explore'], p=[0.6, 0.4])
            
        else:
            return np.random.choice(['exploit', 'explore', 'verify'], p=[0.5, 0.3, 0.2])
    
    def _simulate_ndcg(self, features: QueryFeatures, alpha: float, beta: float, plan: str) -> float:
        """Simulate nDCG score based on parameters and plan."""
        
        base_score = 0.65
        
        # Parameter alignment bonus
        if features.has_code_symbol and alpha > beta:
            base_score += 0.1
        if not features.has_code_symbol and beta > alpha:
            base_score += 0.08
            
        # Plan alignment bonus
        if plan == 'verify' and features.has_error_token:
            base_score += 0.05
        if plan == 'explore' and features.complexity_score > 0.5:
            base_score += 0.07
            
        # Add noise
        score = base_score + np.random.normal(0, 0.1)
        return np.clip(score, 0.1, 1.0)
    
    def train_models(self, training_data: Optional[List[TrainingExample]] = None) -> Dict[str, Any]:
        """Train both fusion and plan selection models."""
        
        if training_data is None:
            training_data = self.generate_synthetic_training_data()
            
        # Train fusion model
        fusion_results = self.fusion_model.train(training_data)
        
        # Train plan selector
        plan_results = self.plan_selector.train(training_data)
        
        # Combined results
        results = {
            'fusion_model': fusion_results,
            'plan_selector': plan_results,
            'training_data_size': len(training_data),
            'timestamp': time.time()
        }
        
        return results
    
    def save_models(self) -> Dict[str, str]:
        """Save trained models to disk."""
        
        model_paths = {}
        
        # Save fusion model
        fusion_path = self.models_dir / 'dynamic_fusion_model.joblib'
        joblib.dump({
            'alpha_model': self.fusion_model.alpha_model,
            'beta_model': self.fusion_model.beta_model,
            'scaler': self.fusion_model.scaler,
            'feature_names': self.fusion_model.feature_names,
            'model_type': self.fusion_model.model_type
        }, fusion_path)
        model_paths['fusion_model'] = str(fusion_path)
        
        # Save plan selector
        plan_path = self.models_dir / 'learned_plan_selector.joblib'
        joblib.dump({
            'model': self.plan_selector.model,
            'scaler': self.plan_selector.scaler,
            'feature_names': self.plan_selector.feature_names,
            'plan_classes': self.plan_selector.plan_classes
        }, plan_path)
        model_paths['plan_selector'] = str(plan_path)
        
        # Save feature extractor configuration
        feature_path = self.models_dir / 'feature_extractor.json'
        with open(feature_path, 'w') as f:
            json.dump({
                'code_patterns': self.feature_extractor.code_patterns,
                'error_patterns': self.feature_extractor.error_patterns,
                'file_patterns': self.feature_extractor.file_patterns,
            }, f, indent=2)
        model_paths['feature_extractor'] = str(feature_path)
        
        logger.info(f"Models saved to {self.models_dir}")
        return model_paths
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        
        try:
            # Load fusion model
            fusion_path = self.models_dir / 'dynamic_fusion_model.joblib'
            if fusion_path.exists():
                fusion_data = joblib.load(fusion_path)
                self.fusion_model.alpha_model = fusion_data['alpha_model']
                self.fusion_model.beta_model = fusion_data['beta_model']
                self.fusion_model.scaler = fusion_data['scaler']
                self.fusion_model.feature_names = fusion_data['feature_names']
                self.fusion_model.model_type = fusion_data['model_type']
            
            # Load plan selector
            plan_path = self.models_dir / 'learned_plan_selector.joblib'
            if plan_path.exists():
                plan_data = joblib.load(plan_path)
                self.plan_selector.model = plan_data['model']
                self.plan_selector.scaler = plan_data['scaler']
                self.plan_selector.feature_names = plan_data['feature_names']
                self.plan_selector.plan_classes = plan_data['plan_classes']
            
            # Load feature extractor config
            feature_path = self.models_dir / 'feature_extractor.json'
            if feature_path.exists():
                with open(feature_path) as f:
                    config = json.load(f)
                    self.feature_extractor.code_patterns = config['code_patterns']
                    self.feature_extractor.error_patterns = config['error_patterns']
                    self.feature_extractor.file_patterns = config['file_patterns']
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def predict_parameters(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict optimal parameters for a query."""
        
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_features(query, context)
        
        # Predict fusion parameters
        alpha, beta = self.fusion_model.predict(features)
        
        # Predict plan
        plan = self.plan_selector.predict(features, context)
        plan_proba = self.plan_selector.predict_proba(features, context)
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'alpha': alpha,
            'beta': beta,
            'plan': plan,
            'plan_probabilities': plan_proba,
            'prediction_time_ms': prediction_time,
            'features': features.__dict__,
            'query': query
        }


def train_iteration3_models():
    """Main training function for Iteration 3 models."""
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Iteration 3 ML training pipeline")
    
    # Initialize pipeline
    pipeline = Iteration3MLPipeline()
    
    # Train models
    results = pipeline.train_models()
    
    # Save models
    model_paths = pipeline.save_models()
    
    # Validate model loading and prediction speed
    pipeline_test = Iteration3MLPipeline()
    load_success = pipeline_test.load_models()
    
    if load_success:
        # Test prediction speed
        test_queries = [
            "async error handling in TypeScript",
            "React component optimization techniques",
            "database indexing and performance tuning",
            "authentication middleware implementation",
            "microservices communication patterns"
        ]
        
        prediction_times = []
        for query in test_queries:
            prediction = pipeline_test.predict_parameters(query)
            prediction_times.append(prediction['prediction_time_ms'])
        
        avg_prediction_time = np.mean(prediction_times)
        max_prediction_time = np.max(prediction_times)
        
        logger.info(f"Model loading successful")
        logger.info(f"Average prediction time: {avg_prediction_time:.1f}ms")
        logger.info(f"Maximum prediction time: {max_prediction_time:.1f}ms")
        
        # Check if within 200ms budget
        speed_check_passed = max_prediction_time <= 200
        logger.info(f"Speed requirement (<200ms): {'PASS' if speed_check_passed else 'FAIL'}")
        
    else:
        logger.error("Model loading failed")
        
    # Save final results
    final_results = {
        'training_results': results,
        'model_paths': model_paths,
        'validation': {
            'load_success': load_success,
            'avg_prediction_time_ms': avg_prediction_time if load_success else None,
            'max_prediction_time_ms': max_prediction_time if load_success else None,
            'speed_requirement_met': speed_check_passed if load_success else False
        },
        'timestamp': time.time()
    }
    
    results_path = Path('artifacts') / f'iter3_training_results_{int(time.time())}.json'
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
        
    logger.info(f"Training results saved to {results_path}")
    
    return final_results


if __name__ == "__main__":
    train_iteration3_models()