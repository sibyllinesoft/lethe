"""
Cross-encoder implementation for reranking.

Uses public Hugging Face checkpoints for query-document scoring.
Implements efficient batching and handles various model architectures.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available - using fallback scorer")


class CrossEncoderReranker:
    """
    Cross-encoder reranker using pre-trained models.
    
    Supports various architectures including:
    - MS-MARCO trained models
    - BERT-based models
    - MiniLM variants for efficiency
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize cross-encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cpu/cuda), auto-detected if None
            max_length: Maximum input sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._detect_device(device)
        
        # Model components
        self.tokenizer = None
        self.model = None
        
        # Performance tracking
        self.inference_times: List[float] = []
        
        # Initialize model
        self._load_model()
        
        logger.info(f"CrossEncoderReranker initialized with {model_name} on {self.device}")
    
    def _detect_device(self, device: Optional[str]) -> str:
        """Detect best available device."""
        if device:
            return device
        
        if TRANSFORMERS_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        
        return "cpu"
    
    def _load_model(self):
        """Load tokenizer and model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - using fallback implementation")
            return
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Warmup inference
            self._warmup()
            
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
    
    def _warmup(self):
        """Warmup model with dummy input."""
        if not self.model or not self.tokenizer:
            return
        
        try:
            dummy_query = "test query"
            dummy_doc = "test document for warmup"
            
            # Single warmup inference
            self._score_single_pair(dummy_query, dummy_doc)
            
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def score_pairs(
        self,
        query: str,
        doc_ids: List[str],
        documents: Optional[Dict[str, str]] = None,
        batch_size: int = 32,
        max_length: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Score query-document pairs using cross-encoder.
        
        Args:
            query: Query text
            doc_ids: List of document IDs to score
            documents: Optional mapping of doc_id -> content
            batch_size: Batch size for efficient inference
            max_length: Override default max_length
        
        Returns:
            Dictionary mapping doc_id -> relevance_score
        """
        if not doc_ids:
            return {}
        
        if not self.model or not self.tokenizer:
            # Fallback to random scores
            return self._fallback_scoring(doc_ids)
        
        max_len = max_length or self.max_length
        scores = {}
        
        start_time = time.time()
        
        try:
            # Process in batches for efficiency
            for i in range(0, len(doc_ids), batch_size):
                batch_doc_ids = doc_ids[i:i + batch_size]
                batch_scores = self._score_batch(
                    query, batch_doc_ids, documents, max_len
                )
                scores.update(batch_scores)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            logger.debug(
                f"Cross-encoder scoring: {len(doc_ids)} pairs in {inference_time:.3f}s "
                f"({len(doc_ids)/inference_time:.1f} pairs/sec)"
            )
            
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            # Fallback to random scores
            scores = self._fallback_scoring(doc_ids)
        
        return scores
    
    def _score_batch(
        self,
        query: str,
        doc_ids: List[str],
        documents: Optional[Dict[str, str]],
        max_length: int
    ) -> Dict[str, float]:
        """Score a batch of query-document pairs."""
        import torch
        
        # Prepare input pairs
        pairs = []
        valid_doc_ids = []
        
        for doc_id in doc_ids:
            # Get document content
            if documents and doc_id in documents:
                doc_content = documents[doc_id]
            else:
                # Use doc_id as content (fallback)
                doc_content = f"Document {doc_id}"
            
            pairs.append((query, doc_content))
            valid_doc_ids.append(doc_id)
        
        if not pairs:
            return {}
        
        # Tokenize batch
        inputs = self.tokenizer(
            pairs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get logits and convert to scores
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            
            # Handle different output formats
            if logits.shape[-1] == 1:
                # Regression output
                scores_tensor = logits.squeeze(-1)
            else:
                # Classification output - use positive class
                scores_tensor = torch.softmax(logits, dim=-1)[:, -1]
            
            scores_array = scores_tensor.cpu().numpy()
        
        # Create result dictionary
        result = {}
        for doc_id, score in zip(valid_doc_ids, scores_array):
            result[doc_id] = float(score)
        
        return result
    
    def _score_single_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            (query, document),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            
            if logits.shape[-1] == 1:
                score = logits.squeeze(-1).item()
            else:
                score = torch.softmax(logits, dim=-1)[0, -1].item()
        
        return score
    
    def _fallback_scoring(self, doc_ids: List[str]) -> Dict[str, float]:
        """Fallback scoring when model is not available."""
        # Return random scores for testing purposes
        np.random.seed(42)  # For reproducibility
        scores = {}
        
        for doc_id in doc_ids:
            # Use doc_id hash for consistency
            hash_val = hash(doc_id) % 1000000
            score = hash_val / 1000000.0  # Normalize to [0,1]
            scores[doc_id] = score
        
        logger.warning(f"Using fallback scoring for {len(doc_ids)} documents")
        return scores
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {"total_inferences": 0}
        
        times = np.array(self.inference_times)
        
        return {
            "total_inferences": len(self.inference_times),
            "mean_inference_time": float(np.mean(times)),
            "median_inference_time": float(np.median(times)),
            "p95_inference_time": float(np.percentile(times, 95)),
            "total_inference_time": float(np.sum(times))
        }
    
    def clear_stats(self):
        """Clear performance statistics."""
        self.inference_times.clear()
        logger.info("Cross-encoder performance stats cleared")


class CrossEncoderEnsemble:
    """
    Ensemble of cross-encoders for improved reranking.
    
    Combines multiple models to get more robust relevance scores.
    """
    
    def __init__(self, model_names: List[str], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.
        
        Args:
            model_names: List of HuggingFace model names
            weights: Optional weights for ensemble combination
        """
        self.model_names = model_names
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)
        
        if len(self.weights) != len(model_names):
            raise ValueError("Number of weights must match number of models")
        
        # Initialize individual rerankers
        self.rerankers = []
        for name in model_names:
            try:
                reranker = CrossEncoderReranker(model_name=name)
                self.rerankers.append(reranker)
                logger.info(f"Loaded ensemble member: {name}")
            except Exception as e:
                logger.error(f"Failed to load ensemble member {name}: {e}")
        
        if not self.rerankers:
            logger.error("No models loaded in ensemble")
    
    def score_pairs(
        self,
        query: str,
        doc_ids: List[str],
        documents: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Score pairs using ensemble of models."""
        if not self.rerankers:
            # Fallback to single model
            fallback = CrossEncoderReranker()
            return fallback.score_pairs(query, doc_ids, documents, **kwargs)
        
        # Get scores from all models
        all_scores = []
        for reranker in self.rerankers:
            model_scores = reranker.score_pairs(query, doc_ids, documents, **kwargs)
            all_scores.append(model_scores)
        
        # Weighted combination
        ensemble_scores = {}
        for doc_id in doc_ids:
            weighted_score = 0.0
            total_weight = 0.0
            
            for scores, weight in zip(all_scores, self.weights):
                if doc_id in scores:
                    weighted_score += weight * scores[doc_id]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_scores[doc_id] = weighted_score / total_weight
            else:
                ensemble_scores[doc_id] = 0.0
        
        logger.debug(f"Ensemble scoring completed for {len(doc_ids)} documents")
        return ensemble_scores
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance stats from all ensemble members."""
        stats = {"ensemble_size": len(self.rerankers)}
        
        for i, reranker in enumerate(self.rerankers):
            model_stats = reranker.get_performance_stats()
            stats[f"model_{i}_{self.model_names[i]}"] = model_stats
        
        return stats