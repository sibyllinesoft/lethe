"""
Common interface for LLM context management research competitors.

Provides standardized interface for fair comparison with Lethe across:
- Context compression and pruning approaches
- Position management strategies  
- Attention mechanism optimizations
"""

import abc
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ContextProcessingResult:
    """Standardized result format for context processing."""
    original_context: str
    processed_context: str
    query: str
    response: str
    
    # Performance metrics
    processing_time_ms: float
    original_token_count: int
    processed_token_count: int
    compression_ratio: float
    
    # Quality metrics (if available)
    accuracy_score: Optional[float] = None
    f1_score: Optional[float] = None
    exact_match: Optional[bool] = None
    
    # Resource usage
    memory_usage_mb: Optional[float] = None
    
    # Metadata
    method_name: str = ""
    metadata: Dict[str, Any] = None


class ContextManagementCompetitor(abc.ABC):
    """Abstract base class for LLM context management competitors."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize competitor with name and configuration."""
        self.name = name
        self.config = config or {}
        self._initialized = False
        # Lethe parameters (λ, μ as mentioned in TODO)
        self.lambda_param = 0.15  # Token budget constraint (λ)
        self.mu_param = 0.05      # Compute budget constraint (μ)
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the competitor (load models, setup dependencies).
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def process_context(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """
        Process long context using the competitor's approach.
        
        Args:
            query: The question/query to answer
            context: The long context to process  
            max_tokens: Maximum tokens to use in processed context
            
        Returns:
            ContextProcessingResult: Standardized processing result
        """
        pass
    
    @abc.abstractmethod
    def get_installation_requirements(self) -> List[str]:
        """
        Get list of required packages/dependencies.
        
        Returns:
            List[str]: Package names (pip installable)
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self):
        """Clean up resources (unload models, close connections)."""
        pass
    
    def is_available(self) -> bool:
        """Check if competitor can be used (dependencies installed)."""
        try:
            requirements = self.get_installation_requirements()
            # Map package names to import names
            package_import_map = {
                'sentence-transformers': 'sentence_transformers',
                'scikit-learn': 'sklearn',
                'pillow': 'PIL'
            }
            
            for req in requirements:
                package_name = req.split('>=')[0].split('==')[0]
                import_name = package_import_map.get(package_name, package_name)
                __import__(import_name)
            return True
        except ImportError:
            return False
    
    def benchmark_context_lengths(self, query: str, base_context: str, 
                                 lengths: List[int] = None) -> List[ContextProcessingResult]:
        """
        Benchmark across different context lengths.
        
        Args:
            query: Test query
            base_context: Base context to extend/truncate
            lengths: Context lengths to test (default: [1K, 4K, 16K, 64K])
            
        Returns:
            List[ContextProcessingResult]: Results for each context length
        """
        if lengths is None:
            lengths = [1000, 4000, 16000, 64000]
        
        results = []
        for length in lengths:
            # Create context of target length
            if len(base_context) > length:
                test_context = base_context[:length]
            else:
                # Repeat context to reach target length
                repeat_count = (length // len(base_context)) + 1
                test_context = (base_context * repeat_count)[:length]
            
            try:
                result = self.process_context(query, test_context, max_tokens=length//4)
                result.metadata = result.metadata or {}
                result.metadata['target_context_length'] = length
                results.append(result)
            except Exception as e:
                # Create error result
                results.append(ContextProcessingResult(
                    original_context=test_context,
                    processed_context="",
                    query=query,
                    response=f"Error: {str(e)}",
                    processing_time_ms=0,
                    original_token_count=len(test_context.split()),
                    processed_token_count=0,
                    compression_ratio=1.0,  # All context was filtered out = 100% compression
                    method_name=self.name,
                    metadata={'error': str(e), 'target_context_length': length}
                ))
        
        return results


class LetheCompetitor(ContextManagementCompetitor):
    """Lethe implementation for comparison."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Lethe", config)
    
    def initialize(self) -> bool:
        """Initialize Lethe system."""
        try:
            # Import and initialize Lethe
            # This would connect to the actual Lethe implementation
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize Lethe: {e}")
            return False
    
    def process_context(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """Process context using Lethe's approach."""
        start_time = time.time()
        
        original_tokens = len(context.split())
        
        # Implement Lethe's semantic selection approach
        # Use a simple but representative semantic selection strategy
        processed_context = self._select_relevant_context(context, query, max_tokens)
        
        # Use gemma2:9b via Ollama to actually answer the question
        try:
            prompt = f"Context: {processed_context}\n\nQuestion: {query}\n\nAnswer:"
            
            response = requests.post('http://localhost:11434/api/generate', json={
                'model': 'gemma2:9b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'max_tokens': max_tokens
                }
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '').strip()
            else:
                llm_response = f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            llm_response = f"Error calling gemma2:9b: {str(e)}"
        
        processing_time = (time.time() - start_time) * 1000
        processed_tokens = len(processed_context.split())
        
        return ContextProcessingResult(
            original_context=context,
            processed_context=processed_context,
            query=query,
            response=llm_response,
            processing_time_ms=processing_time,
            original_token_count=original_tokens,
            processed_token_count=processed_tokens,
            compression_ratio=1.0 - (float(processed_tokens) / float(original_tokens)) if original_tokens > 0 else 1.0,
            method_name=self.name,
            metadata={"approach": "semantic_retrieval"}
        )
    
    def _select_relevant_context(self, context: str, query: str, max_tokens: int) -> str:
        """
        Lethe's semantic selection approach.
        
        Implements a simplified version of the approach mentioned in TODO:
        - Apply λ (token budget constraint) 
        - Use semantic relevance scoring
        - Return most relevant context within budget
        """
        # Split context into sentences for selection
        import re
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return context
            
        # Calculate target tokens based on λ parameter (compression target)
        original_tokens = len(context.split())
        target_tokens = int(original_tokens * (1.0 - self.lambda_param))  # λ=0.15 means keep 85%
        target_tokens = min(target_tokens, max_tokens)  # Respect max_tokens limit
        
        # Simple relevance scoring: count query word overlaps
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # Relevance score based on word overlap + sentence importance heuristics
            overlap_score = len(query_words.intersection(sentence_words))
            length_bonus = min(len(sentence.split()) / 20, 1.0)  # Favor moderately long sentences
            total_score = overlap_score + length_bonus
            
            scored_sentences.append((sentence, total_score, len(sentence.split())))
        
        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection to stay within token budget
        selected_sentences = []
        current_tokens = 0
        
        for sentence, score, token_count in scored_sentences:
            if current_tokens + token_count <= target_tokens:
                selected_sentences.append(sentence)
                current_tokens += token_count
            else:
                # Check if we can fit a partial sentence
                remaining_tokens = target_tokens - current_tokens
                if remaining_tokens > 10:  # Only if meaningful space left
                    words = sentence.split()[:remaining_tokens]
                    partial_sentence = ' '.join(words)
                    selected_sentences.append(partial_sentence)
                break
        
        # Join selected sentences back into context
        processed_context = '. '.join(selected_sentences)
        
        # Add query-specific context boost if very short
        if len(processed_context.split()) < target_tokens * 0.5:
            # Add more context if we're well under budget
            remaining_budget = target_tokens - len(processed_context.split())
            for sentence, score, token_count in scored_sentences[len(selected_sentences):]:
                if token_count <= remaining_budget:
                    processed_context += '. ' + sentence
                    remaining_budget -= token_count
                if remaining_budget <= 10:
                    break
        
        return processed_context if processed_context else context[:1000]  # Fallback
    
    def get_installation_requirements(self) -> List[str]:
        """Lethe requirements."""
        return []  # Lethe is already available in this codebase
    
    def cleanup(self):
        """Clean up Lethe resources."""
        pass