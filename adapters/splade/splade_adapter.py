#!/usr/bin/env python3
"""
SPLADE v2 System Adapter Implementation
Real SPLADE sparse lexical expansion system
"""

import sys
sys.path.append('/app')

from adapter_harness import SystemAdapter, SearchResult
from typing import Dict, List, Any
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import time
import json
from pathlib import Path


class SPLADEAdapter(SystemAdapter):
    """SPLADE v2 sparse lexical expansion adapter"""
    
    def __init__(self):
        self.model_name = "naver/splade-cocondenser-ensembledistil"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.indexes = {}  # Store built indexes
        self.load_model()
    
    def load_model(self):
        """Load SPLADE model and tokenizer"""
        print(f"🔄 Loading SPLADE model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ SPLADE model loaded on {self.device}")
    
    def encode_sparse(self, text: str) -> Dict[str, float]:
        """Encode text to sparse representation"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, max_length=512).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply ReLU and sum over sequence dimension
            sparse_rep = torch.relu(logits).sum(dim=1).squeeze()
            
            # Convert to dictionary (token_id -> weight)
            sparse_dict = {}
            for idx, weight in enumerate(sparse_rep):
                if weight > 0:
                    token = self.tokenizer.decode([idx])
                    sparse_dict[token] = float(weight)
            
            return sparse_dict
    
    def build_index(self, dataset: str) -> str:
        """Build search index for dataset"""
        index_id = f"splade_{dataset}_{int(time.time())}"
        print(f"🔍 Building SPLADE index: {index_id}")
        
        # Load dataset documents (placeholder - would load real dataset)
        documents = self.load_dataset(dataset)
        
        # Encode all documents
        doc_embeddings = {}
        for doc_id, doc_text in documents.items():
            doc_embeddings[doc_id] = self.encode_sparse(doc_text)
        
        # Store index
        self.indexes[index_id] = {
            "documents": documents,
            "embeddings": doc_embeddings,
            "dataset": dataset,
            "created": time.time()
        }
        
        print(f"✅ SPLADE index built: {len(documents)} documents")
        return index_id
    
    def load_dataset(self, dataset: str) -> Dict[str, str]:
        """Load dataset documents (placeholder implementation)"""
        # In production, this would load the actual dataset
        return {
            f"doc_{i}": f"Sample document {i} for {dataset} with some technical content about retrieval and search."
            for i in range(100)  # Placeholder docs
        }
    
    def search(self, query: str, budget: float, k: int, index_id: str) -> SearchResult:
        """Search with budget constraint"""
        start_time = time.time()
        
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")
        
        index = self.indexes[index_id]
        
        # Encode query
        query_embedding = self.encode_sparse(query)
        
        # Compute similarities
        scores = {}
        for doc_id, doc_embedding in index["embeddings"].items():
            # Sparse dot product
            score = sum(query_embedding.get(token, 0) * weight 
                       for token, weight in doc_embedding.items())
            scores[doc_id] = score
        
        # Apply budget constraint (simulate token limit)
        max_docs = max(1, int(len(scores) * budget))  # budget as fraction
        
        # Sort and get top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_docs]
        top_k = sorted_docs[:k]
        
        candidates = [
            {
                "doc_id": doc_id,
                "score": score,
                "text": index["documents"][doc_id][:200] + "..."  # Preview
            }
            for doc_id, score in top_k
        ]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            candidates=candidates,
            timings={"middleware_ms": elapsed_ms},
            metadata={
                "system": "splade_v2",
                "query_tokens": len(query_embedding),
                "budget_applied": budget,
                "total_docs": len(scores)
            }
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Return system metadata"""
        return {
            "name": "SPLADE_v2",
            "version": "2.0",
            "model": self.model_name,
            "description": "Sparse lexical expansion for rare term recovery",
            "category": "Learned Sparse",
            "device": str(self.device),
            "config": {
                "max_seq_length": 512,
                "sparse_threshold": 0.0,
                "aggregation": "sum"
            }
        }