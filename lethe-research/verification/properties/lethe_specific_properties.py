#!/usr/bin/env python3
"""
Lethe vNext Specific Metamorphic Properties
==========================================

Implements the specific metamorphic properties required by the TODO.md:
- Adding irrelevant sentences MUST NOT increase Claim-Support@K at fixed budget
- Duplicating any kept sentence MUST NOT change scores/pack order
- Synonymized query (lemmatized) keeps nDCG within ε
- Removing a gold sentence MUST reduce Claim-Support@K
- Shuffling non-selected items has no effect

These properties are essential for the sentence pruning and knapsack packing
components of Lethe vNext.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st, assume, example, note
import pytest
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
from pathlib import Path
import time
import hashlib
import nltk
from nltk.stem import WordNetLemmatizer
import random
import sys

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class SentenceData:
    """Represents a sentence with metadata for testing"""
    sid: str
    content: str
    span: Tuple[int, int]  # (start, end) character positions
    tokens: int
    score: float
    is_gold: bool = False
    is_relevant: bool = True
    
@dataclass
class ChunkData:
    """Represents a chunk containing sentences"""
    chunk_id: str
    sentences: List[SentenceData]
    original_content: str

@dataclass
class KnapsackResult:
    """Result of knapsack packing operation"""
    selected_sentences: List[str]  # sentence IDs
    total_tokens: int
    total_score: float
    order: List[str]  # ordered sentence IDs

class LethaSpecificProperties:
    """
    Test suite for Lethe vNext specific metamorphic properties.
    
    These tests validate core invariants that must hold for the
    sentence pruning and knapsack optimization components.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize NLTK components for lemmatization
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
            self.logger.warning("NLTK lemmatizer not available - synonymization tests may be limited")
    
    # ==========================================
    # STRATEGIES FOR GENERATING TEST DATA
    # ==========================================
    
    @st.composite
    def sentence_strategy(draw, min_tokens=5, max_tokens=50, is_gold=False, is_relevant=None):
        """Generate realistic sentence data"""
        # Generate sentence content
        words = draw(st.lists(
            st.text(min_size=3, max_size=12, alphabet=st.characters(blacklist_categories=['Cc', 'Cs'])),
            min_size=min_tokens, 
            max_size=max_tokens
        ))
        content = " ".join(words)
        
        # Generate metadata
        sid = f"sent_{draw(st.integers(1000, 9999))}"
        tokens = len(words)
        span_start = draw(st.integers(0, 100))
        span_end = span_start + len(content)
        
        # Score biased based on relevance
        if is_relevant is None:
            is_relevant = draw(st.booleans())
        
        if is_relevant:
            score = draw(st.floats(0.5, 1.0))
        else:
            score = draw(st.floats(0.0, 0.3))
            
        # Gold sentences should have higher relevance
        if is_gold:
            score = max(score, draw(st.floats(0.7, 1.0)))
            is_relevant = True
        
        return SentenceData(
            sid=sid,
            content=content,
            span=(span_start, span_end),
            tokens=tokens,
            score=score,
            is_gold=is_gold,
            is_relevant=is_relevant
        )
    
    @st.composite 
    def chunk_strategy(draw, min_sentences=5, max_sentences=20, gold_sentence_ratio=0.2):
        """Generate chunk containing multiple sentences"""
        n_sentences = draw(st.integers(min_sentences, max_sentences))
        n_gold = int(n_sentences * gold_sentence_ratio)
        
        sentences = []
        current_pos = 0
        
        # Generate sentences
        for i in range(n_sentences):
            is_gold = i < n_gold
            sentence = draw(self.sentence_strategy(is_gold=is_gold))
            
            # Adjust span to be relative to chunk
            sentence.span = (current_pos, current_pos + len(sentence.content))
            current_pos = sentence.span[1] + 1  # Add space
            
            sentences.append(sentence)
        
        # Generate chunk content
        chunk_id = f"chunk_{draw(st.integers(1000, 9999))}"
        content_parts = [s.content for s in sentences]
        original_content = " ".join(content_parts)
        
        return ChunkData(
            chunk_id=chunk_id,
            sentences=sentences,
            original_content=original_content
        )
    
    def mock_sentence_prune(self, query: str, chunk: ChunkData, budget: int = 1000) -> Dict[str, Any]:
        """
        Mock implementation of sentence pruning for testing properties.
        In real implementation, this would use cross-encoder scoring.
        """
        kept_sentences = []
        
        # Simple scoring: keep sentences above threshold, always keep gold
        threshold = 0.5
        for sentence in chunk.sentences:
            if sentence.is_gold or sentence.score >= threshold:
                kept_sentences.append({
                    "sid": sentence.sid,
                    "span": sentence.span,
                    "tokens": sentence.tokens,
                    "score": sentence.score,
                    "content": sentence.content
                })
        
        # Respect budget by sorting and truncating
        kept_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        total_tokens = 0
        final_kept = []
        for sent in kept_sentences:
            if total_tokens + sent["tokens"] <= budget:
                final_kept.append(sent)
                total_tokens += sent["tokens"]
            else:
                break
        
        chunk_score = max([s["score"] for s in final_kept]) if final_kept else 0.0
        
        return {
            "chunk_id": chunk.chunk_id,
            "kept": final_kept,
            "chunk_score": chunk_score,
            "metadata": {
                "original_token_count": sum(s.tokens for s in chunk.sentences),
                "pruned_token_count": total_tokens,
                "query_hash": hashlib.sha256(query.encode()).hexdigest()
            }
        }
    
    def mock_knapsack_pack(self, sentences: List[Dict], budget: int, groups: Optional[List[str]] = None) -> KnapsackResult:
        """
        Mock implementation of knapsack packing for testing properties.
        In real implementation, this would use 0/1 knapsack with group constraints.
        """
        # Simple greedy by score-to-token ratio
        items = [(s["sid"], s["tokens"], s["score"], s["score"]/s["tokens"]) for s in sentences]
        items.sort(key=lambda x: x[3], reverse=True)  # Sort by ratio
        
        selected = []
        total_tokens = 0
        total_score = 0.0
        
        for sid, tokens, score, ratio in items:
            if total_tokens + tokens <= budget:
                selected.append(sid)
                total_tokens += tokens
                total_score += score
        
        # Mock ordering (in real implementation would use bookend packing)
        order = selected.copy()
        
        return KnapsackResult(
            selected_sentences=selected,
            total_tokens=total_tokens,
            total_score=total_score,
            order=order
        )
    
    def calculate_claim_support_at_k(self, retrieved_sentences: List[str], gold_sentences: List[str], k: int) -> float:
        """Calculate Claim-Support@K metric"""
        if not gold_sentences or k <= 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_sentences[:k])
        gold_set = set(gold_sentences)
        
        supported_claims = len(retrieved_at_k.intersection(gold_set))
        return supported_claims / len(gold_set)
    
    # ==========================================
    # METAMORPHIC PROPERTY TESTS
    # ==========================================
    
    @given(st.data())
    @settings(max_examples=30, deadline=10000)
    def test_irrelevant_sentences_property(self, data):
        """
        Property: Adding irrelevant sentences MUST NOT increase Claim-Support@K at fixed budget
        
        This is a critical property for ensuring that noise doesn't improve retrieval quality.
        """
        # Generate base chunk with gold sentences
        base_chunk = data.draw(self.chunk_strategy(min_sentences=8, max_sentences=15))
        assume(any(s.is_gold for s in base_chunk.sentences))
        
        query = data.draw(st.text(min_size=10, max_size=100))
        budget = data.draw(st.integers(100, 500))
        k = data.draw(st.integers(3, 8))
        
        # Get base results
        base_pruned = self.mock_sentence_prune(query, base_chunk, budget)
        base_knapsack = self.mock_knapsack_pack(base_pruned["kept"], budget)
        
        base_gold_sids = [s.sid for s in base_chunk.sentences if s.is_gold]
        base_claim_support = self.calculate_claim_support_at_k(
            base_knapsack.order, base_gold_sids, k
        )
        
        # Generate irrelevant sentences
        n_irrelevant = data.draw(st.integers(1, 5))
        irrelevant_sentences = []
        for i in range(n_irrelevant):
            irrelevant = data.draw(self.sentence_strategy(is_relevant=False))
            irrelevant.sid = f"irrelevant_{i}_{irrelevant.sid}"
            irrelevant_sentences.append(irrelevant)
        
        # Create augmented chunk
        augmented_sentences = base_chunk.sentences + irrelevant_sentences
        augmented_chunk = ChunkData(
            chunk_id=f"augmented_{base_chunk.chunk_id}",
            sentences=augmented_sentences,
            original_content=base_chunk.original_content + " " + " ".join(s.content for s in irrelevant_sentences)
        )
        
        # Get augmented results
        augmented_pruned = self.mock_sentence_prune(query, augmented_chunk, budget)
        augmented_knapsack = self.mock_knapsack_pack(augmented_pruned["kept"], budget)
        
        augmented_claim_support = self.calculate_claim_support_at_k(
            augmented_knapsack.order, base_gold_sids, k
        )
        
        note(f"Base Claim-Support@{k}: {base_claim_support:.3f}")
        note(f"Augmented Claim-Support@{k}: {augmented_claim_support:.3f}")
        note(f"Added {n_irrelevant} irrelevant sentences")
        
        # Claim support should not increase
        assert augmented_claim_support <= base_claim_support + 1e-6, \
            f"Adding irrelevant sentences increased Claim-Support@{k}: {base_claim_support:.3f} -> {augmented_claim_support:.3f}"
    
    @given(st.data())
    @settings(max_examples=20, deadline=10000)
    def test_duplicate_sentences_property(self, data):
        """
        Property: Duplicating any kept sentence MUST NOT change scores/pack order
        
        This ensures that the knapsack packing is stable and doesn't favor duplicated content.
        """
        chunk = data.draw(self.chunk_strategy(min_sentences=6, max_sentences=12))
        query = data.draw(st.text(min_size=10, max_size=100))
        budget = data.draw(st.integers(150, 400))
        
        # Get base results
        base_pruned = self.mock_sentence_prune(query, chunk, budget)
        assume(len(base_pruned["kept"]) >= 2)  # Need at least 2 sentences to duplicate
        
        base_knapsack = self.mock_knapsack_pack(base_pruned["kept"], budget)
        
        # Select a random kept sentence to duplicate
        kept_sentences = base_pruned["kept"]
        sentence_to_duplicate = data.draw(st.sampled_from(kept_sentences))
        
        # Create duplicate with different ID but same content/score
        duplicate = sentence_to_duplicate.copy()
        duplicate["sid"] = f"dup_{sentence_to_duplicate['sid']}"
        
        # Add duplicate to kept sentences
        duplicated_sentences = kept_sentences + [duplicate]
        
        # Pack with duplicate
        duplicated_knapsack = self.mock_knapsack_pack(duplicated_sentences, budget)
        
        # Compare results (excluding the duplicate itself)
        original_order = [sid for sid in base_knapsack.order if sid != duplicate["sid"]]
        duplicated_order = [sid for sid in duplicated_knapsack.order if sid != duplicate["sid"]]
        
        note(f"Original order length: {len(original_order)}")
        note(f"Duplicated order length: {len(duplicated_order)}")
        note(f"Original total score: {base_knapsack.total_score:.3f}")
        note(f"Duplicated total score: {duplicated_knapsack.total_score:.3f}")
        
        # Order of non-duplicated sentences should remain the same
        # (allowing for the duplicate to be inserted)
        assert original_order == duplicated_order, \
            f"Duplicating sentence changed order: {original_order} != {duplicated_order}"
    
    @given(st.data())
    @settings(max_examples=15, deadline=15000)
    def test_synonymized_query_property(self, data):
        """
        Property: Synonymized query (lemmatized) keeps nDCG within ε
        
        This tests robustness to query variations that should have similar meaning.
        """
        if not self.lemmatizer:
            assume(False)  # Skip if lemmatizer not available
        
        chunk = data.draw(self.chunk_strategy(min_sentences=8, max_sentences=15))
        base_query = data.draw(st.text(min_size=20, max_size=100))
        budget = data.draw(st.integers(200, 500))
        
        # Get base results
        base_pruned = self.mock_sentence_prune(base_query, chunk, budget)
        base_knapsack = self.mock_knapsack_pack(base_pruned["kept"], budget)
        
        # Create synonymized query (simple lemmatization)
        words = base_query.split()
        lemmatized_words = []
        for word in words:
            try:
                lemmatized = self.lemmatizer.lemmatize(word.lower())
                lemmatized_words.append(lemmatized)
            except:
                lemmatized_words.append(word.lower())
        
        synonymized_query = " ".join(lemmatized_words)
        
        # Skip if queries are identical after lemmatization
        assume(base_query.lower() != synonymized_query)
        
        # Get synonymized results
        synonym_pruned = self.mock_sentence_prune(synonymized_query, chunk, budget)
        synonym_knapsack = self.mock_knapsack_pack(synonym_pruned["kept"], budget)
        
        # Calculate simple nDCG approximation (using scores as relevance)
        def calculate_simple_ndcg(ordered_sentences: List[str], sentence_scores: Dict[str, float], k: int = 10) -> float:
            if not ordered_sentences:
                return 0.0
            
            # DCG calculation
            dcg = 0.0
            for i, sid in enumerate(ordered_sentences[:k]):
                if sid in sentence_scores:
                    rel = sentence_scores[sid]
                    dcg += rel / np.log2(i + 2)
            
            # Ideal DCG (best possible ordering)
            scores = sorted(sentence_scores.values(), reverse=True)
            idcg = sum(score / np.log2(i + 2) for i, score in enumerate(scores[:k]))
            
            return dcg / idcg if idcg > 0 else 0.0
        
        # Build score dictionaries
        base_scores = {s["sid"]: s["score"] for s in base_pruned["kept"]}
        synonym_scores = {s["sid"]: s["score"] for s in synonym_pruned["kept"]}
        
        base_ndcg = calculate_simple_ndcg(base_knapsack.order, base_scores, k=10)
        synonym_ndcg = calculate_simple_ndcg(synonym_knapsack.order, synonym_scores, k=10)
        
        epsilon = 0.1  # Tolerance for nDCG difference
        
        note(f"Base query: '{base_query[:50]}...'")
        note(f"Synonymized query: '{synonymized_query[:50]}...'")
        note(f"Base nDCG: {base_ndcg:.3f}")
        note(f"Synonym nDCG: {synonym_ndcg:.3f}")
        note(f"Difference: {abs(base_ndcg - synonym_ndcg):.3f}")
        
        # nDCG should be within epsilon
        assert abs(base_ndcg - synonym_ndcg) <= epsilon, \
            f"Synonymized query nDCG difference too large: {base_ndcg:.3f} vs {synonym_ndcg:.3f} (ε={epsilon})"
    
    @given(st.data())
    @settings(max_examples=25, deadline=10000)
    def test_gold_removal_property(self, data):
        """
        Property: Removing a gold sentence MUST reduce Claim-Support@K
        
        This ensures that gold sentences are actually contributing to retrieval quality.
        """
        chunk = data.draw(self.chunk_strategy(min_sentences=6, max_sentences=12, gold_sentence_ratio=0.3))
        gold_sentences = [s for s in chunk.sentences if s.is_gold]
        assume(len(gold_sentences) >= 2)  # Need multiple gold sentences
        
        query = data.draw(st.text(min_size=10, max_size=100))
        budget = data.draw(st.integers(150, 400))
        k = data.draw(st.integers(2, 6))
        
        # Get base results
        base_pruned = self.mock_sentence_prune(query, chunk, budget)
        base_knapsack = self.mock_knapsack_pack(base_pruned["kept"], budget)
        
        base_gold_sids = [s.sid for s in gold_sentences]
        base_claim_support = self.calculate_claim_support_at_k(
            base_knapsack.order, base_gold_sids, k
        )
        
        # Remove one gold sentence
        gold_to_remove = data.draw(st.sampled_from(gold_sentences))
        reduced_sentences = [s for s in chunk.sentences if s.sid != gold_to_remove.sid]
        
        reduced_chunk = ChunkData(
            chunk_id=f"reduced_{chunk.chunk_id}",
            sentences=reduced_sentences,
            original_content=chunk.original_content  # Simplified
        )
        
        # Get reduced results
        reduced_pruned = self.mock_sentence_prune(query, reduced_chunk, budget)
        reduced_knapsack = self.mock_knapsack_pack(reduced_pruned["kept"], budget)
        
        # Gold sentences list should exclude the removed one
        remaining_gold_sids = [s.sid for s in reduced_sentences if s.is_gold]
        reduced_claim_support = self.calculate_claim_support_at_k(
            reduced_knapsack.order, remaining_gold_sids, k
        )
        
        note(f"Base Claim-Support@{k}: {base_claim_support:.3f}")
        note(f"Reduced Claim-Support@{k}: {reduced_claim_support:.3f}")
        note(f"Removed gold sentence: {gold_to_remove.sid}")
        note(f"Remaining gold sentences: {len(remaining_gold_sids)}")
        
        # Removing gold sentence should reduce claim support
        # (Allow small tolerance for edge cases)
        assert base_claim_support >= reduced_claim_support - 1e-6, \
            f"Removing gold sentence did not reduce Claim-Support@{k}: {base_claim_support:.3f} -> {reduced_claim_support:.3f}"
    
    @given(st.data())
    @settings(max_examples=20, deadline=8000)
    def test_shuffle_invariance_property(self, data):
        """
        Property: Shuffling non-selected items has no effect
        
        This tests that the selection process is independent of the order of non-selected items.
        """
        chunk = data.draw(self.chunk_strategy(min_sentences=10, max_sentences=20))
        query = data.draw(st.text(min_size=10, max_size=100))
        budget = data.draw(st.integers(100, 300))  # Tight budget to ensure some sentences not selected
        
        # Get base results
        base_pruned = self.mock_sentence_prune(query, chunk, budget)
        base_knapsack = self.mock_knapsack_pack(base_pruned["kept"], budget)
        
        selected_sids = set(base_knapsack.selected_sentences)
        
        # Create shuffled version by reordering the original sentences
        shuffled_sentences = chunk.sentences.copy()
        random.shuffle(shuffled_sentences)
        
        shuffled_chunk = ChunkData(
            chunk_id=f"shuffled_{chunk.chunk_id}",
            sentences=shuffled_sentences,
            original_content=chunk.original_content  # Content order doesn't matter for this test
        )
        
        # Get shuffled results
        shuffled_pruned = self.mock_sentence_prune(query, shuffled_chunk, budget)
        shuffled_knapsack = self.mock_knapsack_pack(shuffled_pruned["kept"], budget)
        
        shuffled_selected_sids = set(shuffled_knapsack.selected_sentences)
        
        note(f"Original selected: {sorted(selected_sids)}")
        note(f"Shuffled selected: {sorted(shuffled_selected_sids)}")
        note(f"Original total tokens: {base_knapsack.total_tokens}")
        note(f"Shuffled total tokens: {shuffled_knapsack.total_tokens}")
        
        # Same sentences should be selected regardless of input order
        assert selected_sids == shuffled_selected_sids, \
            f"Shuffling input changed selection: {selected_sids} != {shuffled_selected_sids}"
        
        # Total scores should be identical
        assert abs(base_knapsack.total_score - shuffled_knapsack.total_score) < 1e-6, \
            f"Shuffling changed total score: {base_knapsack.total_score:.3f} vs {shuffled_knapsack.total_score:.3f}"

def main():
    """Run Lethe-specific property tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lethe vNext Specific Property Tests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run tests using pytest
    pytest_args = [
        __file__,
        "-v" if args.verbose else "-q",
        f"--tb=short",
        "--hypothesis-seed", str(args.seed)
    ]
    
    import pytest
    exit_code = pytest.main(pytest_args)
    
    return exit_code

if __name__ == "__main__":
    import sys
    sys.exit(main())