#!/usr/bin/env python3
"""
Entity-Based Diversification Engine

Implements sophisticated diversification using session-IDF weighted entities
with exact identifier matching guarantees and diminishing returns.

Key Features:
- Greedy diversification: f(S) = ∑_e min(1, |S ∩ D_e|)  
- Session-IDF entity weights for importance
- Exact identifier matches guaranteed before diversity
- Token budget enforcement
- Diminishing returns with per-entity contribution caps
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, NamedTuple
import re
import time
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity with metadata for diversification."""
    name: str
    kind: str  # id, file, error, api, tool, person, org, misc
    weight: float  # Session-IDF weight
    doc_ids: Set[str] = field(default_factory=set)  # Documents containing this entity

class DiversificationResult(NamedTuple):
    """Result of diversification process."""
    doc_ids: List[str]
    scores: List[float] 
    entity_coverage: Dict[str, int]  # entity -> count in selected docs
    objective_value: float           # Final f(S) value
    exact_matches_count: int        # Number of exact matches included
    tokens_used: int                # Total tokens in selection
    selection_time_ms: float        # Time taken for selection

@dataclass
class DiversificationConfig:
    """Configuration for entity-based diversification."""
    
    # Budget constraints
    max_tokens: int = 8000          # Maximum tokens in selection
    max_docs: int = 100             # Maximum documents to select
    
    # Entity contribution caps (diminishing returns)
    max_entity_contribution: float = 1.0  # Cap per entity in objective
    
    # Exact match patterns (guaranteed inclusion)
    exact_match_patterns: List[str] = field(default_factory=lambda: [
        r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(',  # Function calls
        r'\b[A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+',  # Class.method
        r'\b[a-zA-Z0-9_]+_[a-zA-Z0-9_]+\b',      # snake_case identifiers
        r'\b[A-Z][a-z]+[A-Z][a-z]*\b',           # CamelCase
    ])
    
    # Token estimation (simple heuristic: ~4 chars per token)
    chars_per_token: float = 4.0
    
    def __post_init__(self):
        """Compile regex patterns for efficiency."""
        self._exact_match_regex = [re.compile(p) for p in self.exact_match_patterns]

class EntityDiversificationEngine:
    """
    Entity-aware diversification engine with exact match guarantees.
    
    Core Algorithm:
    1. Identify exact identifier matches in query -> guarantee inclusion
    2. Build entity-document mapping with session-IDF weights  
    3. Greedy diversification to maximize f(S) = ∑_e min(1, |S ∩ D_e|)
    4. Enforce token budget and diminishing returns
    """
    
    def __init__(self, config: Optional[DiversificationConfig] = None):
        """Initialize diversification engine."""
        self.config = config or DiversificationConfig()
        
        # Entity database
        self.entities: Dict[str, Entity] = {}              # entity_name -> Entity
        self.doc_entities: Dict[str, Set[str]] = {}        # doc_id -> entity_names
        self.doc_tokens: Dict[str, int] = {}               # doc_id -> token_count
        self.doc_text: Dict[str, str] = {}                 # doc_id -> text (for exact matching)
        
        logger.info("EntityDiversificationEngine initialized")
    
    def update_entity_database(
        self,
        doc_id: str,
        text: str,
        entities: List[Tuple[str, str, float]],  # (name, kind, weight)
        token_count: Optional[int] = None
    ):
        """
        Update entity database with document information.
        
        Args:
            doc_id: Document identifier
            text: Full document text
            entities: List of (entity_name, entity_kind, session_idf_weight)
            token_count: Number of tokens (estimated if None)
        """
        # Store document info
        self.doc_text[doc_id] = text
        self.doc_tokens[doc_id] = token_count or self._estimate_tokens(text)
        
        # Update entity mappings
        doc_entity_names = set()
        
        for entity_name, entity_kind, weight in entities:
            # Create or update entity
            if entity_name not in self.entities:
                self.entities[entity_name] = Entity(
                    name=entity_name,
                    kind=entity_kind,
                    weight=weight,
                    doc_ids=set()
                )
            
            # Update entity-document mapping
            self.entities[entity_name].doc_ids.add(doc_id)
            doc_entity_names.add(entity_name)
            
            # Update weight if this is higher (max across sessions)
            if weight > self.entities[entity_name].weight:
                self.entities[entity_name].weight = weight
        
        self.doc_entities[doc_id] = doc_entity_names
        
        logger.debug(f"Updated entity DB: doc {doc_id} has {len(doc_entity_names)} entities")
    
    def diversify_selection(
        self,
        query: str,
        candidate_doc_ids: List[str],
        candidate_scores: List[float],
        enforce_budget: bool = True
    ) -> DiversificationResult:
        """
        Perform entity-based diversification with exact match guarantees.
        
        Algorithm:
        1. Find exact identifier matches -> force include
        2. Greedy diversification on remaining budget
        3. Maximize f(S) = ∑_e min(1, |S ∩ D_e|) weighted by session-IDF
        
        Args:
            query: Original query text
            candidate_doc_ids: Candidate document IDs (ranked)
            candidate_scores: Corresponding relevance scores
            enforce_budget: Whether to enforce token/document budget
            
        Returns:
            DiversificationResult with selected documents and metadata
        """
        start_time = time.time()
        
        if not candidate_doc_ids:
            return DiversificationResult([], [], {}, 0.0, 0, 0, 0.0)
        
        # Phase 1: Identify and guarantee exact matches
        exact_matches = self._find_exact_matches(query, candidate_doc_ids)
        guaranteed_docs = list(exact_matches)
        
        logger.info(f"Found {len(guaranteed_docs)} exact identifier matches")
        
        # Phase 2: Calculate budget remaining after exact matches
        used_tokens = sum(self.doc_tokens.get(doc_id, 0) for doc_id in guaranteed_docs)
        remaining_token_budget = max(0, self.config.max_tokens - used_tokens) if enforce_budget else float('inf')
        remaining_doc_budget = max(0, self.config.max_docs - len(guaranteed_docs)) if enforce_budget else float('inf')
        
        # Phase 3: Greedy diversification on remaining candidates
        remaining_candidates = [
            (doc_id, score) for doc_id, score in zip(candidate_doc_ids, candidate_scores) 
            if doc_id not in exact_matches
        ]
        
        diversified_docs, objective_value = self._greedy_diversification(
            remaining_candidates,
            remaining_token_budget,
            remaining_doc_budget,
            guaranteed_docs
        )
        
        # Phase 4: Combine guaranteed and diversified selections
        final_doc_ids = guaranteed_docs + diversified_docs
        final_scores = []
        
        # Reconstruct scores in final order
        score_map = dict(zip(candidate_doc_ids, candidate_scores))
        for doc_id in final_doc_ids:
            final_scores.append(score_map.get(doc_id, 0.0))
        
        # Phase 5: Compute final statistics
        entity_coverage = self._compute_entity_coverage(final_doc_ids)
        final_objective = self._compute_objective_value(final_doc_ids)
        total_tokens = sum(self.doc_tokens.get(doc_id, 0) for doc_id in final_doc_ids)
        
        selection_time = (time.time() - start_time) * 1000
        
        result = DiversificationResult(
            doc_ids=final_doc_ids,
            scores=final_scores,
            entity_coverage=entity_coverage,
            objective_value=final_objective,
            exact_matches_count=len(guaranteed_docs),
            tokens_used=total_tokens,
            selection_time_ms=selection_time
        )
        
        logger.info(
            f"Diversification complete: {len(final_doc_ids)} docs, "
            f"{len(guaranteed_docs)} exact matches, "
            f"objective={final_objective:.2f}, "
            f"tokens={total_tokens}, "
            f"time={selection_time:.1f}ms"
        )
        
        return result
    
    def _find_exact_matches(self, query: str, candidate_doc_ids: List[str]) -> Set[str]:
        """
        Find documents that contain exact identifier matches from query.
        
        Uses regex patterns to identify code identifiers, function calls, etc.
        and guarantees inclusion of documents containing these patterns.
        """
        exact_matches = set()
        
        # Extract potential identifiers from query
        query_identifiers = set()
        for regex in self.config._exact_match_regex:
            matches = regex.findall(query)
            query_identifiers.update(matches)
        
        if not query_identifiers:
            return exact_matches
        
        logger.debug(f"Extracted {len(query_identifiers)} identifiers from query")
        
        # Find documents containing these identifiers
        for doc_id in candidate_doc_ids:
            if doc_id not in self.doc_text:
                continue
                
            doc_text = self.doc_text[doc_id]
            
            # Check for exact identifier matches in document text
            for identifier in query_identifiers:
                if identifier in doc_text:
                    exact_matches.add(doc_id)
                    logger.debug(f"Exact match: '{identifier}' found in doc {doc_id}")
                    break
        
        return exact_matches
    
    def _greedy_diversification(
        self,
        candidates: List[Tuple[str, float]],  # (doc_id, score)
        token_budget: float,
        doc_budget: float,
        selected_docs: List[str]
    ) -> Tuple[List[str], float]:
        """
        Greedy diversification to maximize weighted entity coverage.
        
        Algorithm:
        - For each candidate, compute marginal gain in objective f(S)
        - Select candidate with highest gain per token (efficiency)
        - Continue until budget exhausted or no positive gains
        """
        diversified = []
        current_selection = set(selected_docs)  # Include pre-selected docs
        remaining_tokens = token_budget
        remaining_docs = doc_budget
        
        # Pre-compute entity coverage from guaranteed docs
        current_coverage = defaultdict(int)
        for doc_id in selected_docs:
            if doc_id in self.doc_entities:
                for entity_name in self.doc_entities[doc_id]:
                    current_coverage[entity_name] += 1
        
        while candidates and remaining_docs > 0 and remaining_tokens > 0:
            best_candidate = None
            best_gain_per_token = 0.0
            best_gain = 0.0
            
            for i, (doc_id, score) in enumerate(candidates):
                # Skip if already selected or budget exceeded
                doc_tokens = self.doc_tokens.get(doc_id, 0)
                if doc_id in current_selection or doc_tokens > remaining_tokens:
                    continue
                
                # Compute marginal gain in objective function
                marginal_gain = self._compute_marginal_gain(doc_id, current_coverage)
                
                if marginal_gain > 0:
                    # Efficiency: gain per token (with relevance boost)
                    relevance_boost = 1.0 + score  # Slight preference for higher scored docs
                    gain_per_token = (marginal_gain * relevance_boost) / max(doc_tokens, 1)
                    
                    if gain_per_token > best_gain_per_token:
                        best_candidate = (i, doc_id)
                        best_gain_per_token = gain_per_token
                        best_gain = marginal_gain
            
            # No positive gains left
            if best_candidate is None:
                break
            
            # Select best candidate
            best_idx, best_doc_id = best_candidate
            diversified.append(best_doc_id)
            current_selection.add(best_doc_id)
            
            # Update coverage and budgets
            if best_doc_id in self.doc_entities:
                for entity_name in self.doc_entities[best_doc_id]:
                    current_coverage[entity_name] += 1
            
            remaining_tokens -= self.doc_tokens.get(best_doc_id, 0)
            remaining_docs -= 1
            
            # Remove selected candidate
            candidates.pop(best_idx)
            
            logger.debug(f"Selected doc {best_doc_id}: gain={best_gain:.3f}, efficiency={best_gain_per_token:.3f}")
        
        final_objective = self._compute_objective_value(list(current_selection))
        
        return diversified, final_objective
    
    def _compute_marginal_gain(
        self,
        doc_id: str,
        current_coverage: Dict[str, int]
    ) -> float:
        """
        Compute marginal gain in objective function from adding doc_id.
        
        f(S) = ∑_e w_e * min(1, |S ∩ D_e|)
        
        Marginal gain accounts for diminishing returns - each entity
        contributes at most 1.0 to the objective regardless of frequency.
        """
        if doc_id not in self.doc_entities:
            return 0.0
        
        marginal_gain = 0.0
        
        for entity_name in self.doc_entities[doc_id]:
            if entity_name not in self.entities:
                continue
                
            entity = self.entities[entity_name]
            current_count = current_coverage.get(entity_name, 0)
            
            # Diminishing returns: min(1, count) contribution
            current_contribution = min(current_count, 1)
            new_contribution = min(current_count + 1, 1)
            
            # Weighted marginal gain
            gain = (new_contribution - current_contribution) * entity.weight
            marginal_gain += gain
        
        return marginal_gain
    
    def _compute_objective_value(self, selected_docs: List[str]) -> float:
        """
        Compute total objective value f(S) = ∑_e w_e * min(1, |S ∩ D_e|).
        """
        entity_counts = defaultdict(int)
        
        # Count entity occurrences in selected docs
        for doc_id in selected_docs:
            if doc_id in self.doc_entities:
                for entity_name in self.doc_entities[doc_id]:
                    entity_counts[entity_name] += 1
        
        # Compute weighted objective with diminishing returns
        objective_value = 0.0
        for entity_name, count in entity_counts.items():
            if entity_name in self.entities:
                weight = self.entities[entity_name].weight
                contribution = min(count, self.config.max_entity_contribution)
                objective_value += weight * contribution
        
        return objective_value
    
    def _compute_entity_coverage(self, selected_docs: List[str]) -> Dict[str, int]:
        """Compute entity coverage statistics for selected documents."""
        entity_coverage = defaultdict(int)
        
        for doc_id in selected_docs:
            if doc_id in self.doc_entities:
                for entity_name in self.doc_entities[doc_id]:
                    entity_coverage[entity_name] += 1
        
        return dict(entity_coverage)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return int(len(text) / self.config.chars_per_token)
    
    def get_entity_stats(self) -> Dict:
        """Get entity database statistics."""
        return {
            'total_entities': len(self.entities),
            'total_documents': len(self.doc_entities),
            'avg_entities_per_doc': sum(len(entities) for entities in self.doc_entities.values()) / max(len(self.doc_entities), 1),
            'entity_kinds': {
                kind: sum(1 for e in self.entities.values() if e.kind == kind)
                for kind in set(e.kind for e in self.entities.values())
            }
        }


def create_diversification_engine(config: Optional[DiversificationConfig] = None) -> EntityDiversificationEngine:
    """Create entity diversification engine with optional configuration."""
    return EntityDiversificationEngine(config)