#!/usr/bin/env python3
"""
Adaptive Planning Policy for Agent-Context Manager

Implements calibrated rule-based policy for determining retrieval strategy
based on query characteristics and agent conversation history.

Key Features:
- VERIFY/EXPLORE/EXPLOIT strategy mapping  
- Agent-aware feature extraction
- Configurable thresholds with sensible defaults
- Session-aware entity overlap computation
- Tool overlap detection for agent context
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)

class PlanningStrategy(Enum):
    """Agent-aware retrieval strategies."""
    VERIFY = "verify"      # High precision, exact matches (α=0.7)
    EXPLORE = "explore"    # High recall, broad search (α=0.3) 
    EXPLOIT = "exploit"    # Balanced precision/recall (α=0.5)

@dataclass
class PlanningConfiguration:
    """Configuration for adaptive planning policy."""
    
    # Core threshold parameters
    tau_verify_idf: float = 8.0      # High IDF threshold for VERIFY
    tau_entity_overlap: float = 0.3   # Entity overlap threshold
    tau_novelty: float = 0.1          # Novelty threshold for EXPLORE
    
    # Alpha values for fusion weights
    alpha_verify: float = 0.7         # BM25-heavy for exact matches
    alpha_explore: float = 0.3        # Vector-heavy for exploration
    alpha_exploit: float = 0.5        # Balanced fusion
    
    # ANN search parameters by strategy
    ef_search_verify: int = 50        # Lower recall, faster
    ef_search_explore: int = 200      # Higher recall, broader
    ef_search_exploit: int = 100      # Balanced
    
    # History window for context
    history_window: int = 10          # Recent turns to consider
    
    # Feature extraction patterns
    code_patterns: List[str] = field(default_factory=lambda: [
        r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(',  # Function calls
        r'\b[A-Z][a-zA-Z0-9_]*\.',         # Class methods  
        r'\[[^\]]+\]',                      # Array access
        r'\{[^}]+\}',                       # Object literals
    ])
    
    error_patterns: List[str] = field(default_factory=lambda: [
        r'Error\b',
        r'Exception\b', 
        r'Failed\b',
        r'ERROR:\s',
        r'TypeError\b',
        r'ValueError\b'
    ])
    
    identifier_patterns: List[str] = field(default_factory=lambda: [
        r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b',  # Variable names
        r'[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+',   # Dotted names
        r'[A-Z][a-z]+[A-Z][a-z]*',        # CamelCase
    ])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.alpha_verify <= 1.0):
            raise ValueError(f"alpha_verify must be in [0,1], got {self.alpha_verify}")
        if not (0.0 <= self.alpha_explore <= 1.0):
            raise ValueError(f"alpha_explore must be in [0,1], got {self.alpha_explore}")
        if not (0.0 <= self.alpha_exploit <= 1.0):
            raise ValueError(f"alpha_exploit must be in [0,1], got {self.alpha_exploit}")
            
        # Compile regex patterns for efficiency
        self._code_regex = [re.compile(p, re.IGNORECASE) for p in self.code_patterns]
        self._error_regex = [re.compile(p, re.IGNORECASE) for p in self.error_patterns]
        self._identifier_regex = [re.compile(p) for p in self.identifier_patterns]

@dataclass 
class QueryFeatures:
    """Extracted features from query for planning decision."""
    
    # IDF features
    max_idf: float
    mean_idf: float
    
    # Length features  
    query_length: int
    token_count: int
    
    # Entity overlap with history
    entity_overlap_jaccard: float
    entity_overlap_count: int
    
    # Pattern matching flags
    has_code: bool
    has_error: bool  
    has_identifier: bool
    
    # Tool context
    tool_overlap: bool
    recent_tool_mentions: int
    
    # Session context
    session_id: str
    turn_idx: int

@dataclass
class PlanningResult:
    """Result of adaptive planning decision."""
    
    strategy: PlanningStrategy
    alpha: float              # Fusion weight for BM25 vs vector
    ef_search: int           # ANN search parameter
    features: QueryFeatures  # Input features used
    confidence: float        # Decision confidence [0,1]
    reasoning: str          # Human-readable explanation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'strategy': self.strategy.value,
            'alpha': self.alpha,
            'ef_search': self.ef_search,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'features': {
                'max_idf': self.features.max_idf,
                'entity_overlap': self.features.entity_overlap_jaccard,
                'has_code': self.features.has_code,
                'has_error': self.features.has_error,
                'tool_overlap': self.features.tool_overlap
            }
        }

class AdaptivePlanningEngine:
    """
    Agent-aware adaptive planning engine.
    
    Determines optimal retrieval strategy based on query characteristics
    and conversation history using calibrated rule-based policy.
    """
    
    def __init__(self, config: Optional[PlanningConfiguration] = None):
        """Initialize planning engine."""
        self.config = config or PlanningConfiguration()
        
        # Known tools cache for overlap detection
        self.known_tools: Set[str] = set()
        self.entity_history: Dict[str, Set[str]] = {}  # session_id -> entities
        self.tool_history: Dict[str, Set[str]] = {}    # session_id -> tools
        
        # Decision logging
        self.decision_log: List[Dict] = []
        
        logger.info("AdaptivePlanningEngine initialized")
    
    def update_session_context(
        self,
        session_id: str,
        entities: Set[str],
        tools: Set[str]
    ):
        """Update session context with new entities and tools."""
        if session_id not in self.entity_history:
            self.entity_history[session_id] = set()
        if session_id not in self.tool_history:
            self.tool_history[session_id] = set()
            
        self.entity_history[session_id].update(entities)
        self.tool_history[session_id].update(tools)
        self.known_tools.update(tools)
        
        logger.debug(f"Updated session {session_id}: {len(entities)} entities, {len(tools)} tools")
    
    def extract_features(
        self,
        query: str,
        session_id: str,
        turn_idx: int,
        term_idfs: Optional[Dict[str, float]] = None
    ) -> QueryFeatures:
        """
        Extract features from query for planning decision.
        
        Args:
            query: Input query text
            session_id: Session identifier  
            turn_idx: Turn index in session
            term_idfs: Pre-computed IDF scores for query terms
            
        Returns:
            Extracted query features
        """
        # Tokenize query (simple whitespace tokenization)
        tokens = query.lower().split()
        
        # IDF features
        if term_idfs:
            idf_values = [term_idfs.get(token, 0.0) for token in tokens]
            max_idf = max(idf_values) if idf_values else 0.0
            mean_idf = sum(idf_values) / len(idf_values) if idf_values else 0.0
        else:
            max_idf = mean_idf = 0.0
        
        # Length features
        query_length = len(query)
        token_count = len(tokens)
        
        # Extract entities (simplified - would use proper NER in production)
        query_entities = self._extract_entities(query)
        
        # Entity overlap with session history
        session_entities = self.entity_history.get(session_id, set())
        if session_entities and query_entities:
            intersection = len(query_entities & session_entities)
            union = len(query_entities | session_entities)
            entity_overlap_jaccard = intersection / union if union > 0 else 0.0
            entity_overlap_count = intersection
        else:
            entity_overlap_jaccard = 0.0
            entity_overlap_count = 0
        
        # Pattern matching
        has_code = any(regex.search(query) for regex in self.config._code_regex)
        has_error = any(regex.search(query) for regex in self.config._error_regex)  
        has_identifier = any(regex.search(query) for regex in self.config._identifier_regex)
        
        # Tool overlap detection
        session_tools = self.tool_history.get(session_id, set())
        tool_mentions = []
        for tool in self.known_tools:
            if tool.lower() in query.lower():
                tool_mentions.append(tool)
        
        tool_overlap = len(tool_mentions) > 0
        recent_tool_mentions = len([t for t in tool_mentions if t in session_tools])
        
        return QueryFeatures(
            max_idf=max_idf,
            mean_idf=mean_idf,
            query_length=query_length,
            token_count=token_count,
            entity_overlap_jaccard=entity_overlap_jaccard,
            entity_overlap_count=entity_overlap_count,
            has_code=has_code,
            has_error=has_error,
            has_identifier=has_identifier,
            tool_overlap=tool_overlap,
            recent_tool_mentions=recent_tool_mentions,
            session_id=session_id,
            turn_idx=turn_idx
        )
    
    def plan_retrieval(
        self,
        query: str,
        session_id: str,
        turn_idx: int,
        term_idfs: Optional[Dict[str, float]] = None
    ) -> PlanningResult:
        """
        Determine optimal retrieval strategy using calibrated rules.
        
        Core Logic:
        - VERIFY: High precision for exact matches (high IDF + entity overlap + identifiers)
        - EXPLORE: High recall for novel queries (low entity overlap + no tool context)  
        - EXPLOIT: Balanced approach for everything else
        
        Args:
            query: Input query text
            session_id: Session identifier
            turn_idx: Turn index in session  
            term_idfs: Pre-computed IDF scores
            
        Returns:
            Planning result with strategy and parameters
        """
        # Extract query features
        features = self.extract_features(query, session_id, turn_idx, term_idfs)
        
        # Apply calibrated decision rules
        strategy, confidence, reasoning = self._apply_decision_rules(features)
        
        # Map strategy to parameters
        if strategy == PlanningStrategy.VERIFY:
            alpha = self.config.alpha_verify
            ef_search = self.config.ef_search_verify
        elif strategy == PlanningStrategy.EXPLORE:
            alpha = self.config.alpha_explore  
            ef_search = self.config.ef_search_explore
        else:  # EXPLOIT
            alpha = self.config.alpha_exploit
            ef_search = self.config.ef_search_exploit
        
        result = PlanningResult(
            strategy=strategy,
            alpha=alpha,
            ef_search=ef_search,
            features=features,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Log decision for analysis
        self.decision_log.append(result.to_dict())
        
        logger.info(
            f"Planning decision: {strategy.value} (α={alpha:.2f}, ef={ef_search}) "
            f"- {reasoning}"
        )
        
        return result
    
    def _apply_decision_rules(
        self,
        features: QueryFeatures
    ) -> Tuple[PlanningStrategy, float, str]:
        """
        Apply calibrated decision rules to determine strategy.
        
        Rules (in priority order):
        1. VERIFY: high IDF + entity overlap + identifiers + NOT error-only  
        2. EXPLORE: low entity overlap OR no tool overlap
        3. EXPLOIT: default case
        
        Returns:
            (strategy, confidence, reasoning)
        """
        # Rule 1: VERIFY conditions
        verify_conditions = [
            features.max_idf > self.config.tau_verify_idf,
            features.entity_overlap_jaccard > self.config.tau_entity_overlap,
            features.has_identifier,
            not (features.has_error and not features.has_code)  # Not error-only
        ]
        
        if sum(verify_conditions) >= 3:  # At least 3/4 conditions
            confidence = sum(verify_conditions) / 4.0
            reasoning = f"High precision needed: IDF={features.max_idf:.1f}, overlap={features.entity_overlap_jaccard:.2f}, has_id={features.has_identifier}"
            return PlanningStrategy.VERIFY, confidence, reasoning
        
        # Rule 2: EXPLORE conditions  
        explore_conditions = [
            features.entity_overlap_jaccard < self.config.tau_novelty,
            not features.tool_overlap,
            features.recent_tool_mentions == 0
        ]
        
        if any(explore_conditions):
            confidence = 0.8 if sum(explore_conditions) >= 2 else 0.6
            reasoning = f"Novel query needs exploration: entity_overlap={features.entity_overlap_jaccard:.2f}, tool_overlap={features.tool_overlap}"
            return PlanningStrategy.EXPLORE, confidence, reasoning
        
        # Rule 3: EXPLOIT (default)
        confidence = 0.7
        reasoning = f"Balanced approach: moderate overlap={features.entity_overlap_jaccard:.2f}, has_tools={features.tool_overlap}"
        return PlanningStrategy.EXPLOIT, confidence, reasoning
    
    def _extract_entities(self, query: str) -> Set[str]:
        """
        Extract entities from query using simple heuristics.
        
        In production, this would use a proper NER model.
        For now, use identifier patterns and known tokens.
        """
        entities = set()
        
        # Extract identifier-like tokens
        for regex in self.config._identifier_regex:
            matches = regex.findall(query)
            entities.update(matches)
        
        # Add capitalized words (potential proper nouns)
        tokens = query.split()
        for token in tokens:
            if token[0].isupper() and len(token) > 2:
                entities.add(token.lower())
        
        return entities
    
    def get_decision_log(self) -> List[Dict]:
        """Get decision log for analysis."""
        return self.decision_log.copy()
    
    def clear_decision_log(self):
        """Clear decision log."""
        self.decision_log.clear()
        logger.info("Decision log cleared")


def create_planning_engine(config: Optional[PlanningConfiguration] = None) -> AdaptivePlanningEngine:
    """Create adaptive planning engine with optional configuration."""
    return AdaptivePlanningEngine(config)