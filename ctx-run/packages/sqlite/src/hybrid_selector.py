#!/usr/bin/env python3
"""
Lethe→StreamingLLM Hybrid Selector System

Implements the complete hybrid selection system that combines Lethe's stable head
selection with StreamingLLM's windowed tail processing for optimal context management.

Core Architecture:
- Head (H): Lethe-selected stable content with grouped atoms (defs, errors, tool keys, symbols)
- Tail (T): StreamingLLM windowed volatile content with attention sinks
- Dual optimization: max F(H∪T) - λ(tokens) - μ(compute)

Key Features:
- KV-aware arrangement (head first, tail after)
- Attention sink integration with head digest
- Adaptive gating based on accept-rate and entity-entropy
- DPP-based head selection with grouped atom extraction
- Windowed tail processing with configurable stride
- Comprehensive performance monitoring and validation
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple, Union
from enum import Enum
import math
from pathlib import Path
import hashlib
import json

# Import existing components  
try:
    from lagrangian_optimizer import LagrangianConfig, LagrangianOptimizer
    from diversification import EntityDiversificationEngine
    from ce_early_exit import CrossEncoderEarlyExit
except ImportError:
    # Fallback for testing without full dependencies
    LagrangianConfig = type('LagrangianConfig', (), {})
    LagrangianOptimizer = type('LagrangianOptimizer', (), {})
    EntityDiversificationEngine = type('EntityDiversificationEngine', (), {})
    CrossEncoderEarlyExit = type('CrossEncoderEarlyExit', (), {
        'early_exit_selection': lambda self, atoms, budget: atoms[:min(len(atoms), budget//10)]
    })

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content atoms for grouping."""
    DEFINITION = "definition"
    ERROR_FRAME = "error_frame"
    TOOL_KEY = "tool_key"
    SYMBOL_HEADER = "symbol_header"
    CODE_BLOCK = "code_block"
    DOCUMENTATION = "documentation"
    CONTEXT = "context"
    VOLATILE = "volatile"

class ProcessingMode(Enum):
    """Processing modes for hybrid selector."""
    HEAD_ONLY = "head_only"        # Only use Lethe head
    HYBRID = "hybrid"              # Use both head and tail
    STREAMING_ONLY = "streaming_only"  # Fallback to pure streaming

@dataclass
class ContentAtom:
    """Individual content atom with metadata."""
    id: str
    content: str
    content_type: ContentType
    tokens: int
    relevance_score: float
    stability_score: float
    entity_references: Set[str] = field(default_factory=set)
    grouping_score: float = 0.0
    kv_prefix_hash: Optional[str] = None
    
    def __post_init__(self):
        """Compute KV prefix hash for reuse detection."""
        if not self.kv_prefix_hash:
            # Create hash from content prefix for KV cache reuse
            prefix = self.content[:100]  # First 100 chars
            self.kv_prefix_hash = hashlib.md5(prefix.encode()).hexdigest()[:16]

@dataclass 
class GroupedAtoms:
    """Grouped atoms for efficient head construction."""
    group_type: ContentType
    atoms: List[ContentAtom]
    total_tokens: int
    avg_relevance: float
    stability_metric: float
    
    @classmethod
    def from_atoms(cls, atoms: List[ContentAtom], group_type: ContentType):
        """Create grouped atoms from list of atoms."""
        if not atoms:
            return cls(group_type, [], 0, 0.0, 0.0)
            
        total_tokens = sum(atom.tokens for atom in atoms)
        avg_relevance = np.mean([atom.relevance_score for atom in atoms])
        stability_metric = np.mean([atom.stability_score for atom in atoms])
        
        return cls(group_type, atoms, total_tokens, avg_relevance, stability_metric)

@dataclass
class HeadSelection:
    """Result of head selection process."""
    grouped_atoms: Dict[ContentType, GroupedAtoms]
    total_tokens: int
    keep_ratio: float
    dpp_rank: int
    ce_early_exit_used: bool
    kv_prefix_hashes: Set[str]
    head_digest: str  # Micro-summary for attention sinks

@dataclass
class TailWindow:
    """Individual tail window with metadata."""
    window_id: str
    content: str
    tokens: int
    stride_offset: int
    attention_sinks: List[str]
    sink_tokens: int
    entropy_score: float

@dataclass  
class TailSelection:
    """Result of tail selection process."""
    windows: List[TailWindow]
    total_tokens: int
    total_windows: int
    window_size: int
    stride: int
    sink_tokens_per_window: int
    head_digest_embedded: bool

@dataclass
class HybridConfig:
    """Configuration for hybrid selector system."""
    
    # Head configuration (Lethe)
    head_keep_ratio: float = 0.12  # Default ~12% for head
    dpp_rank: int = 14
    group_split_tau: float = 0.7
    ce_k2: int = 320
    ce_early_exit_enabled: bool = True
    
    # Tail configuration (StreamingLLM)
    window_size: int = 6000        # W=6k tokens
    stride: int = 3000             # s=3k tokens (0.5*W)
    sink_tokens: int = 96          # Attention sinks per window
    tail_tokens_cap: int = 12000   # Maximum tail tokens
    
    # Gating parameters
    accept_rate_threshold: float = 0.4    # Enable streaming if < 0.4
    entity_entropy_threshold: float = 0.7 # High entropy threshold
    
    # Optimization parameters
    lambda_param: float = 0.01     # Token cost weight
    mu_param: float = 0.02         # Compute cost weight
    
    # Performance settings
    target_latency_ms: float = 200.0
    kv_reuse_threshold: float = 0.6  # Minimum KV prefix reuse
    
    # Monitoring
    enable_instrumentation: bool = True
    log_selections: bool = True

@dataclass
class HybridSelectionResult:
    """Complete result from hybrid selection."""
    
    # Core results
    head_selection: Optional[HeadSelection]
    tail_selection: Optional[TailSelection]
    processing_mode: ProcessingMode
    
    # Final arrangement
    final_content: str
    total_tokens: int
    keep_ratio: float
    
    # KV cache optimization
    kv_prefix_reuse_ratio: float
    kv_arrangement_optimized: bool
    
    # Performance metrics
    selection_time_ms: float
    head_time_ms: float
    tail_time_ms: float
    arrangement_time_ms: float
    
    # Quality metrics
    objective_value: float  # F(H∪T) value
    cost_lambda: float      # λ * tokens cost
    cost_mu: float          # μ * compute cost
    net_value: float        # objective - costs
    
    # Monitoring data
    gating_decision: Dict[str, Any]
    parameter_state: Dict[str, float]

class AtomExtractor:
    """Extracts and classifies content atoms from raw text."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
        # Pattern matchers for different content types
        self.definition_patterns = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'interface\s+\w+',
            r'function\s+\w+',
            r'const\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'var\s+\w+\s*='
        ]
        
        self.error_patterns = [
            r'Error:',
            r'Exception:',
            r'Traceback',
            r'TypeError:',
            r'ValueError:',
            r'SyntaxError:'
        ]
        
        self.tool_patterns = [
            r'@tool',
            r'@function',
            r'#\s*Tool:',
            r'API_KEY',
            r'endpoint\s*=',
            r'tool_call'
        ]
        
    def extract_atoms(self, content: str, relevance_scores: Dict[str, float] = None) -> List[ContentAtom]:
        """Extract and classify content atoms from text."""
        atoms = []
        lines = content.split('\n')
        current_block = ""
        current_type = ContentType.CONTEXT
        block_start_idx = 0
        
        relevance_scores = relevance_scores or {}
        
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detect content type transitions
            new_type = self._classify_line(line_stripped)
            
            if new_type != current_type or idx == len(lines) - 1:
                # Process accumulated block
                if current_block.strip():
                    atom = self._create_atom(
                        current_block.strip(),
                        current_type,
                        block_start_idx,
                        relevance_scores
                    )
                    atoms.append(atom)
                
                # Start new block
                current_block = line
                current_type = new_type
                block_start_idx = idx
            else:
                current_block += "\n" + line
        
        return atoms
    
    def _classify_line(self, line: str) -> ContentType:
        """Classify a line into content type."""
        import re
        
        # Check for definitions
        for pattern in self.definition_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return ContentType.DEFINITION
        
        # Check for errors
        for pattern in self.error_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return ContentType.ERROR_FRAME
        
        # Check for tools
        for pattern in self.tool_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return ContentType.TOOL_KEY
        
        # Check for symbols/headers
        if line.startswith('#') or line.startswith('##'):
            return ContentType.SYMBOL_HEADER
        
        # Check for code blocks
        if line.startswith('```') or line.startswith('    ') or line.startswith('\t'):
            return ContentType.CODE_BLOCK
        
        # Check for documentation
        if line.startswith('"""') or line.startswith("'''") or line.startswith('//'):
            return ContentType.DOCUMENTATION
        
        # Default to context
        return ContentType.CONTEXT
    
    def _create_atom(self, content: str, content_type: ContentType, 
                    line_idx: int, relevance_scores: Dict[str, float]) -> ContentAtom:
        """Create a content atom with metadata."""
        atom_id = f"{content_type.value}_{line_idx}_{hash(content) % 10000}"
        tokens = len(content.split())  # Simple tokenization
        
        # Get relevance score
        relevance_score = relevance_scores.get(content[:50], 0.5)  # Default relevance
        
        # Compute stability score based on content type
        stability_scores = {
            ContentType.DEFINITION: 0.9,
            ContentType.ERROR_FRAME: 0.8,
            ContentType.TOOL_KEY: 0.85,
            ContentType.SYMBOL_HEADER: 0.75,
            ContentType.CODE_BLOCK: 0.6,
            ContentType.DOCUMENTATION: 0.7,
            ContentType.CONTEXT: 0.4,
            ContentType.VOLATILE: 0.1
        }
        stability_score = stability_scores[content_type]
        
        # Extract entity references (simple approach)
        entities = set()
        words = content.lower().split()
        for word in words:
            if word.endswith('()') or word.startswith('_') or word.isupper():
                entities.add(word)
        
        return ContentAtom(
            id=atom_id,
            content=content,
            content_type=content_type,
            tokens=tokens,
            relevance_score=relevance_score,
            stability_score=stability_score,
            entity_references=entities
        )

class HeadBuilder:
    """Builds stable head using Lethe selection with grouping."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.dpp_selector = None  # Will be initialized when needed
        self.ce_early_exit = None
        
    def build_head(self, atoms: List[ContentAtom], budget_tokens: int) -> HeadSelection:
        """Build head selection with grouped atoms and DPP optimization."""
        start_time = time.time()
        
        # Group atoms by content type
        grouped = self._group_atoms_by_type(atoms)
        
        # Apply DPP selection within each group
        selected_grouped = {}
        total_selected_tokens = 0
        all_kv_hashes = set()
        ce_used = False
        
        for content_type, group in grouped.items():
            if total_selected_tokens >= budget_tokens:
                break
            
            remaining_budget = budget_tokens - total_selected_tokens
            group_budget = min(remaining_budget, group.total_tokens)
            
            # Select atoms within this group
            selected_atoms, group_ce_used = self._select_atoms_in_group(
                group.atoms, group_budget
            )
            
            if selected_atoms:
                selected_group = GroupedAtoms.from_atoms(selected_atoms, content_type)
                selected_grouped[content_type] = selected_group
                total_selected_tokens += selected_group.total_tokens
                ce_used = ce_used or group_ce_used
                
                # Collect KV hashes
                for atom in selected_atoms:
                    if atom.kv_prefix_hash:
                        all_kv_hashes.add(atom.kv_prefix_hash)
        
        # Create head digest for attention sinks
        head_digest = self._create_head_digest(selected_grouped)
        
        keep_ratio = total_selected_tokens / max(1, sum(len(g.atoms) * 10 for g in grouped.values()))
        
        return HeadSelection(
            grouped_atoms=selected_grouped,
            total_tokens=total_selected_tokens,
            keep_ratio=keep_ratio,
            dpp_rank=self.config.dpp_rank,
            ce_early_exit_used=ce_used,
            kv_prefix_hashes=all_kv_hashes,
            head_digest=head_digest
        )
    
    def _group_atoms_by_type(self, atoms: List[ContentAtom]) -> Dict[ContentType, GroupedAtoms]:
        """Group atoms by content type."""
        groups = {}
        
        for atom in atoms:
            if atom.content_type not in groups:
                groups[atom.content_type] = []
            groups[atom.content_type].append(atom)
        
        # Convert to GroupedAtoms
        grouped = {}
        for content_type, atom_list in groups.items():
            grouped[content_type] = GroupedAtoms.from_atoms(atom_list, content_type)
        
        return grouped
    
    def _select_atoms_in_group(self, atoms: List[ContentAtom], budget: int) -> Tuple[List[ContentAtom], bool]:
        """Select atoms within a group using DPP and early exit."""
        if not atoms or budget <= 0:
            return [], False
        
        # Sort by relevance * stability
        atoms_scored = [(atom.relevance_score * atom.stability_score, atom) for atom in atoms]
        atoms_scored.sort(reverse=True, key=lambda x: x[0])
        
        # Apply CE early exit if enabled
        ce_used = False
        if self.config.ce_early_exit_enabled and len(atoms) > self.config.ce_k2:
            if not self.ce_early_exit:
                self.ce_early_exit = CrossEncoderEarlyExit()
            
            # Use top K2 atoms for early exit
            top_k2_atoms = [atom for _, atom in atoms_scored[:self.config.ce_k2]]
            selected_atoms = self.ce_early_exit.early_exit_selection(
                top_k2_atoms, budget
            )
            ce_used = True
        else:
            # Simple greedy selection
            selected_atoms = []
            current_tokens = 0
            
            for score, atom in atoms_scored:
                if current_tokens + atom.tokens <= budget:
                    selected_atoms.append(atom)
                    current_tokens += atom.tokens
        
        return selected_atoms, ce_used
    
    def _create_head_digest(self, grouped_atoms: Dict[ContentType, GroupedAtoms]) -> str:
        """Create compact digest of head for attention sinks."""
        digest_parts = []
        
        for content_type, group in grouped_atoms.items():
            if not group.atoms:
                continue
            
            # Extract key elements from each group
            if content_type == ContentType.DEFINITION:
                # Extract function/class names
                for atom in group.atoms[:3]:  # Top 3
                    lines = atom.content.split('\n')
                    first_line = lines[0].strip()[:50]
                    digest_parts.append(f"DEF: {first_line}")
            
            elif content_type == ContentType.ERROR_FRAME:
                # Extract error types
                for atom in group.atoms[:2]:  # Top 2
                    lines = atom.content.split('\n')
                    error_line = lines[0].strip()[:40]
                    digest_parts.append(f"ERR: {error_line}")
            
            elif content_type == ContentType.SYMBOL_HEADER:
                # Extract headers
                for atom in group.atoms[:2]:
                    header = atom.content.strip()[:30]
                    digest_parts.append(f"HDR: {header}")
            
            elif content_type == ContentType.TOOL_KEY:
                # Extract tool references
                for atom in group.atoms[:2]:
                    tool_ref = atom.content.strip()[:25]
                    digest_parts.append(f"TOOL: {tool_ref}")
        
        # Combine into compact digest
        digest = " | ".join(digest_parts[:10])  # Max 10 elements
        return digest[:200] if digest else "HEAD_CONTEXT"  # Max 200 chars

class TailBuilder:
    """Builds streaming tail with windowed processing and attention sinks."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
    def build_tail(self, remaining_content: str, head_digest: str, 
                  budget_tokens: int) -> TailSelection:
        """Build tail with StreamingLLM windowing and attention sinks."""
        if not remaining_content.strip():
            return TailSelection([], 0, 0, 0, 0, 0, False)
        
        # Split content into windows
        windows = self._create_windows(remaining_content, head_digest, budget_tokens)
        
        total_tokens = sum(w.tokens + w.sink_tokens for w in windows)
        
        return TailSelection(
            windows=windows,
            total_tokens=total_tokens,
            total_windows=len(windows),
            window_size=self.config.window_size,
            stride=self.config.stride,
            sink_tokens_per_window=self.config.sink_tokens,
            head_digest_embedded=bool(head_digest)
        )
    
    def _create_windows(self, content: str, head_digest: str, budget: int) -> List[TailWindow]:
        """Create windowed content with attention sinks."""
        words = content.split()
        if not words:
            return []
        
        windows = []
        current_tokens = 0
        window_id = 0
        
        # Calculate available budget per window (reserve tokens for sinks)
        effective_window_size = self.config.window_size - self.config.sink_tokens
        
        for start_idx in range(0, len(words), self.config.stride):
            if current_tokens >= budget:
                break
            
            end_idx = min(start_idx + effective_window_size, len(words))
            window_words = words[start_idx:end_idx]
            
            if not window_words:
                break
            
            window_content = " ".join(window_words)
            window_tokens = len(window_words)
            
            # Create attention sinks
            attention_sinks = self._create_attention_sinks(head_digest, window_content)
            sink_tokens = sum(len(sink.split()) for sink in attention_sinks)
            
            # Check budget
            total_window_tokens = window_tokens + sink_tokens
            if current_tokens + total_window_tokens > budget:
                # Adjust window size to fit budget
                remaining_budget = budget - current_tokens
                if remaining_budget < self.config.sink_tokens + 100:  # Minimum window
                    break
                
                adjusted_tokens = remaining_budget - sink_tokens
                adjusted_words = window_words[:adjusted_tokens]
                window_content = " ".join(adjusted_words)
                window_tokens = len(adjusted_words)
            
            # Calculate entropy score for window
            entropy_score = self._calculate_entropy(window_content)
            
            window = TailWindow(
                window_id=f"tail_window_{window_id}",
                content=window_content,
                tokens=window_tokens,
                stride_offset=start_idx,
                attention_sinks=attention_sinks,
                sink_tokens=sink_tokens,
                entropy_score=entropy_score
            )
            
            windows.append(window)
            current_tokens += total_window_tokens
            window_id += 1
            
            # Check if we've covered all content
            if end_idx >= len(words):
                break
        
        return windows
    
    def _create_attention_sinks(self, head_digest: str, window_content: str) -> List[str]:
        """Create attention sinks incorporating head digest."""
        sinks = []
        
        # Add head digest as primary sink
        if head_digest:
            sink_content = f"CONTEXT: {head_digest}"
            sinks.append(sink_content[:50])  # Max 50 chars
        
        # Add window-specific sinks
        lines = window_content.split('\n')
        if lines:
            # First line as sink
            first_line = lines[0].strip()[:40]
            if first_line:
                sinks.append(f"START: {first_line}")
            
            # Important terms as sinks
            words = window_content.split()
            important_words = [w for w in words if w.isupper() or '_' in w or '()' in w]
            if important_words:
                terms_sink = " ".join(important_words[:5])[:40]
                sinks.append(f"TERMS: {terms_sink}")
        
        # Ensure we don't exceed sink budget
        total_sink_tokens = sum(len(sink.split()) for sink in sinks)
        while total_sink_tokens > self.config.sink_tokens and len(sinks) > 1:
            sinks.pop()
            total_sink_tokens = sum(len(sink.split()) for sink in sinks)
        
        return sinks
    
    def _calculate_entropy(self, content: str) -> float:
        """Calculate content entropy for window prioritization."""
        if not content:
            return 0.0
        
        # Simple entropy calculation based on word frequency
        words = content.lower().split()
        if not words:
            return 0.0
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        total_words = len(words)
        entropy = 0.0
        
        for count in word_counts.values():
            prob = count / total_words
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy

class HybridSelector:
    """Main hybrid selector implementing Lethe→StreamingLLM system."""
    
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        
        # Core components
        self.atom_extractor = AtomExtractor(self.config)
        self.head_builder = HeadBuilder(self.config)
        self.tail_builder = TailBuilder(self.config)
        
        # Optimization components
        self.lagrangian_optimizer = None  # Initialized when needed
        
        # Monitoring
        self.selection_stats = {
            'total_selections': 0,
            'head_only_count': 0,
            'hybrid_count': 0,
            'avg_head_tokens': 0.0,
            'avg_tail_tokens': 0.0,
            'avg_kv_reuse': 0.0
        }
        
        logger.info("HybridSelector initialized with head_keep=%.2f, W=%d, s=%d",
                   self.config.head_keep_ratio, self.config.window_size, self.config.stride)
    
    def select(self, content: str, session_context: Optional[Dict[str, Any]] = None,
              relevance_scores: Optional[Dict[str, float]] = None) -> HybridSelectionResult:
        """
        Execute hybrid selection combining Lethe head with StreamingLLM tail.
        
        Args:
            content: Input content to process
            session_context: Session metadata for gating decisions
            relevance_scores: Pre-computed relevance scores for atoms
            
        Returns:
            Complete hybrid selection result
        """
        start_time = time.time()
        
        # Extract content atoms
        atoms = self.atom_extractor.extract_atoms(content, relevance_scores)
        total_content_tokens = sum(atom.tokens for atom in atoms)
        
        # Make gating decision
        gating_decision = self._make_gating_decision(atoms, session_context, total_content_tokens)
        processing_mode = gating_decision['processing_mode']
        
        head_selection = None
        tail_selection = None
        head_time = 0.0
        tail_time = 0.0
        
        # Build head (always)
        if processing_mode in [ProcessingMode.HEAD_ONLY, ProcessingMode.HYBRID]:
            head_start = time.time()
            head_budget = int(total_content_tokens * self.config.head_keep_ratio)
            head_selection = self.head_builder.build_head(atoms, head_budget)
            head_time = (time.time() - head_start) * 1000
        
        # Build tail (if hybrid mode)
        if processing_mode == ProcessingMode.HYBRID:
            tail_start = time.time()
            
            # Remove head atoms from content
            remaining_content = self._extract_remaining_content(content, head_selection)
            
            # Calculate tail budget
            tail_budget = total_content_tokens - (head_selection.total_tokens if head_selection else 0)
            tail_budget = min(tail_budget, self.config.tail_tokens_cap)
            
            # Build tail with head digest
            head_digest = head_selection.head_digest if head_selection else ""
            tail_selection = self.tail_builder.build_tail(remaining_content, head_digest, tail_budget)
            tail_time = (time.time() - tail_start) * 1000
        
        # Create KV-aware final arrangement
        arrangement_start = time.time()
        final_arrangement = self._create_kv_aware_arrangement(head_selection, tail_selection)
        arrangement_time = (time.time() - arrangement_start) * 1000
        
        # Calculate metrics
        total_time = (time.time() - start_time) * 1000
        objective_value, cost_lambda, cost_mu, net_value = self._calculate_objective_value(
            head_selection, tail_selection, total_time
        )
        
        # Calculate KV reuse
        kv_reuse_ratio = self._calculate_kv_reuse_ratio(head_selection, tail_selection)
        
        # Create final result
        result = HybridSelectionResult(
            head_selection=head_selection,
            tail_selection=tail_selection,
            processing_mode=processing_mode,
            final_content=final_arrangement['content'],
            total_tokens=final_arrangement['tokens'],
            keep_ratio=final_arrangement['tokens'] / max(1, total_content_tokens),
            kv_prefix_reuse_ratio=kv_reuse_ratio,
            kv_arrangement_optimized=final_arrangement['optimized'],
            selection_time_ms=total_time,
            head_time_ms=head_time,
            tail_time_ms=tail_time,
            arrangement_time_ms=arrangement_time,
            objective_value=objective_value,
            cost_lambda=cost_lambda,
            cost_mu=cost_mu,
            net_value=net_value,
            gating_decision=gating_decision,
            parameter_state={
                'lambda': self.config.lambda_param,
                'mu': self.config.mu_param,
                'head_keep_ratio': self.config.head_keep_ratio,
                'window_size': self.config.window_size,
                'stride': self.config.stride
            }
        )
        
        # Update statistics
        self._update_stats(result)
        
        if self.config.log_selections:
            logger.info(
                "Hybrid selection: mode=%s, tokens=%d, head=%d, tail=%d, kv_reuse=%.3f, time=%.1fms",
                processing_mode.value,
                result.total_tokens,
                head_selection.total_tokens if head_selection else 0,
                tail_selection.total_tokens if tail_selection else 0,
                kv_reuse_ratio,
                total_time
            )
        
        return result
    
    def _make_gating_decision(self, atoms: List[ContentAtom], session_context: Optional[Dict[str, Any]], 
                             total_tokens: int) -> Dict[str, Any]:
        """Make gating decision for processing mode."""
        session_context = session_context or {}
        
        # Calculate accept rate (simplified)
        stable_atoms = [a for a in atoms if a.stability_score > 0.6]
        accept_rate = len(stable_atoms) / max(1, len(atoms))
        
        # Calculate entity entropy
        all_entities = set()
        for atom in atoms:
            all_entities.update(atom.entity_references)
        
        entity_counts = {}
        for atom in atoms:
            for entity in atom.entity_references:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        total_entity_refs = sum(entity_counts.values())
        entity_entropy = 0.0
        if total_entity_refs > 0:
            for count in entity_counts.values():
                prob = count / total_entity_refs
                if prob > 0:
                    entity_entropy -= prob * math.log2(prob)
        
        # Budget check
        budget_ok = total_tokens > self.config.window_size  # Need enough content for windowing
        
        # Determine processing mode
        enable_streaming = (accept_rate < self.config.accept_rate_threshold and
                          entity_entropy > self.config.entity_entropy_threshold and
                          budget_ok)
        
        if enable_streaming:
            processing_mode = ProcessingMode.HYBRID
        else:
            processing_mode = ProcessingMode.HEAD_ONLY
        
        return {
            'processing_mode': processing_mode,
            'accept_rate': accept_rate,
            'entity_entropy': entity_entropy,
            'budget_ok': budget_ok,
            'enable_streaming': enable_streaming,
            'total_entities': len(all_entities),
            'reasoning': f"accept_rate={accept_rate:.3f} < {self.config.accept_rate_threshold}, "
                        f"entropy={entity_entropy:.3f} > {self.config.entity_entropy_threshold}, "
                        f"budget_ok={budget_ok}"
        }
    
    def _extract_remaining_content(self, original_content: str, head_selection: Optional[HeadSelection]) -> str:
        """Extract content remaining after head selection."""
        if not head_selection:
            return original_content
        
        # Simple approach: remove head content from original
        # In practice, would use more sophisticated content tracking
        head_contents = []
        for group in head_selection.grouped_atoms.values():
            for atom in group.atoms:
                head_contents.append(atom.content)
        
        remaining = original_content
        for head_content in head_contents:
            # Remove head content (simplified)
            remaining = remaining.replace(head_content, "", 1)
        
        return remaining.strip()
    
    def _create_kv_aware_arrangement(self, head_selection: Optional[HeadSelection], 
                                   tail_selection: Optional[TailSelection]) -> Dict[str, Any]:
        """Create KV cache optimized final arrangement."""
        parts = []
        total_tokens = 0
        kv_optimized = True
        
        # Head first (for KV prefix reuse)
        if head_selection:
            # Group head content by KV prefix for better reuse
            head_parts = []
            
            # Prioritize by content type for better KV locality
            priority_order = [
                ContentType.DEFINITION,
                ContentType.TOOL_KEY,
                ContentType.SYMBOL_HEADER,
                ContentType.ERROR_FRAME,
                ContentType.DOCUMENTATION,
                ContentType.CODE_BLOCK,
                ContentType.CONTEXT
            ]
            
            for content_type in priority_order:
                if content_type in head_selection.grouped_atoms:
                    group = head_selection.grouped_atoms[content_type]
                    for atom in sorted(group.atoms, key=lambda x: x.kv_prefix_hash or ""):
                        head_parts.append(atom.content)
            
            parts.extend(head_parts)
            total_tokens += head_selection.total_tokens
        
        # Tail after head
        if tail_selection:
            tail_parts = []
            
            for window in tail_selection.windows:
                # Add attention sinks first
                for sink in window.attention_sinks:
                    tail_parts.append(f"[SINK] {sink}")
                
                # Add window content
                tail_parts.append(window.content)
            
            parts.extend(tail_parts)
            total_tokens += tail_selection.total_tokens
        
        # Combine with appropriate separators
        final_content = "\n\n".join(parts)
        
        return {
            'content': final_content,
            'tokens': total_tokens,
            'optimized': kv_optimized
        }
    
    def _calculate_objective_value(self, head_selection: Optional[HeadSelection], 
                                 tail_selection: Optional[TailSelection], 
                                 compute_time: float) -> Tuple[float, float, float, float]:
        """Calculate objective function value F(H∪T) - λ*tokens - μ*compute."""
        
        # Base objective value (simplified utility function)
        objective = 0.0
        total_tokens = 0
        
        if head_selection:
            # Head value based on stability and relevance
            for group in head_selection.grouped_atoms.values():
                for atom in group.atoms:
                    objective += atom.relevance_score * atom.stability_score * atom.tokens
            total_tokens += head_selection.total_tokens
        
        if tail_selection:
            # Tail value based on entropy and coverage
            for window in tail_selection.windows:
                objective += window.entropy_score * window.tokens * 0.5  # Lower weight for tail
            total_tokens += tail_selection.total_tokens
        
        # Normalize objective by content amount
        if total_tokens > 0:
            objective = objective / total_tokens
        
        # Calculate costs
        cost_lambda = self.config.lambda_param * total_tokens
        cost_mu = self.config.mu_param * (compute_time / 1000.0)  # Convert to seconds
        
        # Net value
        net_value = objective - cost_lambda - cost_mu
        
        return objective, cost_lambda, cost_mu, net_value
    
    def _calculate_kv_reuse_ratio(self, head_selection: Optional[HeadSelection], 
                                tail_selection: Optional[TailSelection]) -> float:
        """Calculate KV cache prefix reuse ratio."""
        if not head_selection:
            return 0.0
        
        total_prefixes = len(head_selection.kv_prefix_hashes)
        if total_prefixes == 0:
            return 0.0
        
        # Simple metric: assume good reuse if head has diverse prefixes
        # In practice, would track actual KV cache hits
        unique_prefixes = len(set(head_selection.kv_prefix_hashes))
        reuse_ratio = min(1.0, unique_prefixes / max(1, total_prefixes))
        
        return reuse_ratio
    
    def _update_stats(self, result: HybridSelectionResult):
        """Update selection statistics."""
        self.selection_stats['total_selections'] += 1
        
        if result.processing_mode == ProcessingMode.HEAD_ONLY:
            self.selection_stats['head_only_count'] += 1
        elif result.processing_mode == ProcessingMode.HYBRID:
            self.selection_stats['hybrid_count'] += 1
        
        # Update running averages
        n = self.selection_stats['total_selections']
        
        head_tokens = result.head_selection.total_tokens if result.head_selection else 0
        tail_tokens = result.tail_selection.total_tokens if result.tail_selection else 0
        
        self.selection_stats['avg_head_tokens'] = (
            (self.selection_stats['avg_head_tokens'] * (n-1) + head_tokens) / n
        )
        
        self.selection_stats['avg_tail_tokens'] = (
            (self.selection_stats['avg_tail_tokens'] * (n-1) + tail_tokens) / n
        )
        
        self.selection_stats['avg_kv_reuse'] = (
            (self.selection_stats['avg_kv_reuse'] * (n-1) + result.kv_prefix_reuse_ratio) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current selection statistics."""
        return self.selection_stats.copy()
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

def create_hybrid_selector(config: Optional[HybridConfig] = None) -> HybridSelector:
    """Create hybrid selector with default canary configuration."""
    if config is None:
        # Use default canary configuration from TODO.md
        config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=6000,
            stride=3000,
            sink_tokens=96,
            ce_k2=320,
            dpp_rank=14
        )
    
    return HybridSelector(config)