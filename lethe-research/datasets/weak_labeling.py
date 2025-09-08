#!/usr/bin/env python3
"""
Weak Labeling Pipeline for LetheBench-Agents
============================================

Implements heuristic weak supervision for identifying minimal supporting atoms
needed to reproduce agent actions and outputs. Uses pattern-based analysis to
mark relevant evidence without requiring expensive human annotation.

Labeling Heuristics:
- Tool call atoms: Exact tool calls that produced referenced errors/outputs
- Dependency atoms: Config/plan atoms that actions depend on
- Definition atoms: Most recent definitions of referenced identifiers/files
- Context atoms: Recent conversation context relevant to current action

Key Features:
- Deterministic labeling with consistent seeding
- Confidence scoring for label quality assessment
- Support for multiple atom types and relationships
- Integration with existing entity extraction pipeline
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import logging

from agent_harness import AgentAtom

@dataclass
class WeakLabel:
    """Weak supervision label for supporting atoms"""
    query_id: str
    supporting_atom_ids: List[str]
    label_type: str  # 'tool_call', 'dependency', 'definition', 'context'
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class QueryContext:
    """Context information for a query requiring labeling"""
    query_id: str
    session_id: str
    turn_index: int
    query_content: str
    query_atom: AgentAtom
    session_atoms: List[AgentAtom]
    target_action: Optional[str] = None
    target_output: Optional[str] = None

class AtomDependencyAnalyzer:
    """Analyzes dependencies between atoms in agent conversations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns for identifying different types of references
        self.reference_patterns = {
            'file_reference': r'([a-zA-Z_][a-zA-Z0-9_./\\-]*\.(?:py|js|java|cpp|h|ts|jsx|tsx|go|rs|rb|php|txt|md|csv|json|yaml|yml|xml))',
            'function_reference': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            'variable_reference': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b',
            'error_reference': r'([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Exception)',
            'command_reference': r'`([^`]+)`',
            'tool_reference': r'(?:using|with|via|through)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'output_reference': r'(?:output|result|return)(?:ed|s)?\s*:?\s*["\']?([^"\'.\n]{5,50})["\']?'
        }
        
        # Dependency relationship patterns
        self.dependency_keywords = {
            'requires': ['need', 'require', 'depends', 'based on', 'using'],
            'produces': ['output', 'result', 'return', 'generate', 'create'],
            'modifies': ['change', 'update', 'modify', 'edit', 'fix'],
            'references': ['see', 'refer to', 'mentioned', 'above', 'previous']
        }
    
    def extract_references(self, atom: AgentAtom) -> Dict[str, List[str]]:
        """Extract all references from an atom's content"""
        references = defaultdict(list)
        
        for ref_type, pattern in self.reference_patterns.items():
            matches = re.findall(pattern, atom.content, re.IGNORECASE)
            references[ref_type] = list(set(matches))  # Remove duplicates
        
        return dict(references)
    
    def find_definition_atoms(self, reference: str, reference_type: str, 
                            session_atoms: List[AgentAtom], before_turn: int) -> List[Tuple[AgentAtom, float]]:
        """Find atoms that define or introduce a reference"""
        definition_atoms = []
        
        # Look for atoms before the current turn that define this reference
        for atom in session_atoms:
            if atom.turn_index >= before_turn:
                continue
            
            confidence = self._calculate_definition_confidence(reference, reference_type, atom)
            if confidence > 0.3:  # Minimum threshold for considering as definition
                definition_atoms.append((atom, confidence))
        
        # Sort by confidence (descending) and recency (descending)
        definition_atoms.sort(key=lambda x: (x[1], x[0].turn_index), reverse=True)
        
        return definition_atoms
    
    def find_dependency_atoms(self, target_atom: AgentAtom, 
                            session_atoms: List[AgentAtom]) -> List[Tuple[AgentAtom, str, float]]:
        """Find atoms that the target atom depends on"""
        dependency_atoms = []
        
        # Look for atoms before the target that it might depend on
        for atom in session_atoms:
            if atom.turn_index >= target_atom.turn_index:
                continue
            
            dependency_type, confidence = self._calculate_dependency_confidence(target_atom, atom)
            if confidence > 0.4:  # Minimum threshold for dependency
                dependency_atoms.append((atom, dependency_type, confidence))
        
        # Sort by confidence and recency
        dependency_atoms.sort(key=lambda x: (x[2], x[0].turn_index), reverse=True)
        
        return dependency_atoms
    
    def find_tool_call_atoms(self, target_content: str, 
                           session_atoms: List[AgentAtom]) -> List[Tuple[AgentAtom, float]]:
        """Find tool call atoms that produced referenced outputs/errors"""
        tool_call_atoms = []
        
        # Extract potential tool outputs/errors from target content
        output_patterns = [
            r'error:?\s*([^\n]{10,100})',
            r'output:?\s*([^\n]{10,100})',
            r'result:?\s*([^\n]{10,100})',
            r'(?:returned|produced):?\s*([^\n]{10,100})'
        ]
        
        target_outputs = []
        for pattern in output_patterns:
            matches = re.findall(pattern, target_content, re.IGNORECASE)
            target_outputs.extend(matches)
        
        if not target_outputs:
            return tool_call_atoms
        
        # Find tool observation atoms with matching outputs
        for atom in session_atoms:
            if atom.atom_type not in ['tool_observation', 'tool_action']:
                continue
            
            confidence = self._calculate_output_match_confidence(target_outputs, atom)
            if confidence > 0.5:
                tool_call_atoms.append((atom, confidence))
        
        tool_call_atoms.sort(key=lambda x: x[1], reverse=True)
        return tool_call_atoms
    
    def _calculate_definition_confidence(self, reference: str, reference_type: str, 
                                       atom: AgentAtom) -> float:
        """Calculate confidence that an atom defines a reference"""
        content = atom.content.lower()
        ref_lower = reference.lower()
        
        base_confidence = 0.0
        
        # Exact reference match
        if ref_lower in content:
            base_confidence = 0.6
        
        # Boost confidence based on atom type and context
        if atom.atom_type == 'agent_plan':
            if any(keyword in content for keyword in ['define', 'create', 'implement', 'add']):
                base_confidence += 0.2
        elif atom.atom_type == 'tool_action':
            if reference_type == 'file_reference' and 'file_write' in atom.metadata.get('tool_call', {}).get('tool_name', ''):
                base_confidence += 0.3
        elif atom.atom_type == 'agent_response':
            if any(keyword in content for keyword in ['created', 'defined', 'implemented', 'added']):
                base_confidence += 0.2
        
        # Boost for code blocks containing the reference
        if '```' in atom.content and ref_lower in content:
            base_confidence += 0.3
        
        return min(1.0, base_confidence)
    
    def _calculate_dependency_confidence(self, target_atom: AgentAtom, 
                                       candidate_atom: AgentAtom) -> Tuple[str, float]:
        """Calculate dependency confidence between atoms"""
        target_content = target_atom.content.lower()
        candidate_content = candidate_atom.content.lower()
        
        max_confidence = 0.0
        dependency_type = 'unknown'
        
        # Check for explicit dependency keywords
        for dep_type, keywords in self.dependency_keywords.items():
            for keyword in keywords:
                if keyword in target_content:
                    # Look for shared entities or references
                    shared_entities = self._find_shared_entities(target_atom, candidate_atom)
                    if shared_entities:
                        confidence = 0.5 + 0.1 * len(shared_entities)
                        if confidence > max_confidence:
                            max_confidence = confidence
                            dependency_type = dep_type
        
        # Temporal proximity boost (more recent = more likely dependency)
        turn_distance = target_atom.turn_index - candidate_atom.turn_index
        if turn_distance <= 3:  # Within 3 turns
            max_confidence += 0.2 * (1.0 - turn_distance / 3.0)
        
        # Same session boost
        if target_atom.session_id == candidate_atom.session_id:
            max_confidence += 0.1
        
        return dependency_type, min(1.0, max_confidence)
    
    def _calculate_output_match_confidence(self, target_outputs: List[str], 
                                         atom: AgentAtom) -> float:
        """Calculate confidence that atom produced one of the target outputs"""
        if atom.atom_type != 'tool_observation':
            return 0.0
        
        atom_content = atom.content.lower()
        tool_result = atom.metadata.get('tool_result', {})
        
        # Check for output matches in various tool result fields
        search_fields = [
            tool_result.get('stdout', ''),
            tool_result.get('stderr', ''),
            tool_result.get('error', ''),
            tool_result.get('content', ''),
            str(tool_result.get('result', ''))
        ]
        
        max_confidence = 0.0
        
        for target_output in target_outputs:
            target_lower = target_output.lower().strip()
            if len(target_lower) < 5:  # Skip very short outputs
                continue
            
            for field_content in search_fields:
                field_lower = str(field_content).lower()
                
                # Exact match
                if target_lower in field_lower:
                    max_confidence = max(max_confidence, 0.9)
                
                # Partial match (at least 70% of words)
                target_words = set(target_lower.split())
                field_words = set(field_lower.split())
                
                if target_words and field_words:
                    overlap = len(target_words & field_words) / len(target_words)
                    if overlap >= 0.7:
                        max_confidence = max(max_confidence, 0.6 + 0.2 * overlap)
        
        return max_confidence
    
    def _find_shared_entities(self, atom1: AgentAtom, atom2: AgentAtom) -> List[str]:
        """Find shared entities between two atoms"""
        entities1 = set(entity['value'] for entity in atom1.entities)
        entities2 = set(entity['value'] for entity in atom2.entities)
        return list(entities1 & entities2)

class WeakLabeler:
    """Main weak labeling pipeline for agent conversations"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.dependency_analyzer = AtomDependencyAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Labeling configuration
        self.confidence_thresholds = {
            'tool_call': 0.7,
            'dependency': 0.5,
            'definition': 0.6,
            'context': 0.4
        }
        
        self.max_supporting_atoms = {
            'tool_call': 3,
            'dependency': 5,
            'definition': 2,
            'context': 8
        }
    
    def generate_weak_labels(self, session_atoms: List[AgentAtom]) -> List[WeakLabel]:
        """Generate weak labels for all queries in a session"""
        weak_labels = []
        
        # Identify query points (user requests and agent responses that need support)
        query_contexts = self._identify_query_contexts(session_atoms)
        
        for query_context in query_contexts:
            labels = self._label_query_context(query_context)
            weak_labels.extend(labels)
        
        return weak_labels
    
    def _identify_query_contexts(self, session_atoms: List[AgentAtom]) -> List[QueryContext]:
        """Identify points in conversation that need weak labels"""
        query_contexts = []
        
        for atom in session_atoms:
            # Label user requests (need supporting context)
            if atom.atom_type == 'user_request':
                query_contexts.append(QueryContext(
                    query_id=f"query_{atom.atom_id}",
                    session_id=atom.session_id,
                    turn_index=atom.turn_index,
                    query_content=atom.content,
                    query_atom=atom,
                    session_atoms=session_atoms
                ))
            
            # Label agent responses (need supporting evidence)
            elif atom.atom_type == 'agent_response':
                # Only label responses that reference specific outputs or make claims
                if self._has_supporting_evidence_need(atom):
                    query_contexts.append(QueryContext(
                        query_id=f"query_{atom.atom_id}",
                        session_id=atom.session_id,
                        turn_index=atom.turn_index,
                        query_content=atom.content,
                        query_atom=atom,
                        session_atoms=session_atoms
                    ))
            
            # Label tool actions (need dependency context)
            elif atom.atom_type == 'tool_action':
                query_contexts.append(QueryContext(
                    query_id=f"query_{atom.atom_id}",
                    session_id=atom.session_id,
                    turn_index=atom.turn_index,
                    query_content=atom.content,
                    query_atom=atom,
                    session_atoms=session_atoms
                ))
        
        return query_contexts
    
    def _has_supporting_evidence_need(self, atom: AgentAtom) -> bool:
        """Check if agent response needs supporting evidence"""
        content_lower = atom.content.lower()
        
        # Responses that reference outputs, errors, or make specific claims
        evidence_indicators = [
            'output:', 'result:', 'error:', 'shows that', 'indicates',
            'according to', 'based on', 'the file contains', 'the command returned',
            'i found', 'the analysis shows', 'the data reveals'
        ]
        
        return any(indicator in content_lower for indicator in evidence_indicators)
    
    def _label_query_context(self, query_context: QueryContext) -> List[WeakLabel]:
        """Generate weak labels for a specific query context"""
        labels = []
        
        # Generate different types of supporting atom labels
        tool_call_labels = self._generate_tool_call_labels(query_context)
        dependency_labels = self._generate_dependency_labels(query_context)
        definition_labels = self._generate_definition_labels(query_context)
        context_labels = self._generate_context_labels(query_context)
        
        labels.extend(tool_call_labels)
        labels.extend(dependency_labels)
        labels.extend(definition_labels)
        labels.extend(context_labels)
        
        # Remove duplicate atom references and apply confidence filtering
        labels = self._deduplicate_and_filter_labels(labels, query_context.query_id)
        
        return labels
    
    def _generate_tool_call_labels(self, query_context: QueryContext) -> List[WeakLabel]:
        """Generate labels for tool calls that produced referenced outputs"""
        labels = []
        
        # Find tool call atoms that match referenced outputs in the query
        tool_atoms = self.dependency_analyzer.find_tool_call_atoms(
            query_context.query_content, query_context.session_atoms
        )
        
        # Group by tool call sequence (action + observation)
        tool_sequences = self._group_tool_sequences(tool_atoms, query_context.session_atoms)
        
        for sequence, confidence in tool_sequences[:self.max_supporting_atoms['tool_call']]:
            if confidence >= self.confidence_thresholds['tool_call']:
                labels.append(WeakLabel(
                    query_id=query_context.query_id,
                    supporting_atom_ids=[atom.atom_id for atom in sequence],
                    label_type='tool_call',
                    confidence=confidence,
                    reasoning=f'Tool sequence produced output referenced in query',
                    metadata={
                        'tool_names': [atom.metadata.get('tool_name', 'unknown') for atom in sequence],
                        'sequence_length': len(sequence)
                    }
                ))
        
        return labels
    
    def _generate_dependency_labels(self, query_context: QueryContext) -> List[WeakLabel]:
        """Generate labels for atoms that the query depends on"""
        labels = []
        
        # Find atoms that this query depends on
        dependency_atoms = self.dependency_analyzer.find_dependency_atoms(
            query_context.query_atom, query_context.session_atoms
        )
        
        for atom, dep_type, confidence in dependency_atoms[:self.max_supporting_atoms['dependency']]:
            if confidence >= self.confidence_thresholds['dependency']:
                labels.append(WeakLabel(
                    query_id=query_context.query_id,
                    supporting_atom_ids=[atom.atom_id],
                    label_type='dependency',
                    confidence=confidence,
                    reasoning=f'Query {dep_type} information from this atom',
                    metadata={
                        'dependency_type': dep_type,
                        'atom_type': atom.atom_type,
                        'turn_distance': query_context.turn_index - atom.turn_index
                    }
                ))
        
        return labels
    
    def _generate_definition_labels(self, query_context: QueryContext) -> List[WeakLabel]:
        """Generate labels for atoms that define referenced entities"""
        labels = []
        
        # Extract references from query content
        references = self.dependency_analyzer.extract_references(query_context.query_atom)
        
        for ref_type, ref_list in references.items():
            for reference in ref_list:
                definition_atoms = self.dependency_analyzer.find_definition_atoms(
                    reference, ref_type, query_context.session_atoms, query_context.turn_index
                )
                
                # Take the most confident definition (usually the most recent)
                for atom, confidence in definition_atoms[:self.max_supporting_atoms['definition']]:
                    if confidence >= self.confidence_thresholds['definition']:
                        labels.append(WeakLabel(
                            query_id=query_context.query_id,
                            supporting_atom_ids=[atom.atom_id],
                            label_type='definition',
                            confidence=confidence,
                            reasoning=f'Defines referenced {ref_type}: {reference}',
                            metadata={
                                'reference_type': ref_type,
                                'reference_value': reference,
                                'definition_atom_type': atom.atom_type
                            }
                        ))
        
        return labels
    
    def _generate_context_labels(self, query_context: QueryContext) -> List[WeakLabel]:
        """Generate labels for recent conversational context"""
        labels = []
        
        # Get recent atoms before this query (within reasonable context window)
        recent_atoms = [
            atom for atom in query_context.session_atoms
            if atom.turn_index < query_context.turn_index 
            and query_context.turn_index - atom.turn_index <= 5  # 5 turn context window
        ]
        
        # Sort by recency and relevance
        context_candidates = []
        for atom in recent_atoms:
            relevance_score = self._calculate_contextual_relevance(query_context, atom)
            if relevance_score > self.confidence_thresholds['context']:
                context_candidates.append((atom, relevance_score))
        
        context_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top context atoms
        selected_atoms = context_candidates[:self.max_supporting_atoms['context']]
        
        if selected_atoms:
            labels.append(WeakLabel(
                query_id=query_context.query_id,
                supporting_atom_ids=[atom.atom_id for atom, _ in selected_atoms],
                label_type='context',
                confidence=sum(score for _, score in selected_atoms) / len(selected_atoms),
                reasoning='Recent conversational context relevant to query',
                metadata={
                    'context_window_size': len(selected_atoms),
                    'avg_relevance_score': sum(score for _, score in selected_atoms) / len(selected_atoms),
                    'turn_range': [min(atom.turn_index for atom, _ in selected_atoms),
                                 max(atom.turn_index for atom, _ in selected_atoms)]
                }
            ))
        
        return labels
    
    def _group_tool_sequences(self, tool_atoms: List[Tuple[AgentAtom, float]], 
                            session_atoms: List[AgentAtom]) -> List[Tuple[List[AgentAtom], float]]:
        """Group tool action and observation atoms into sequences"""
        sequences = []
        
        # Find action-observation pairs
        for atom, confidence in tool_atoms:
            if atom.atom_type == 'tool_observation':
                # Find the corresponding tool action
                action_atom = None
                for other_atom in session_atoms:
                    if (other_atom.atom_type == 'tool_action' and 
                        other_atom.turn_index == atom.turn_index and
                        other_atom.timestamp < atom.timestamp):
                        action_atom = other_atom
                        break
                
                if action_atom:
                    sequences.append(([action_atom, atom], confidence))
                else:
                    sequences.append(([atom], confidence))
        
        sequences.sort(key=lambda x: x[1], reverse=True)
        return sequences
    
    def _calculate_contextual_relevance(self, query_context: QueryContext, 
                                      candidate_atom: AgentAtom) -> float:
        """Calculate how relevant a candidate atom is as context"""
        query_content = query_context.query_content.lower()
        candidate_content = candidate_atom.content.lower()
        
        # Base relevance from shared entities
        shared_entities = self.dependency_analyzer._find_shared_entities(
            query_context.query_atom, candidate_atom
        )
        base_relevance = min(0.6, len(shared_entities) * 0.15)
        
        # Boost for lexical similarity
        query_words = set(query_content.split())
        candidate_words = set(candidate_content.split())
        
        if query_words and candidate_words:
            word_overlap = len(query_words & candidate_words) / len(query_words | candidate_words)
            base_relevance += word_overlap * 0.3
        
        # Boost for recency (more recent = more relevant)
        turn_distance = query_context.turn_index - candidate_atom.turn_index
        recency_boost = max(0, 0.2 * (1.0 - turn_distance / 5.0))
        base_relevance += recency_boost
        
        # Atom type specific boosts
        if candidate_atom.atom_type in ['agent_plan', 'agent_response']:
            base_relevance += 0.1  # Plans and responses tend to be more relevant
        elif candidate_atom.atom_type == 'tool_observation':
            base_relevance += 0.05  # Tool results can provide useful context
        
        return min(1.0, base_relevance)
    
    def _deduplicate_and_filter_labels(self, labels: List[WeakLabel], 
                                     query_id: str) -> List[WeakLabel]:
        """Remove duplicate atom references and apply quality filtering"""
        # Group by atom ID to handle duplicates
        atom_to_labels = defaultdict(list)
        
        for label in labels:
            for atom_id in label.supporting_atom_ids:
                atom_to_labels[atom_id].append(label)
        
        # Keep highest confidence label for each atom
        final_labels = []
        seen_atom_ids = set()
        
        for label in sorted(labels, key=lambda x: x.confidence, reverse=True):
            # Check if any atoms in this label are already used
            if not any(atom_id in seen_atom_ids for atom_id in label.supporting_atom_ids):
                final_labels.append(label)
                seen_atom_ids.update(label.supporting_atom_ids)
        
        return final_labels
    
    def export_labels(self, weak_labels: List[WeakLabel], output_path: str):
        """Export weak labels to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for label in weak_labels:
                f.write(json.dumps(label.to_dict()) + '\n')
        
        self.logger.info(f"Exported {len(weak_labels)} weak labels to {output_path}")
    
    def validate_labels(self, weak_labels: List[WeakLabel]) -> Dict[str, Any]:
        """Validate label quality and consistency"""
        validation_results = {
            'total_labels': len(weak_labels),
            'label_type_distribution': Counter(label.label_type for label in weak_labels),
            'confidence_stats': {
                'mean': sum(label.confidence for label in weak_labels) / len(weak_labels) if weak_labels else 0,
                'min': min(label.confidence for label in weak_labels) if weak_labels else 0,
                'max': max(label.confidence for label in weak_labels) if weak_labels else 0
            },
            'quality_issues': []
        }
        
        # Check for quality issues
        low_confidence_labels = [label for label in weak_labels if label.confidence < 0.3]
        if low_confidence_labels:
            validation_results['quality_issues'].append(
                f"Found {len(low_confidence_labels)} labels with very low confidence (<0.3)"
            )
        
        # Check for over-labeling (too many supporting atoms per query)
        query_atom_counts = Counter()
        for label in weak_labels:
            query_atom_counts[label.query_id] += len(label.supporting_atom_ids)
        
        over_labeled_queries = [
            (query_id, count) for query_id, count in query_atom_counts.items() 
            if count > 15  # More than 15 supporting atoms per query
        ]
        
        if over_labeled_queries:
            validation_results['quality_issues'].append(
                f"Found {len(over_labeled_queries)} over-labeled queries (>15 supporting atoms)"
            )
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    from agent_harness import AgentConversationSimulator
    
    # Generate sample conversation
    simulator = AgentConversationSimulator(seed=42)
    atoms = simulator.generate_session('coding', target_turns=20, complexity_level='medium')
    
    # Generate weak labels
    labeler = WeakLabeler(seed=42)
    weak_labels = labeler.generate_weak_labels(atoms)
    
    print(f"Generated {len(weak_labels)} weak labels for {len(atoms)} atoms")
    
    # Show label distribution
    label_types = Counter(label.label_type for label in weak_labels)
    print(f"Label type distribution: {dict(label_types)}")
    
    # Validate labels
    validation_results = labeler.validate_labels(weak_labels)
    print(f"Validation results: {validation_results}")
    
    # Export sample
    labeler.export_labels(weak_labels[:10], 'sample_weak_labels.jsonl')