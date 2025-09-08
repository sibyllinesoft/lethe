#!/usr/bin/env python3
"""
Prose Gold Labeler for LetheBench

Implements weak supervision for identifying supporting evidence spans
in long-form prose discussions through entity and temporal overlap analysis.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import re
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False
from typing import List, Dict, Set, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import dateutil.parser as date_parser

@dataclass
class ProseChunk:
    """Represents a gold prose evidence chunk annotation."""
    chunk_id: str
    session_id: str
    turn_id: int
    content: str
    chunk_type: str  # 'entity_support', 'temporal_support', 'factual_evidence', 'quote'
    entities: List[str]
    temporal_refs: List[str]
    context_start: int
    context_end: int
    confidence: float
    metadata: Dict

class ProseLabeler:
    """
    Generates gold annotations for prose discussion sessions.
    
    Uses weak supervision to identify:
    - Supporting evidence spans through entity overlap
    - Temporal references and time-based evidence
    - Factual claims and their supporting context
    - Direct quotes and attributions
    - Causal relationships and explanations
    """
    
    def __init__(self):
        """Initialize prose labeler with NLP models."""
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model for NER and linguistic analysis
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model 'en_core_web_sm' not found. Using basic patterns.")
                self.nlp = None
        else:
            self.logger.warning("spaCy not available. Using basic pattern matching only.")
            self.nlp = None
        
        # Entity patterns for when spaCy is not available
        self.entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'ORG': r'\b(?:Inc|Corp|LLC|Ltd|Company|Organization|Institute|University|Department)\b',
            'GPE': r'\b(?:United States|USA|UK|China|Japan|Germany|France|Canada|Australia)\b',
            'DATE': r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'MONEY': r'\$[\d,]+(?:\.\d{2})?',
            'PERCENT': r'\d+(?:\.\d+)?%',
            'CARDINAL': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        }
        
        # Temporal expression patterns
        self.temporal_patterns = {
            'absolute_date': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'relative_date': r'\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|ago|later|before|after|during|since|until)\b',
            'year': r'\b(?:19|20)\d{2}\b',
            'time_period': r'\b(?:morning|afternoon|evening|night|week|month|year|decade|century)\b',
            'duration': r'\b\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?)\b'
        }
        
        # Evidence indicator patterns
        self.evidence_patterns = {
            'attribution': r'\b(?:according to|as stated by|reported by|said|claims|argues|believes)\b',
            'citation': r'\b(?:study|research|report|paper|article|survey|analysis|investigation)\b',
            'causation': r'\b(?:because|due to|caused by|resulted in|led to|as a result|therefore|consequently)\b',
            'contrast': r'\b(?:however|but|although|despite|nevertheless|on the other hand|in contrast)\b',
            'support': r'\b(?:evidence|proof|data|statistics|findings|results|shows|demonstrates|indicates)\b'
        }
        
        # Question types for identifying information needs
        self.question_patterns = {
            'who': r'\bwho\b.*\?',
            'what': r'\bwhat\b.*\?',
            'when': r'\bwhen\b.*\?', 
            'where': r'\bwhere\b.*\?',
            'why': r'\bwhy\b.*\?',
            'how': r'\bhow\b.*\?'
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text using spaCy or pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entity dictionaries with type, text, and span info
        """
        entities = []
        
        if self.nlp:
            # Use spaCy for robust NER
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9  # spaCy is generally reliable
                })
        else:
            # Fall back to pattern matching
            for label, pattern in self.entity_patterns.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        'text': match.group(0),
                        'label': label,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.7  # Pattern matching is less reliable
                    })
        
        return entities
    
    def extract_temporal_expressions(self, text: str) -> List[Dict]:
        """
        Extract temporal expressions and time references.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of temporal expression dictionaries
        """
        temporal_refs = []
        
        for temp_type, pattern in self.temporal_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                temporal_refs.append({
                    'text': match.group(0),
                    'type': temp_type,
                    'start': match.start(),
                    'end': match.end(),
                    'normalized': self._normalize_temporal_expression(match.group(0), temp_type)
                })
        
        return temporal_refs
    
    def _normalize_temporal_expression(self, text: str, temp_type: str) -> Optional[str]:
        """Normalize temporal expressions to standard format."""
        text_lower = text.lower()
        
        if temp_type == 'absolute_date':
            try:
                # Try to parse the date
                parsed_date = date_parser.parse(text)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                return None
        
        elif temp_type == 'year':
            if text.isdigit() and 1900 <= int(text) <= 2030:
                return text
        
        elif temp_type == 'relative_date':
            # Map relative expressions to approximate dates
            relative_mappings = {
                'yesterday': -1,
                'today': 0,
                'tomorrow': 1,
                'last week': -7,
                'next week': 7,
                'last month': -30,
                'next month': 30
            }
            
            if text_lower in relative_mappings:
                days_offset = relative_mappings[text_lower]
                reference_date = datetime.now() + timedelta(days=days_offset)
                return reference_date.strftime('%Y-%m-%d')
        
        return text  # Return original if normalization fails
    
    def identify_question_entities(self, question_text: str) -> Tuple[str, List[str]]:
        """
        Identify the question type and entities being asked about.
        
        Args:
            question_text: Text of the question
            
        Returns:
            Tuple of (question_type, list_of_entities)
        """
        question_type = 'unknown'
        
        # Determine question type
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, question_text, re.IGNORECASE):
                question_type = q_type
                break
        
        # Extract entities from the question
        entities = self.extract_entities(question_text)
        entity_texts = [ent['text'] for ent in entities]
        
        return question_type, entity_texts
    
    def find_supporting_spans(self, 
                            question_entities: List[str],
                            question_type: str,
                            candidate_text: str) -> List[Dict]:
        """
        Find text spans that provide supporting evidence for a question.
        
        Args:
            question_entities: Entities mentioned in the question
            question_type: Type of question (who, what, when, etc.)
            candidate_text: Text to search for supporting evidence
            
        Returns:
            List of supporting span dictionaries
        """
        supporting_spans = []
        
        # Split candidate text into sentences for analysis
        sentences = self._split_into_sentences(candidate_text)
        
        for i, sentence in enumerate(sentences):
            # Extract entities and temporal refs from sentence
            sent_entities = self.extract_entities(sentence)
            sent_temporal = self.extract_temporal_expressions(sentence)
            
            # Calculate overlap scores
            entity_overlap = self._calculate_entity_overlap(question_entities, sent_entities)
            temporal_relevance = self._calculate_temporal_relevance(question_type, sent_temporal)
            evidence_strength = self._calculate_evidence_strength(sentence)
            
            # Combined confidence score
            confidence = (entity_overlap * 0.4 + 
                         temporal_relevance * 0.3 + 
                         evidence_strength * 0.3)
            
            if confidence > 0.3:  # Threshold for inclusion
                # Find sentence boundaries in original text
                sentence_start = candidate_text.find(sentence)
                sentence_end = sentence_start + len(sentence)
                
                supporting_spans.append({
                    'text': sentence,
                    'start': sentence_start,
                    'end': sentence_end,
                    'confidence': confidence,
                    'entities': [ent['text'] for ent in sent_entities],
                    'temporal_refs': [temp['text'] for temp in sent_temporal],
                    'evidence_indicators': self._find_evidence_indicators(sentence),
                    'sentence_index': i
                })
        
        # Sort by confidence and return top spans
        supporting_spans.sort(key=lambda x: x['confidence'], reverse=True)
        return supporting_spans[:10]  # Limit to top 10 spans
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic heuristics."""
        # Simple sentence splitting (could be improved with spaCy)
        sentences = re.split(r'[.!?]+\s+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def _calculate_entity_overlap(self, question_entities: List[str], sentence_entities: List[Dict]) -> float:
        """Calculate entity overlap between question and sentence."""
        if not question_entities or not sentence_entities:
            return 0.0
        
        sent_entity_texts = [ent['text'].lower() for ent in sentence_entities]
        question_entities_lower = [ent.lower() for ent in question_entities]
        
        # Calculate Jaccard similarity
        overlap = len(set(question_entities_lower) & set(sent_entity_texts))
        union = len(set(question_entities_lower) | set(sent_entity_texts))
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_temporal_relevance(self, question_type: str, temporal_refs: List[Dict]) -> float:
        """Calculate temporal relevance for time-based questions."""
        if question_type != 'when' or not temporal_refs:
            return 0.0 if question_type == 'when' else 0.1  # Small boost for non-temporal questions
        
        # Higher scores for more specific temporal expressions
        max_score = 0.0
        for temp_ref in temporal_refs:
            score = {
                'absolute_date': 0.9,
                'year': 0.8,
                'relative_date': 0.7,
                'duration': 0.6,
                'time_period': 0.5
            }.get(temp_ref['type'], 0.3)
            
            max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_evidence_strength(self, sentence: str) -> float:
        """Calculate how much the sentence looks like evidence."""
        sentence_lower = sentence.lower()
        evidence_score = 0.0
        
        # Check for evidence indicators
        for indicator_type, pattern in self.evidence_patterns.items():
            if re.search(pattern, sentence_lower):
                score_values = {
                    'attribution': 0.3,
                    'citation': 0.4,
                    'causation': 0.3,
                    'contrast': 0.2,
                    'support': 0.4
                }
                evidence_score += score_values.get(indicator_type, 0.2)
        
        # Boost for sentences with numbers/statistics
        if re.search(r'\b\d+(?:\.\d+)?%\b|\b\d{4}\b|\$[\d,]+', sentence):
            evidence_score += 0.2
        
        # Boost for longer, more detailed sentences
        if len(sentence) > 100:
            evidence_score += 0.1
        
        return min(1.0, evidence_score)
    
    def _find_evidence_indicators(self, sentence: str) -> List[str]:
        """Find evidence indicator phrases in sentence."""
        indicators = []
        sentence_lower = sentence.lower()
        
        for indicator_type, pattern in self.evidence_patterns.items():
            matches = re.findall(pattern, sentence_lower)
            indicators.extend(matches)
        
        return indicators
    
    def label_session_turns(self, session_turns: List[Dict]) -> List[ProseChunk]:
        """
        Generate gold prose chunks for a complete dialog session.
        
        Args:
            session_turns: List of turns in LetheBench format
            
        Returns:
            List of ProseChunk annotations
        """
        chunks = []
        session_id = session_turns[0]['session_id'] if session_turns else "unknown"
        
        # Find questions in the session
        questions = self._identify_questions(session_turns)
        
        for question_turn, question_entities, question_type in questions:
            # Find supporting evidence in subsequent turns
            for turn in session_turns:
                if turn['turn'] <= question_turn['turn']:
                    continue  # Skip the question and earlier turns
                
                supporting_spans = self.find_supporting_spans(
                    question_entities, question_type, turn['text']
                )
                
                # Create chunks from high-confidence spans
                for i, span in enumerate(supporting_spans):
                    if span['confidence'] > 0.5:  # Quality threshold
                        chunk_id = f"{session_id}_q{question_turn['turn']}_turn{turn['turn']}_span{i}"
                        
                        # Determine chunk type based on characteristics
                        chunk_type = self._classify_prose_chunk(span, question_type)
                        
                        chunks.append(ProseChunk(
                            chunk_id=chunk_id,
                            session_id=session_id,
                            turn_id=turn['turn'],
                            content=span['text'],
                            chunk_type=chunk_type,
                            entities=span['entities'],
                            temporal_refs=span['temporal_refs'],
                            context_start=span['start'],
                            context_end=span['end'],
                            confidence=span['confidence'],
                            metadata={
                                'question_turn': question_turn['turn'],
                                'question_type': question_type,
                                'question_entities': question_entities,
                                'evidence_indicators': span['evidence_indicators'],
                                'sentence_index': span['sentence_index'],
                                'turn_role': turn['role'],
                                'turn_meta': turn.get('meta', {})
                            }
                        ))
        
        return chunks
    
    def _identify_questions(self, session_turns: List[Dict]) -> List[Tuple[Dict, List[str], str]]:
        """Identify questions in the session turns."""
        questions = []
        
        for turn in session_turns:
            text = turn['text']
            
            # Look for question marks and question patterns
            if '?' in text:
                question_sentences = [sent for sent in self._split_into_sentences(text) if '?' in sent]
                
                for question in question_sentences:
                    question_type, question_entities = self.identify_question_entities(question)
                    
                    # Only process questions that ask about specific entities or have clear type
                    if question_entities or question_type != 'unknown':
                        questions.append((turn, question_entities, question_type))
                        break  # One question per turn for simplicity
        
        return questions
    
    def _classify_prose_chunk(self, span: Dict, question_type: str) -> str:
        """Classify the type of prose evidence chunk."""
        text = span['text'].lower()
        
        # Check for direct quotes
        if '"' in text or "'" in text:
            return 'quote'
        
        # Check for factual evidence
        if any(indicator in text for indicator in ['study', 'research', 'data', 'statistics', 'survey']):
            return 'factual_evidence'
        
        # Check for temporal support
        if span['temporal_refs'] and question_type == 'when':
            return 'temporal_support'
        
        # Default to entity support
        return 'entity_support'

def test_prose_labeler():
    """Test the prose labeler with sample data."""
    labeler = ProseLabeler()
    
    # Test entity extraction
    sample_text = "According to Dr. John Smith from Harvard University, the study published in January 2023 showed that 75% of participants reported improved outcomes."
    
    print("Testing entity extraction...")
    entities = labeler.extract_entities(sample_text)
    for ent in entities:
        print(f"  {ent['label']}: {ent['text']}")
    
    print("\nTesting temporal extraction...")
    temporal_refs = labeler.extract_temporal_expressions(sample_text)
    for temp in temporal_refs:
        print(f"  {temp['type']}: {temp['text']} -> {temp.get('normalized', 'N/A')}")
    
    # Test supporting span identification
    question = "When was the Harvard study published?"
    question_type, question_entities = labeler.identify_question_entities(question)
    
    print(f"\nQuestion analysis:")
    print(f"  Type: {question_type}")
    print(f"  Entities: {question_entities}")
    
    supporting_spans = labeler.find_supporting_spans(question_entities, question_type, sample_text)
    
    print(f"\nSupporting spans:")
    for span in supporting_spans:
        print(f"  Confidence: {span['confidence']:.2f}")
        print(f"  Text: {span['text']}")
        print(f"  Entities: {span['entities']}")
        print(f"  Temporal: {span['temporal_refs']}")
        print()
    
    # Test session labeling
    sample_session = [
        {
            'session_id': 'prose_test_session',
            'turn': 0,
            'role': 'user',
            'text': 'When did the COVID-19 pandemic officially begin according to WHO?',
            'meta': {}
        },
        {
            'session_id': 'prose_test_session',
            'turn': 1,
            'role': 'assistant',
            'text': 'The World Health Organization officially declared COVID-19 a pandemic on March 11, 2020. This declaration came after the virus had spread to multiple countries and continents, with sustained human-to-human transmission observed globally.',
            'meta': {}
        }
    ]
    
    print("Testing session labeling...")
    chunks = labeler.label_session_turns(sample_session)
    
    print(f"Generated {len(chunks)} prose chunks:")
    for chunk in chunks:
        print(f"  {chunk.chunk_id}: {chunk.chunk_type}")
        print(f"    Content: {chunk.content}")
        print(f"    Entities: {chunk.entities}")
        print(f"    Temporal: {chunk.temporal_refs}")
        print(f"    Confidence: {chunk.confidence:.2f}")
        print()

if __name__ == "__main__":
    test_prose_labeler()