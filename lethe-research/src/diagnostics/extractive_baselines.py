"""
Rung 2: Extractive Baselines (No LLM)
====================================

Non-LLM baselines that extract answers directly from text using heuristics.
These establish upper bounds for retrieval-only performance.

Methods:
- Regex/heuristic extractor: Code: most frequent identifier near ERROR blocks; QA: BM25 on sentences  
- Top-span heuristic: For samples where SpanCoverage@K=1, return that span
- Metrics: macro-P@5 and token-F1 on extractive system
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

# Import BM25 implementation (simplified version)
import math

logger = logging.getLogger(__name__)

class SimpleBM25:
    """Simplified BM25 implementation for sentence scoring."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.doc_len = {}
        self.avgdl = 0.0
        self.corpus = []
        
    def fit(self, corpus: List[str]):
        """Fit BM25 on corpus of sentences."""
        self.corpus = corpus
        doc_count = len(corpus)
        
        # Calculate document frequencies
        df = Counter()
        total_len = 0
        
        for i, doc in enumerate(corpus):
            words = doc.lower().split()
            self.doc_len[i] = len(words)
            total_len += len(words)
            
            for word in set(words):
                df[word] += 1
        
        self.avgdl = total_len / doc_count if doc_count > 0 else 0
        
        # Calculate IDF scores
        for word, freq in df.items():
            self.idf[word] = math.log((doc_count - freq + 0.5) / (freq + 0.5))
    
    def score(self, query: str, doc_id: int) -> float:
        """Calculate BM25 score for query against document."""
        if doc_id >= len(self.corpus):
            return 0.0
            
        doc = self.corpus[doc_id]
        query_words = query.lower().split()
        doc_words = Counter(doc.lower().split())
        
        score = 0.0
        for word in query_words:
            if word in doc_words:
                tf = doc_words[word]
                idf = self.idf.get(word, 0)
                doc_len = self.doc_len[doc_id]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                
                score += idf * (numerator / denominator)
        
        return score

class ExtractionBaselines:
    """Non-LLM extractive baselines for answer extraction."""
    
    def __init__(self):
        """Initialize extractive baselines."""
        self.bm25 = SimpleBM25()
        
    def extract_code_identifier_near_error(self, 
                                         context: str, 
                                         error_keywords: List[str] = None) -> str:
        """
        Code extractor: Find most frequent identifier near ERROR blocks.
        
        Args:
            context: Code context to search
            error_keywords: Keywords indicating error locations
            
        Returns:
            Most likely identifier causing the error
        """
        if not error_keywords:
            error_keywords = ['error', 'exception', 'traceback', 'fail', 'bug', 'wrong', 'issue']
        
        # Split into lines for analysis
        lines = context.split('\n')
        error_regions = []
        
        # Find regions around error keywords
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in error_keywords):
                # Include context around error (±3 lines)
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                error_regions.extend(lines[start:end])
        
        # If no error regions found, use entire context
        if not error_regions:
            error_regions = lines
        
        # Extract identifiers from error regions
        error_text = '\n'.join(error_regions)
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', error_text)
        
        # Filter out common keywords and very short identifiers
        python_keywords = {
            'def', 'class', 'if', 'else', 'for', 'while', 'try', 'except', 'import', 
            'from', 'return', 'print', 'True', 'False', 'None', 'and', 'or', 'not',
            'in', 'is', 'with', 'as', 'pass', 'break', 'continue', 'lambda', 'yield'
        }
        
        filtered_identifiers = [
            ident for ident in identifiers 
            if len(ident) > 1 and ident.lower() not in python_keywords
        ]
        
        if not filtered_identifiers:
            return ""
        
        # Return most frequent identifier
        counter = Counter(filtered_identifiers)
        return counter.most_common(1)[0][0]
    
    def extract_qa_answer_with_bm25(self, 
                                   context: str, 
                                   question: str, 
                                   max_answer_length: int = 50) -> str:
        """
        QA extractor: Use BM25 on sentences to find most relevant span.
        
        Args:
            context: Text context to search
            question: Question to answer
            max_answer_length: Maximum length of extracted answer in words
            
        Returns:
            Most likely answer span
        """
        if not context or not question:
            return ""
        
        # Split context into sentences
        sentences = self._split_into_sentences(context)
        if not sentences:
            return ""
        
        # Fit BM25 on sentences
        self.bm25.fit(sentences)
        
        # Score all sentences against question
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self.bm25.score(question, i)
            sentence_scores.append((score, i, sentence))
        
        # Get top scoring sentence
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        if not sentence_scores:
            return ""
        
        best_sentence = sentence_scores[0][2]
        
        # Extract answer span from best sentence
        answer_span = self._extract_answer_span(best_sentence, question, max_answer_length)
        
        return answer_span
    
    def extract_top_span_heuristic(self, 
                                  selected_atoms: List[str], 
                                  gold_answers: List[str], 
                                  coverage_result: Dict[str, Any]) -> str:
        """
        Top-span heuristic: For samples where SpanCoverage@K=1, return that span.
        
        Args:
            selected_atoms: Retrieved atoms
            gold_answers: Ground truth answers
            coverage_result: Coverage analysis result
            
        Returns:
            Extracted span if coverage exists, empty string otherwise
        """
        if not coverage_result.get('coverage', False):
            return ""
        
        covering_atoms = coverage_result.get('covering_atoms', [])
        if not covering_atoms:
            return ""
        
        # Use the first covering atom
        first_covering = covering_atoms[0]
        atom_index = first_covering['atom_index']
        matched_answer = first_covering['matched_answer']
        
        if atom_index >= len(selected_atoms):
            return ""
        
        atom_text = selected_atoms[atom_index]
        
        # Try to extract a clean span around the matched answer
        answer_text = str(matched_answer)
        
        # Find the answer in the atom
        answer_lower = answer_text.lower()
        atom_lower = atom_text.lower()
        
        answer_pos = atom_lower.find(answer_lower)
        if answer_pos == -1:
            return answer_text  # Fallback to original answer
        
        # Extract context around the answer (±20 words)
        words_before = atom_text[:answer_pos].split()[-20:]
        words_answer = atom_text[answer_pos:answer_pos + len(answer_text)].split()
        words_after = atom_text[answer_pos + len(answer_text):].split()[:20]
        
        # Combine and clean
        context_span = ' '.join(words_before + words_answer + words_after)
        context_span = re.sub(r'\s+', ' ', context_span).strip()
        
        # If span is too long, just return the matched answer
        if len(context_span.split()) > 100:
            return answer_text
        
        return context_span
    
    def extract_regex_patterns(self, 
                              context: str, 
                              task_name: str) -> str:
        """
        Task-specific regex extractors for structured tasks.
        
        Args:
            context: Text context to search
            task_name: Name of the task (determines extraction strategy)
            
        Returns:
            Extracted answer using regex patterns
        """
        if not context:
            return ""
        
        # Task-specific extraction patterns
        if 'passkey' in task_name.lower():
            # Extract 5-digit numbers
            matches = re.findall(r'\b\d{5}\b', context)
            return matches[-1] if matches else ""
        
        elif 'number_string' in task_name.lower():
            # Extract numbers from context
            matches = re.findall(r'\b\d+\b', context)
            return matches[-1] if matches else ""
        
        elif 'kv_retrieval' in task_name.lower():
            # Extract key-value pairs
            # Look for patterns like "key: value" or "key = value"
            kv_patterns = [
                r':\s*([a-zA-Z0-9_]+)',  # After colon
                r'=\s*([a-zA-Z0-9_]+)',  # After equals
                r'"([a-zA-Z0-9_]+)"',     # In quotes
                r"'([a-zA-Z0-9_]+)'"      # In single quotes
            ]
            for pattern in kv_patterns:
                matches = re.findall(pattern, context)
                if matches:
                    return matches[-1]  # Return last match
            return ""
        
        elif 'math_find' in task_name.lower():
            # Extract numbers (integers or floats)
            matches = re.findall(r'\b\d+\.?\d*\b', context)
            return matches[-1] if matches else ""
        
        elif 'choice' in task_name.lower():
            # Extract multiple choice answers (A, B, C, D)
            matches = re.findall(r'\b[A-D]\b', context)
            return matches[-1] if matches else ""
        
        elif 'code_debug' in task_name.lower():
            # Extract function or variable names
            matches = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', context)
            # Filter out common keywords
            filtered = [m for m in matches if len(m) > 2 and m not in ['def', 'class', 'return', 'print']]
            return filtered[-1] if filtered else ""
        
        else:
            # Default: extract noun phrases or short sentences
            sentences = self._split_into_sentences(context)
            if sentences:
                # Return shortest sentence (likely to be an answer)
                shortest = min(sentences, key=len)
                if len(shortest.split()) <= 10:
                    return shortest.strip()
        
        return ""
    
    def run_extractive_evaluation(self, 
                                 samples: List[Dict[str, Any]], 
                                 selected_atoms_per_sample: Dict[str, List[str]], 
                                 coverage_results_per_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive extractive baseline evaluation.
        
        Args:
            samples: List of evaluation samples
            selected_atoms_per_sample: Mapping from sample_id to selected atoms
            coverage_results_per_sample: Coverage analysis results per sample
            
        Returns:
            Dict with extractive evaluation results
        """
        logger.info("Running extractive baseline evaluation...")
        
        results_by_method = defaultdict(list)
        results_by_task = defaultdict(lambda: defaultdict(list))
        
        # Import scoring function with fallback
        try:
            from benchmarks.infinitebench.src.compute_scores import get_score_one
        except ImportError:
            # Use fallback from rung0_scoring_sanity
            from .rung0_scoring_sanity import get_score_one
        
        for sample in samples:
            sample_id = sample.get('id', 'unknown')
            task_name = sample.get('task_name', 'unknown')
            ground_truth = sample.get('ground_truth') or sample.get('label')
            question = sample.get('question', '')
            context = sample.get('context', '')
            
            if not ground_truth:
                continue
            
            selected_atoms = selected_atoms_per_sample.get(sample_id, [])
            coverage_result = coverage_results_per_sample.get(sample_id, {})
            
            # Method 1: Task-specific regex extraction
            try:
                regex_prediction = self.extract_regex_patterns(context, task_name)
                if regex_prediction:
                    score = get_score_one(regex_prediction, ground_truth, task_name, "extractive")
                    results_by_method['regex'].append(score)
                    results_by_task[task_name]['regex'].append(score)
            except Exception as e:
                logger.debug(f"Regex extraction failed for {sample_id}: {e}")
            
            # Method 2: Code identifier extraction (for code tasks)
            if 'code' in task_name.lower():
                try:
                    code_prediction = self.extract_code_identifier_near_error(context)
                    if code_prediction:
                        score = get_score_one(code_prediction, ground_truth, task_name, "extractive")
                        results_by_method['code_identifier'].append(score)
                        results_by_task[task_name]['code_identifier'].append(score)
                except Exception as e:
                    logger.debug(f"Code extraction failed for {sample_id}: {e}")
            
            # Method 3: BM25 sentence extraction (for QA tasks)
            if question and ('qa' in task_name.lower() or 'dialogue' in task_name.lower()):
                try:
                    bm25_prediction = self.extract_qa_answer_with_bm25(context, question)
                    if bm25_prediction:
                        score = get_score_one(bm25_prediction, ground_truth, task_name, "extractive")
                        results_by_method['bm25_qa'].append(score)
                        results_by_task[task_name]['bm25_qa'].append(score)
                except Exception as e:
                    logger.debug(f"BM25 extraction failed for {sample_id}: {e}")
            
            # Method 4: Top-span heuristic (when coverage exists)
            if selected_atoms:
                try:
                    # Get gold answers for coverage analysis
                    if isinstance(ground_truth, list):
                        gold_answers = [str(x) for x in ground_truth]
                    else:
                        gold_answers = [str(ground_truth)]
                    
                    span_prediction = self.extract_top_span_heuristic(
                        selected_atoms, gold_answers, coverage_result
                    )
                    if span_prediction:
                        score = get_score_one(span_prediction, ground_truth, task_name, "extractive")
                        results_by_method['top_span'].append(score)
                        results_by_task[task_name]['top_span'].append(score)
                except Exception as e:
                    logger.debug(f"Top-span extraction failed for {sample_id}: {e}")
        
        # Calculate summary statistics
        method_summaries = {}
        for method_name, scores in results_by_method.items():
            if scores:
                method_summaries[method_name] = {
                    'macro_p5': np.mean(scores),  # Approximate P@5
                    'samples': len(scores),
                    'std': np.std(scores),
                    'max_score': np.max(scores)
                }
        
        task_summaries = {}
        for task_name, methods in results_by_task.items():
            task_summaries[task_name] = {}
            for method_name, scores in methods.items():
                if scores:
                    task_summaries[task_name][method_name] = {
                        'mean_score': np.mean(scores),
                        'samples': len(scores)
                    }
        
        # Overall best extractive performance
        overall_scores = []
        for scores in results_by_method.values():
            overall_scores.extend(scores)
        
        overall_performance = {
            'macro_p5': np.mean(overall_scores) if overall_scores else 0.0,
            'total_predictions': len(overall_scores),
            'std': np.std(overall_scores) if overall_scores else 0.0
        }
        
        return {
            'method_summaries': method_summaries,
            'task_summaries': task_summaries,
            'overall_performance': overall_performance,
            'samples_processed': len(samples)
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _extract_answer_span(self, sentence: str, question: str, max_length: int) -> str:
        """Extract answer span from sentence based on question keywords."""
        if not sentence:
            return ""
        
        # Simple heuristic: find noun phrases or phrases after question keywords
        question_lower = question.lower()
        sentence_lower = sentence.lower()
        
        # Look for question word patterns
        question_patterns = {
            'what': r'(is|was|are|were)\s+([^.!?]{1,50})',
            'who': r'(is|was|are|were)\s+([^.!?]{1,50})', 
            'where': r'(in|at|on)\s+([^.!?]{1,50})',
            'when': r'(in|on|at|during)\s+([^.!?]{1,50})',
            'how': r'(by|through|via)\s+([^.!?]{1,50})'
        }
        
        for q_word, pattern in question_patterns.items():
            if q_word in question_lower:
                matches = re.findall(pattern, sentence_lower)
                if matches:
                    return matches[0][1].strip()
        
        # Fallback: return first few words of sentence
        words = sentence.split()[:max_length]
        return ' '.join(words).strip()