#!/usr/bin/env python3
"""
Quality Metrics Calculator for LetheBench

Computes comprehensive quality metrics for dataset validation including:
- Statistical analysis and distribution metrics
- Inter-annotator agreement where applicable
- Data diversity and coverage analysis
- Genre-specific quality assessments

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import defaultdict, Counter
import math

@dataclass
class QualityMetrics:
    """Container for computed quality metrics."""
    # Basic statistics
    total_sessions: int
    total_turns: int
    total_chunks: int
    avg_turns_per_session: float
    avg_text_length: float
    
    # Distribution metrics
    session_length_distribution: Dict[str, int]
    role_distribution: Dict[str, int]
    genre_distribution: Dict[str, int]
    
    # Diversity metrics
    vocabulary_size: int
    entity_diversity: int
    topic_diversity: int
    unique_sessions_ratio: float
    
    # Quality scores
    overall_quality_score: float
    data_coverage_score: float
    annotation_consistency_score: float
    
    # Genre-specific metrics
    genre_specific: Dict[str, Dict[str, Any]]

class QualityMetricsCalculator:
    """
    Calculates comprehensive quality metrics for LetheBench dataset.
    
    Features:
    - Statistical analysis of dataset characteristics
    - Diversity measurements across multiple dimensions
    - Quality scoring based on academic benchmarking standards
    - Genre-specific metric computation
    - Inter-annotator agreement calculation where applicable
    """
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds for academic standards
        self.quality_thresholds = {
            'min_sessions_per_genre': 1000,
            'min_chunks_per_genre': 10000,
            'min_avg_turns': 3.0,
            'min_vocabulary_size': 10000,
            'min_entity_diversity': 500,
            'target_coverage_score': 0.8,
            'target_consistency_score': 0.7
        }
        
        # Weights for overall quality score
        self.quality_weights = {
            'data_volume': 0.25,
            'diversity': 0.25,
            'coverage': 0.25,
            'consistency': 0.25
        }
    
    def calculate_metrics(self, dataset_dir: Path) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics for the dataset.
        
        Args:
            dataset_dir: Directory containing dataset files
            
        Returns:
            QualityMetrics with complete analysis
        """
        self.logger.info("Calculating dataset quality metrics...")
        
        # Load all data
        all_data = self._load_dataset(dataset_dir)
        
        # Basic statistics
        basic_stats = self._calculate_basic_statistics(all_data)
        
        # Distribution analysis
        distributions = self._calculate_distributions(all_data)
        
        # Diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(all_data)
        
        # Quality scores
        quality_scores = self._calculate_quality_scores(all_data, basic_stats, diversity_metrics)
        
        # Genre-specific analysis
        genre_specific = self._calculate_genre_specific_metrics(all_data)
        
        return QualityMetrics(
            # Basic stats
            total_sessions=basic_stats['total_sessions'],
            total_turns=basic_stats['total_turns'],
            total_chunks=basic_stats['total_chunks'],
            avg_turns_per_session=basic_stats['avg_turns_per_session'],
            avg_text_length=basic_stats['avg_text_length'],
            
            # Distributions
            session_length_distribution=distributions['session_lengths'],
            role_distribution=distributions['roles'],
            genre_distribution=distributions['genres'],
            
            # Diversity
            vocabulary_size=diversity_metrics['vocabulary_size'],
            entity_diversity=diversity_metrics['entity_diversity'],
            topic_diversity=diversity_metrics['topic_diversity'],
            unique_sessions_ratio=diversity_metrics['unique_sessions_ratio'],
            
            # Quality scores
            overall_quality_score=quality_scores['overall'],
            data_coverage_score=quality_scores['coverage'],
            annotation_consistency_score=quality_scores['consistency'],
            
            # Genre-specific
            genre_specific=genre_specific
        )
    
    def _load_dataset(self, dataset_dir: Path) -> Dict[str, List[Dict]]:
        """Load all dataset files and organize by genre."""
        data = defaultdict(list)
        
        genres = ['code', 'tool', 'prose']
        splits = ['train', 'dev', 'test']
        
        for genre in genres:
            for split in splits:
                file_path = dataset_dir / genre / f"{split}.jsonl"
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    turn_data = json.loads(line.strip())
                                    turn_data['_genre'] = genre
                                    turn_data['_split'] = split
                                    data[genre].append(turn_data)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        self.logger.error(f"Error loading {file_path}: {e}")
        
        return dict(data)
    
    def _calculate_basic_statistics(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate basic dataset statistics."""
        stats = {
            'total_sessions': 0,
            'total_turns': 0,
            'total_chunks': 0,
            'session_lengths': [],
            'text_lengths': []
        }
        
        all_sessions = set()
        
        for genre, turns in all_data.items():
            sessions_in_genre = defaultdict(list)
            
            for turn in turns:
                session_id = turn.get('session_id', 'unknown')
                all_sessions.add(session_id)
                sessions_in_genre[session_id].append(turn)
                
                stats['total_turns'] += 1
                
                # Text length analysis
                text = turn.get('text', '')
                stats['text_lengths'].append(len(text))
                
                # Count chunks if available
                meta = turn.get('meta', {})
                if 'chunks' in meta or 'gold_chunks' in meta:
                    chunk_count = len(meta.get('chunks', meta.get('gold_chunks', [])))
                    stats['total_chunks'] += chunk_count
            
            # Session length analysis
            for session_turns in sessions_in_genre.values():
                stats['session_lengths'].append(len(session_turns))
        
        stats['total_sessions'] = len(all_sessions)
        stats['avg_turns_per_session'] = (
            stats['total_turns'] / max(1, stats['total_sessions'])
        )
        stats['avg_text_length'] = (
            sum(stats['text_lengths']) / max(1, len(stats['text_lengths']))
        )
        
        return stats
    
    def _calculate_distributions(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Calculate distribution metrics."""
        distributions = {
            'session_lengths': {},
            'roles': defaultdict(int),
            'genres': defaultdict(int)
        }
        
        # Session length distribution (binned)
        all_session_lengths = []
        sessions_by_genre = defaultdict(set)
        
        for genre, turns in all_data.items():
            sessions_in_genre = defaultdict(list)
            
            for turn in turns:
                session_id = turn.get('session_id', 'unknown')
                sessions_in_genre[session_id].append(turn)
                sessions_by_genre[genre].add(session_id)
                
                # Role distribution
                role = turn.get('role', 'unknown')
                distributions['roles'][role] += 1
                
                # Genre distribution
                distributions['genres'][genre] += 1
            
            # Collect session lengths
            for session_turns in sessions_in_genre.values():
                all_session_lengths.append(len(session_turns))
        
        # Bin session lengths
        if all_session_lengths:
            min_len, max_len = min(all_session_lengths), max(all_session_lengths)
            bins = ['1-2', '3-5', '6-10', '11-20', '21-50', '50+']
            
            for length in all_session_lengths:
                if length <= 2:
                    distributions['session_lengths']['1-2'] = \
                        distributions['session_lengths'].get('1-2', 0) + 1
                elif length <= 5:
                    distributions['session_lengths']['3-5'] = \
                        distributions['session_lengths'].get('3-5', 0) + 1
                elif length <= 10:
                    distributions['session_lengths']['6-10'] = \
                        distributions['session_lengths'].get('6-10', 0) + 1
                elif length <= 20:
                    distributions['session_lengths']['11-20'] = \
                        distributions['session_lengths'].get('11-20', 0) + 1
                elif length <= 50:
                    distributions['session_lengths']['21-50'] = \
                        distributions['session_lengths'].get('21-50', 0) + 1
                else:
                    distributions['session_lengths']['50+'] = \
                        distributions['session_lengths'].get('50+', 0) + 1
        
        return distributions
    
    def _calculate_diversity_metrics(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate diversity and coverage metrics."""
        # Vocabulary diversity
        all_words = set()
        all_entities = set()
        all_topics = set()
        all_sessions = set()
        
        for genre, turns in all_data.items():
            for turn in turns:
                session_id = turn.get('session_id', 'unknown')
                all_sessions.add(session_id)
                
                # Extract words for vocabulary analysis
                text = turn.get('text', '').lower()
                words = text.split()
                all_words.update(words)
                
                # Extract entities and topics from metadata
                meta = turn.get('meta', {})
                
                # Entity diversity
                if 'entities' in meta:
                    entities = meta['entities']
                    if isinstance(entities, list):
                        all_entities.update(entities)
                
                # Topic diversity
                if 'tags' in meta:
                    tags = meta['tags']
                    if isinstance(tags, list):
                        all_topics.update(tags)
                
                if 'topics' in meta:
                    topics = meta['topics']
                    if isinstance(topics, list):
                        all_topics.update(topics)
        
        # Calculate diversity metrics
        diversity_metrics = {
            'vocabulary_size': len(all_words),
            'entity_diversity': len(all_entities),
            'topic_diversity': len(all_topics),
            'unique_sessions_ratio': len(all_sessions) / max(1, sum(len(turns) for turns in all_data.values()))
        }
        
        return diversity_metrics
    
    def _calculate_quality_scores(self, 
                                all_data: Dict[str, List[Dict]], 
                                basic_stats: Dict[str, Any],
                                diversity_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores based on academic standards."""
        
        # Data volume score
        volume_score = 0.0
        for genre, turns in all_data.items():
            sessions_in_genre = len(set(turn.get('session_id', 'unknown') for turn in turns))
            chunks_in_genre = basic_stats.get('total_chunks', 0) / max(1, len(all_data))
            
            # Score based on meeting minimum thresholds
            session_score = min(1.0, sessions_in_genre / self.quality_thresholds['min_sessions_per_genre'])
            chunk_score = min(1.0, chunks_in_genre / self.quality_thresholds['min_chunks_per_genre'])
            
            volume_score += (session_score + chunk_score) / 2
        
        volume_score /= max(1, len(all_data))
        
        # Diversity score
        vocab_score = min(1.0, diversity_metrics['vocabulary_size'] / self.quality_thresholds['min_vocabulary_size'])
        entity_score = min(1.0, diversity_metrics['entity_diversity'] / self.quality_thresholds['min_entity_diversity'])
        
        diversity_score = (vocab_score + entity_score) / 2
        
        # Coverage score (how well distributed the data is)
        coverage_score = self._calculate_coverage_score(all_data)
        
        # Consistency score (based on format compliance and annotation quality)
        consistency_score = self._calculate_consistency_score(all_data)
        
        # Overall quality score
        overall_score = (
            self.quality_weights['data_volume'] * volume_score +
            self.quality_weights['diversity'] * diversity_score +
            self.quality_weights['coverage'] * coverage_score +
            self.quality_weights['consistency'] * consistency_score
        )
        
        return {
            'overall': overall_score,
            'volume': volume_score,
            'diversity': diversity_score,
            'coverage': coverage_score,
            'consistency': consistency_score
        }
    
    def _calculate_coverage_score(self, all_data: Dict[str, List[Dict]]) -> float:
        """Calculate how well the dataset covers different scenarios."""
        coverage_factors = []
        
        for genre, turns in all_data.items():
            # Role balance
            roles = [turn.get('role', 'unknown') for turn in turns]
            role_counts = Counter(roles)
            
            if len(role_counts) > 1:
                # Calculate entropy of role distribution
                total = sum(role_counts.values())
                role_entropy = -sum((count/total) * math.log2(count/total) 
                                  for count in role_counts.values())
                max_entropy = math.log2(len(role_counts))
                role_balance = role_entropy / max_entropy if max_entropy > 0 else 0
            else:
                role_balance = 0
            
            coverage_factors.append(role_balance)
            
            # Session length diversity
            sessions = defaultdict(list)
            for turn in turns:
                sessions[turn.get('session_id', 'unknown')].append(turn)
            
            session_lengths = [len(turns) for turns in sessions.values()]
            if session_lengths:
                length_std = np.std(session_lengths)
                length_mean = np.mean(session_lengths)
                length_cv = length_std / max(1, length_mean)  # Coefficient of variation
                length_diversity = min(1.0, length_cv)  # Normalize
            else:
                length_diversity = 0
            
            coverage_factors.append(length_diversity)
        
        return sum(coverage_factors) / max(1, len(coverage_factors))
    
    def _calculate_consistency_score(self, all_data: Dict[str, List[Dict]]) -> float:
        """Calculate consistency score based on format compliance."""
        consistency_factors = []
        
        for genre, turns in all_data.items():
            # Check required field completeness
            required_fields = ['session_id', 'turn', 'role', 'text', 'ts', 'meta']
            field_completeness = []
            
            for turn in turns:
                complete_fields = sum(1 for field in required_fields if field in turn and turn[field])
                completeness = complete_fields / len(required_fields)
                field_completeness.append(completeness)
            
            avg_completeness = sum(field_completeness) / max(1, len(field_completeness))
            consistency_factors.append(avg_completeness)
            
            # Check timestamp format consistency
            timestamps = [turn.get('ts', '') for turn in turns if 'ts' in turn]
            if timestamps:
                # Simple check for consistent format (all ISO-like or all not)
                iso_like_count = sum(1 for ts in timestamps if 'T' in ts and ':' in ts)
                timestamp_consistency = max(iso_like_count, len(timestamps) - iso_like_count) / len(timestamps)
            else:
                timestamp_consistency = 1.0
            
            consistency_factors.append(timestamp_consistency)
        
        return sum(consistency_factors) / max(1, len(consistency_factors))
    
    def _calculate_genre_specific_metrics(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Any]]:
        """Calculate genre-specific quality metrics."""
        genre_metrics = {}
        
        for genre, turns in all_data.items():
            if genre == 'code':
                genre_metrics[genre] = self._calculate_code_metrics(turns)
            elif genre == 'tool':
                genre_metrics[genre] = self._calculate_tool_metrics(turns)
            elif genre == 'prose':
                genre_metrics[genre] = self._calculate_prose_metrics(turns)
        
        return genre_metrics
    
    def _calculate_code_metrics(self, turns: List[Dict]) -> Dict[str, Any]:
        """Calculate code-specific metrics."""
        code_indicators = {
            'has_code_blocks': 0,
            'has_inline_code': 0,
            'programming_languages': set(),
            'accepted_answers': 0,
            'repositories': set()
        }
        
        for turn in turns:
            text = turn.get('text', '')
            meta = turn.get('meta', {})
            
            # Code block detection
            if '```' in text:
                code_indicators['has_code_blocks'] += 1
            
            # Inline code detection
            if '`' in text:
                code_indicators['has_inline_code'] += 1
            
            # Programming languages
            if 'language' in meta:
                code_indicators['programming_languages'].add(meta['language'])
            
            if 'tags' in meta and isinstance(meta['tags'], list):
                for tag in meta['tags']:
                    if tag.lower() in ['python', 'javascript', 'java', 'cpp', 'go', 'rust']:
                        code_indicators['programming_languages'].add(tag)
            
            # Accepted answers
            if meta.get('is_accepted', False):
                code_indicators['accepted_answers'] += 1
            
            # Repository diversity
            if 'repository' in meta:
                code_indicators['repositories'].add(meta['repository'])
        
        # Convert sets to counts
        return {
            'code_block_coverage': code_indicators['has_code_blocks'] / max(1, len(turns)),
            'inline_code_coverage': code_indicators['has_inline_code'] / max(1, len(turns)),
            'language_diversity': len(code_indicators['programming_languages']),
            'accepted_answer_ratio': code_indicators['accepted_answers'] / max(1, len(turns)),
            'repository_diversity': len(code_indicators['repositories']),
            'avg_code_quality_score': (
                (code_indicators['has_code_blocks'] / max(1, len(turns))) +
                (code_indicators['accepted_answers'] / max(1, len(turns)))
            ) / 2
        }
    
    def _calculate_tool_metrics(self, turns: List[Dict]) -> Dict[str, Any]:
        """Calculate tool-specific metrics."""
        tool_indicators = {
            'command_outputs': 0,
            'structured_outputs': 0,
            'tools_used': set(),
            'has_dependencies': 0
        }
        
        for turn in turns:
            text = turn.get('text', '')
            meta = turn.get('meta', {})
            
            # Command output detection
            if any(indicator in text for indicator in ['$', '>', '>>>', 'PS']):
                tool_indicators['command_outputs'] += 1
            
            # Structured output detection
            if any(indicator in text for indicator in ['{', '[', '|', '===', '---']):
                tool_indicators['structured_outputs'] += 1
            
            # Tool identification
            if 'tool_name' in meta:
                tool_indicators['tools_used'].add(meta['tool_name'])
            
            # Dependencies
            if meta.get('dependencies') or 'dependencies' in meta:
                tool_indicators['has_dependencies'] += 1
        
        return {
            'command_output_coverage': tool_indicators['command_outputs'] / max(1, len(turns)),
            'structured_output_coverage': tool_indicators['structured_outputs'] / max(1, len(turns)),
            'tool_diversity': len(tool_indicators['tools_used']),
            'dependency_coverage': tool_indicators['has_dependencies'] / max(1, len(turns)),
            'avg_tool_quality_score': (
                (tool_indicators['command_outputs'] / max(1, len(turns))) +
                (tool_indicators['structured_outputs'] / max(1, len(turns)))
            ) / 2
        }
    
    def _calculate_prose_metrics(self, turns: List[Dict]) -> Dict[str, Any]:
        """Calculate prose-specific metrics."""
        prose_indicators = {
            'has_entities': 0,
            'has_temporal_refs': 0,
            'topics_covered': set(),
            'has_supporting_evidence': 0,
            'question_answer_pairs': 0
        }
        
        for turn in turns:
            text = turn.get('text', '')
            meta = turn.get('meta', {})
            
            # Entity presence
            if 'entities' in meta and meta['entities']:
                prose_indicators['has_entities'] += 1
            
            # Temporal references
            if 'temporal_refs' in meta and meta['temporal_refs']:
                prose_indicators['has_temporal_refs'] += 1
            
            # Topic coverage
            if 'topics' in meta and isinstance(meta['topics'], list):
                prose_indicators['topics_covered'].update(meta['topics'])
            
            # Supporting evidence
            if any(indicator in text.lower() for indicator in ['according to', 'study', 'research', 'data']):
                prose_indicators['has_supporting_evidence'] += 1
            
            # Question detection
            if '?' in text:
                prose_indicators['question_answer_pairs'] += 1
        
        return {
            'entity_coverage': prose_indicators['has_entities'] / max(1, len(turns)),
            'temporal_coverage': prose_indicators['has_temporal_refs'] / max(1, len(turns)),
            'topic_diversity': len(prose_indicators['topics_covered']),
            'evidence_coverage': prose_indicators['has_supporting_evidence'] / max(1, len(turns)),
            'question_coverage': prose_indicators['question_answer_pairs'] / max(1, len(turns)),
            'avg_prose_quality_score': (
                (prose_indicators['has_entities'] / max(1, len(turns))) +
                (prose_indicators['has_supporting_evidence'] / max(1, len(turns)))
            ) / 2
        }
    
    def generate_quality_report(self, 
                              metrics: QualityMetrics, 
                              output_file: Optional[Path] = None) -> str:
        """Generate comprehensive quality metrics report."""
        report_lines = [
            "# LetheBench Quality Metrics Report",
            f"Generated: {self._get_timestamp()}\n"
        ]
        
        # Executive Summary
        report_lines.extend([
            "## Executive Summary",
            f"- **Overall Quality Score**: {metrics.overall_quality_score:.3f} / 1.000",
            f"- **Total Sessions**: {metrics.total_sessions:,}",
            f"- **Total Turns**: {metrics.total_turns:,}",
            f"- **Total Gold Chunks**: {metrics.total_chunks:,}",
            f"- **Average Turns per Session**: {metrics.avg_turns_per_session:.1f}",
            f"- **Average Text Length**: {metrics.avg_text_length:.1f} characters\n"
        ])
        
        # Quality Assessment
        quality_status = "✓ EXCELLENT" if metrics.overall_quality_score >= 0.8 else \
                        "⚠ GOOD" if metrics.overall_quality_score >= 0.6 else \
                        "✗ NEEDS IMPROVEMENT"
        
        report_lines.extend([
            "## Quality Assessment",
            f"**Status**: {quality_status}",
            f"- Data Coverage Score: {metrics.data_coverage_score:.3f}",
            f"- Annotation Consistency: {metrics.annotation_consistency_score:.3f}",
            ""
        ])
        
        # Distribution Analysis
        report_lines.extend([
            "## Data Distributions",
            "### Session Length Distribution"
        ])
        
        for length_bin, count in sorted(metrics.session_length_distribution.items()):
            percentage = (count / metrics.total_sessions) * 100 if metrics.total_sessions > 0 else 0
            report_lines.append(f"- {length_bin} turns: {count:,} sessions ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "### Role Distribution"
        ])
        
        for role, count in sorted(metrics.role_distribution.items()):
            percentage = (count / metrics.total_turns) * 100 if metrics.total_turns > 0 else 0
            report_lines.append(f"- {role}: {count:,} turns ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "### Genre Distribution"
        ])
        
        for genre, count in sorted(metrics.genre_distribution.items()):
            percentage = (count / metrics.total_turns) * 100 if metrics.total_turns > 0 else 0
            report_lines.append(f"- {genre}: {count:,} turns ({percentage:.1f}%)")
        
        # Diversity Metrics
        report_lines.extend([
            "",
            "## Diversity Metrics",
            f"- **Vocabulary Size**: {metrics.vocabulary_size:,} unique words",
            f"- **Entity Diversity**: {metrics.entity_diversity:,} unique entities",
            f"- **Topic Diversity**: {metrics.topic_diversity:,} unique topics",
            f"- **Session Uniqueness**: {metrics.unique_sessions_ratio:.3f}",
            ""
        ])
        
        # Genre-Specific Analysis
        report_lines.extend([
            "## Genre-Specific Quality Analysis"
        ])
        
        for genre, genre_metrics in metrics.genre_specific.items():
            report_lines.extend([
                f"### {genre.title()} Genre"
            ])
            
            for metric_name, value in genre_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"- {metric_name.replace('_', ' ').title()}: {value:.3f}")
                else:
                    report_lines.append(f"- {metric_name.replace('_', ' ').title()}: {value}")
            
            report_lines.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(metrics)
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        if metrics.overall_quality_score < 0.7:
            recommendations.append("Overall quality score is below target. Consider improving data collection and annotation processes.")
        
        if metrics.total_sessions < 3000:  # 1000 per genre minimum
            recommendations.append("Increase dataset size to meet minimum requirements (1000+ sessions per genre).")
        
        if metrics.avg_turns_per_session < 3.0:
            recommendations.append("Sessions are too short on average. Focus on collecting longer, more substantial conversations.")
        
        if metrics.vocabulary_size < 10000:
            recommendations.append("Vocabulary diversity is low. Include more diverse topics and domains.")
        
        if metrics.data_coverage_score < 0.8:
            recommendations.append("Improve data coverage by ensuring balanced representation across different scenarios.")
        
        if metrics.annotation_consistency_score < 0.7:
            recommendations.append("Enhance annotation consistency through better guidelines and quality control.")
        
        # Genre-specific recommendations
        for genre, genre_metrics in metrics.genre_specific.items():
            if genre == 'code':
                if genre_metrics.get('code_block_coverage', 0) < 0.5:
                    recommendations.append(f"Increase code block coverage in {genre} genre.")
                if genre_metrics.get('language_diversity', 0) < 5:
                    recommendations.append(f"Include more programming languages in {genre} genre.")
            
            elif genre == 'tool':
                if genre_metrics.get('command_output_coverage', 0) < 0.6:
                    recommendations.append(f"Include more tool command outputs in {genre} genre.")
                if genre_metrics.get('tool_diversity', 0) < 10:
                    recommendations.append(f"Increase diversity of tools covered in {genre} genre.")
            
            elif genre == 'prose':
                if genre_metrics.get('entity_coverage', 0) < 0.7:
                    recommendations.append(f"Improve entity recognition and coverage in {genre} genre.")
                if genre_metrics.get('topic_diversity', 0) < 20:
                    recommendations.append(f"Expand topic coverage in {genre} genre.")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def test_quality_metrics():
    """Test the quality metrics calculator."""
    calculator = QualityMetricsCalculator()
    
    # Create sample data structure
    sample_data = {
        'code': [
            {
                'session_id': 'code_session_1',
                'turn': 0,
                'role': 'user',
                'text': 'How do I implement binary search?',
                'ts': '2024-01-01T10:00:00Z',
                'meta': {
                    'tags': ['python', 'algorithm'],
                    'source': 'stackoverflow'
                }
            },
            {
                'session_id': 'code_session_1',
                'turn': 1,
                'role': 'assistant',
                'text': '```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    return -1\n```',
                'ts': '2024-01-01T10:01:00Z',
                'meta': {
                    'is_accepted': True,
                    'language': 'python',
                    'source': 'stackoverflow'
                }
            }
        ],
        'tool': [
            {
                'session_id': 'tool_session_1',
                'turn': 0,
                'role': 'user',
                'text': 'How do I check disk usage?',
                'ts': '2024-01-01T11:00:00Z',
                'meta': {'source': 'tutorial'}
            },
            {
                'session_id': 'tool_session_1',
                'turn': 1,
                'role': 'assistant',
                'text': '$ df -h\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/sda1        20G   15G  4.2G  79% /',
                'ts': '2024-01-01T11:01:00Z',
                'meta': {
                    'tool_name': 'df',
                    'source': 'tutorial'
                }
            }
        ]
    }
    
    print("Testing basic statistics calculation...")
    basic_stats = calculator._calculate_basic_statistics(sample_data)
    print(f"Total sessions: {basic_stats['total_sessions']}")
    print(f"Total turns: {basic_stats['total_turns']}")
    print(f"Average turns per session: {basic_stats['avg_turns_per_session']:.1f}")
    
    print("\nTesting distributions calculation...")
    distributions = calculator._calculate_distributions(sample_data)
    print(f"Role distribution: {dict(distributions['roles'])}")
    print(f"Genre distribution: {dict(distributions['genres'])}")
    
    print("\nTesting diversity metrics...")
    diversity = calculator._calculate_diversity_metrics(sample_data)
    print(f"Vocabulary size: {diversity['vocabulary_size']}")
    print(f"Unique sessions ratio: {diversity['unique_sessions_ratio']:.3f}")
    
    print("\nTesting genre-specific metrics...")
    code_metrics = calculator._calculate_code_metrics(sample_data['code'])
    print(f"Code metrics: {code_metrics}")
    
    tool_metrics = calculator._calculate_tool_metrics(sample_data['tool'])
    print(f"Tool metrics: {tool_metrics}")

if __name__ == "__main__":
    test_quality_metrics()