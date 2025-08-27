# Advanced Usage Guide: Analytics & Custom Metrics

Master advanced analytics, statistical comparisons, and custom metrics to unlock the full potential of your prompt monitoring system.

## 🎯 Overview

This guide covers:

- **Statistical Analysis**: Rigorous comparison methods and significance testing
- **Custom Metrics**: Domain-specific quality measurements
- **Performance Analytics**: Deep-dive into execution patterns and optimization
- **Comparative Analysis**: Multi-dimensional prompt evaluation
- **Automated Insights**: ML-driven pattern recognition and recommendations

## 📊 Advanced Analytics

### Statistical Prompt Comparison

Perform rigorous statistical analysis of prompt variants:

```python
# File: examples/statistical_analysis.py
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from src.monitoring import get_prompt_tracker, compare_prompts
import pandas as pd
import matplotlib.pyplot as plt

class AdvancedPromptAnalytics:
    """Advanced statistical analysis for prompt performance."""
    
    def __init__(self):
        self.tracker = get_prompt_tracker()
    
    def statistical_comparison(self, prompt_ids, metrics=['response_quality_score', 'execution_time_ms']):
        """Perform comprehensive statistical comparison of prompts."""
        
        print("📊 Advanced Statistical Comparison")
        print("=" * 40)
        
        # Collect data for each prompt
        prompt_data = {}
        for prompt_id in prompt_ids:
            executions = self.tracker.get_executions_by_prompt_id(prompt_id, limit=100)
            
            if not executions:
                print(f"⚠️ No data found for prompt: {prompt_id}")
                continue
            
            prompt_data[prompt_id] = {
                'executions': executions,
                'quality_scores': [e.get('response_quality_score', 0) for e in executions if e.get('response_quality_score') is not None],
                'execution_times': [e.get('execution_time_ms', 0) for e in executions],
                'success_rate': sum(1 for e in executions if e.get('success', False)) / len(executions)
            }
        
        # Pairwise statistical tests
        results = {}
        
        for i, prompt_a in enumerate(prompt_ids):
            for j, prompt_b in enumerate(prompt_ids[i+1:], i+1):
                
                if prompt_a not in prompt_data or prompt_b not in prompt_data:
                    continue
                
                comparison_key = f"{prompt_a} vs {prompt_b}"
                results[comparison_key] = {}
                
                # Quality Score Comparison
                scores_a = prompt_data[prompt_a]['quality_scores']
                scores_b = prompt_data[prompt_b]['quality_scores']
                
                if scores_a and scores_b:
                    # T-test for means
                    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    cohens_d = self.calculate_cohens_d(scores_a, scores_b)
                    
                    results[comparison_key]['quality'] = {
                        'mean_a': np.mean(scores_a),
                        'mean_b': np.mean(scores_b),
                        'std_a': np.std(scores_a),
                        'std_b': np.std(scores_b),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'u_statistic': u_stat,
                        'u_p_value': u_p_value,
                        'cohens_d': cohens_d,
                        'effect_size': self.interpret_effect_size(cohens_d),
                        'significant': p_value < 0.05,
                        'practical_significance': abs(cohens_d) > 0.2
                    }
                
                # Performance Comparison
                times_a = prompt_data[prompt_a]['execution_times']
                times_b = prompt_data[prompt_b]['execution_times']
                
                if times_a and times_b:
                    perf_t_stat, perf_p_value = stats.ttest_ind(times_a, times_b)
                    
                    results[comparison_key]['performance'] = {
                        'mean_time_a': np.mean(times_a),
                        'mean_time_b': np.mean(times_b),
                        'median_time_a': np.median(times_a),
                        'median_time_b': np.median(times_b),
                        't_statistic': perf_t_stat,
                        'p_value': perf_p_value,
                        'significant': perf_p_value < 0.05,
                        'faster_prompt': prompt_a if np.mean(times_a) < np.mean(times_b) else prompt_b,
                        'time_improvement': abs(np.mean(times_a) - np.mean(times_b))
                    }
                
                # Success Rate Comparison
                success_a = prompt_data[prompt_a]['success_rate']
                success_b = prompt_data[prompt_b]['success_rate']
                
                results[comparison_key]['reliability'] = {
                    'success_rate_a': success_a,
                    'success_rate_b': success_b,
                    'difference': abs(success_a - success_b),
                    'more_reliable': prompt_a if success_a > success_b else prompt_b
                }
        
        self.print_statistical_results(results)
        return results
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d for effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / s_pooled
    
    def interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def print_statistical_results(self, results):
        """Print formatted statistical results."""
        
        for comparison, data in results.items():
            print(f"\n📈 {comparison}")
            print("-" * len(comparison))
            
            # Quality results
            if 'quality' in data:
                q = data['quality']
                better = "A" if q['mean_a'] > q['mean_b'] else "B"
                print(f"  Quality Scores:")
                print(f"    Mean A: {q['mean_a']:.3f} (±{q['std_a']:.3f})")
                print(f"    Mean B: {q['mean_b']:.3f} (±{q['std_b']:.3f})")
                print(f"    Difference: {abs(q['mean_a'] - q['mean_b']):.3f} (favors {better})")
                print(f"    P-value: {q['p_value']:.4f} ({'significant' if q['significant'] else 'not significant'})")
                print(f"    Effect size: {q['cohens_d']:.3f} ({q['effect_size']})")
            
            # Performance results
            if 'performance' in data:
                p = data['performance']
                print(f"  Performance:")
                print(f"    Mean time A: {p['mean_time_a']:.0f}ms")
                print(f"    Mean time B: {p['mean_time_b']:.0f}ms")
                print(f"    Faster: {p['faster_prompt']} (by {p['time_improvement']:.0f}ms)")
                print(f"    P-value: {p['p_value']:.4f} ({'significant' if p['significant'] else 'not significant'})")
            
            # Reliability results
            if 'reliability' in data:
                r = data['reliability']
                print(f"  Reliability:")
                print(f"    Success rate A: {r['success_rate_a']:.2%}")
                print(f"    Success rate B: {r['success_rate_b']:.2%}")
                print(f"    More reliable: {r['more_reliable']}")
    
    def longitudinal_analysis(self, prompt_id, days_back=30, window_size=7):
        """Analyze prompt performance trends over time."""
        
        from datetime import datetime, timedelta
        
        print(f"📅 Longitudinal Analysis: {prompt_id}")
        print("=" * 40)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get historical data
        executions = self.tracker.get_executions_by_date_range(start_date, end_date, prompt_id=prompt_id)
        
        if not executions:
            print("❌ No data available for longitudinal analysis")
            return None
        
        # Group by time windows
        time_windows = []
        current_date = start_date
        
        while current_date < end_date:
            window_end = current_date + timedelta(days=window_size)
            
            window_executions = [
                e for e in executions 
                if current_date <= datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) < window_end
            ]
            
            if window_executions:
                quality_scores = [e.get('response_quality_score', 0) for e in window_executions if e.get('response_quality_score') is not None]
                execution_times = [e.get('execution_time_ms', 0) for e in window_executions]
                success_rate = sum(1 for e in window_executions if e.get('success', False)) / len(window_executions)
                
                time_windows.append({
                    'start_date': current_date,
                    'end_date': window_end,
                    'execution_count': len(window_executions),
                    'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                    'avg_time': np.mean(execution_times) if execution_times else 0,
                    'success_rate': success_rate
                })
            
            current_date = window_end
        
        if len(time_windows) < 2:
            print("❌ Insufficient data for trend analysis")
            return None
        
        # Analyze trends
        quality_trend = self.calculate_trend([w['avg_quality'] for w in time_windows])
        time_trend = self.calculate_trend([w['avg_time'] for w in time_windows])
        success_trend = self.calculate_trend([w['success_rate'] for w in time_windows])
        
        print(f"📊 Trend Analysis (last {days_back} days):")
        print(f"  Quality Score: {quality_trend['direction']} ({quality_trend['slope']:+.4f}/window)")
        print(f"  Execution Time: {time_trend['direction']} ({time_trend['slope']:+.1f}ms/window)")
        print(f"  Success Rate: {success_trend['direction']} ({success_trend['slope']:+.3f}/window)")
        
        # Detect anomalies
        anomalies = self.detect_anomalies(time_windows)
        if anomalies:
            print(f"\n🚨 Anomalies Detected:")
            for anomaly in anomalies:
                print(f"  {anomaly['date']}: {anomaly['type']} - {anomaly['description']}")
        
        return {
            'windows': time_windows,
            'trends': {
                'quality': quality_trend,
                'performance': time_trend,
                'reliability': success_trend
            },
            'anomalies': anomalies
        }
    
    def calculate_trend(self, values):
        """Calculate trend slope and direction."""
        if len(values) < 2:
            return {'slope': 0, 'direction': 'stable'}
        
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        
        if p_value > 0.05:
            direction = 'stable'
        elif slope > 0:
            direction = 'improving'
        else:
            direction = 'declining'
        
        return {
            'slope': slope,
            'direction': direction,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant': p_value <= 0.05
        }
    
    def detect_anomalies(self, time_windows, threshold_std=2):
        """Detect anomalies in time series data."""
        
        quality_values = [w['avg_quality'] for w in time_windows]
        time_values = [w['avg_time'] for w in time_windows]
        
        quality_mean = np.mean(quality_values)
        quality_std = np.std(quality_values)
        time_mean = np.mean(time_values)
        time_std = np.std(time_values)
        
        anomalies = []
        
        for i, window in enumerate(time_windows):
            # Check quality anomalies
            quality_z = abs(window['avg_quality'] - quality_mean) / quality_std if quality_std > 0 else 0
            if quality_z > threshold_std:
                anomaly_type = 'quality_spike' if window['avg_quality'] > quality_mean else 'quality_drop'
                anomalies.append({
                    'date': window['start_date'].strftime('%Y-%m-%d'),
                    'type': anomaly_type,
                    'description': f"Quality score {window['avg_quality']:.3f} (z-score: {quality_z:.2f})"
                })
            
            # Check performance anomalies
            time_z = abs(window['avg_time'] - time_mean) / time_std if time_std > 0 else 0
            if time_z > threshold_std:
                anomaly_type = 'performance_spike' if window['avg_time'] > time_mean else 'performance_improvement'
                anomalies.append({
                    'date': window['start_date'].strftime('%Y-%m-%d'),
                    'type': anomaly_type,
                    'description': f"Execution time {window['avg_time']:.0f}ms (z-score: {time_z:.2f})"
                })
        
        return anomalies

# Usage examples
def main():
    analyzer = AdvancedPromptAnalytics()
    
    # Example 1: Statistical comparison
    print("🧪 Example 1: Statistical Comparison")
    
    prompt_ids = ["prompt_v1", "prompt_v2", "prompt_v3"]
    comparison_results = analyzer.statistical_comparison(prompt_ids)
    
    # Example 2: Longitudinal analysis
    print(f"\n🧪 Example 2: Longitudinal Analysis")
    
    longitudinal_results = analyzer.longitudinal_analysis("main_prompt", days_back=14, window_size=2)
    
    if longitudinal_results:
        print(f"📈 Analyzed {len(longitudinal_results['windows'])} time windows")
        for trend_name, trend_data in longitudinal_results['trends'].items():
            print(f"  {trend_name.title()}: {trend_data['direction']} trend (R²={trend_data['r_squared']:.3f})")

if __name__ == "__main__":
    main()
```

### Custom Quality Metrics

Create domain-specific quality assessment:

```python
# File: examples/custom_metrics.py
import re
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from src.monitoring import track_prompt

@dataclass
class QualityMetric:
    """Definition of a custom quality metric."""
    name: str
    description: str
    min_score: float = 0.0
    max_score: float = 1.0
    weight: float = 1.0

class CustomQualityAssessor:
    """Framework for creating domain-specific quality metrics."""
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.metrics = {}
        self.setup_domain_metrics()
    
    def setup_domain_metrics(self):
        """Setup metrics based on domain."""
        
        if self.domain == "research":
            self.add_research_metrics()
        elif self.domain == "summarization":
            self.add_summarization_metrics()
        elif self.domain == "code":
            self.add_code_metrics()
        else:
            self.add_general_metrics()
    
    def add_research_metrics(self):
        """Add research-specific quality metrics."""
        
        self.metrics = {
            'citation_accuracy': QualityMetric(
                "Citation Accuracy",
                "Accuracy and proper formatting of citations",
                weight=0.25
            ),
            'factual_consistency': QualityMetric(
                "Factual Consistency", 
                "Consistency with established facts and sources",
                weight=0.30
            ),
            'methodology_soundness': QualityMetric(
                "Methodology Soundness",
                "Quality of research methodology and approach",
                weight=0.25
            ),
            'clarity_coherence': QualityMetric(
                "Clarity and Coherence",
                "Logical flow and clear presentation of ideas",
                weight=0.20
            )
        }
    
    def add_summarization_metrics(self):
        """Add summarization-specific quality metrics."""
        
        self.metrics = {
            'coverage': QualityMetric(
                "Content Coverage",
                "How well the summary covers the main points",
                weight=0.30
            ),
            'conciseness': QualityMetric(
                "Conciseness",
                "Efficiency in conveying information",
                weight=0.25
            ),
            'coherence': QualityMetric(
                "Coherence",
                "Logical flow and readability",
                weight=0.25
            ),
            'faithfulness': QualityMetric(
                "Faithfulness",
                "Accuracy to the original content",
                weight=0.20
            )
        }
    
    def add_code_metrics(self):
        """Add code-specific quality metrics."""
        
        self.metrics = {
            'correctness': QualityMetric(
                "Code Correctness",
                "Syntactic and logical correctness",
                weight=0.35
            ),
            'readability': QualityMetric(
                "Code Readability",
                "Clarity and maintainability",
                weight=0.25
            ),
            'efficiency': QualityMetric(
                "Code Efficiency",
                "Performance and resource usage",
                weight=0.20
            ),
            'best_practices': QualityMetric(
                "Best Practices",
                "Adherence to coding standards",
                weight=0.20
            )
        }
    
    def add_general_metrics(self):
        """Add general-purpose quality metrics."""
        
        self.metrics = {
            'relevance': QualityMetric(
                "Relevance",
                "How well the response addresses the prompt",
                weight=0.30
            ),
            'accuracy': QualityMetric(
                "Accuracy",
                "Factual correctness of the response",
                weight=0.25
            ),
            'completeness': QualityMetric(
                "Completeness",
                "Thoroughness of the response",
                weight=0.25
            ),
            'clarity': QualityMetric(
                "Clarity",
                "Understandability and coherence",
                weight=0.20
            )
        }
    
    def assess_quality(self, prompt_text: str, response_text: str, context: Dict = None) -> Dict[str, float]:
        """Assess quality using custom metrics."""
        
        scores = {}
        context = context or {}
        
        for metric_name, metric in self.metrics.items():
            method_name = f"assess_{metric_name}"
            
            if hasattr(self, method_name):
                score = getattr(self, method_name)(prompt_text, response_text, context)
                scores[metric_name] = max(metric.min_score, min(metric.max_score, score))
            else:
                # Default scoring if specific method not implemented
                scores[metric_name] = 0.7  # Neutral score
        
        return scores
    
    def calculate_overall_score(self, metric_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        
        total_weight = sum(metric.weight for metric in self.metrics.values())
        weighted_sum = sum(
            score * self.metrics[metric_name].weight 
            for metric_name, score in metric_scores.items()
            if metric_name in self.metrics
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Research domain assessments
    def assess_citation_accuracy(self, prompt: str, response: str, context: Dict) -> float:
        """Assess citation accuracy for research responses."""
        
        # Count citations
        citation_patterns = [
            r'\[[0-9]+\]',          # [1], [2], etc.
            r'\([^)]*\d{4}[^)]*\)', # (Author, 2023)
            r'\b\w+\s+et\s+al\.',   # Author et al.
        ]
        
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, response))
        
        # Assess based on response length and citation density
        response_length = len(response.split())
        if response_length < 50:
            # Short responses don't need many citations
            return 0.8 if citations_found > 0 else 0.6
        
        # For longer responses, expect more citations
        expected_citations = response_length // 100  # 1 citation per 100 words
        citation_ratio = citations_found / max(1, expected_citations)
        
        return min(1.0, 0.3 + 0.7 * citation_ratio)
    
    def assess_factual_consistency(self, prompt: str, response: str, context: Dict) -> float:
        """Assess factual consistency."""
        
        # Basic fact checking heuristics
        score = 0.7  # Base score
        
        # Look for confidence indicators
        confidence_phrases = [
            'according to', 'research shows', 'studies indicate',
            'evidence suggests', 'data shows', 'findings reveal'
        ]
        
        confidence_count = sum(1 for phrase in confidence_phrases if phrase in response.lower())
        confidence_bonus = min(0.2, confidence_count * 0.05)
        
        # Penalize uncertain language without qualification
        uncertain_phrases = ['maybe', 'perhaps', 'possibly', 'might be']
        uncertainty_count = sum(1 for phrase in uncertain_phrases if phrase in response.lower())
        uncertainty_penalty = min(0.3, uncertainty_count * 0.1)
        
        return max(0.0, score + confidence_bonus - uncertainty_penalty)
    
    def assess_methodology_soundness(self, prompt: str, response: str, context: Dict) -> float:
        """Assess research methodology soundness."""
        
        methodology_indicators = [
            'methodology', 'method', 'approach', 'procedure',
            'systematic', 'controlled', 'randomized', 'sample size',
            'statistical', 'analysis', 'hypothesis', 'variable'
        ]
        
        methodology_mentions = sum(1 for term in methodology_indicators if term in response.lower())
        
        # Score based on methodology discussion depth
        if methodology_mentions >= 3:
            return 0.9
        elif methodology_mentions >= 2:
            return 0.7
        elif methodology_mentions >= 1:
            return 0.5
        else:
            return 0.3
    
    def assess_clarity_coherence(self, prompt: str, response: str, context: Dict) -> float:
        """Assess clarity and coherence."""
        
        # Basic readability assessment
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Optimal sentence length is 15-20 words
        length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        length_score = max(0.3, min(1.0, length_score))
        
        # Look for transition words
        transitions = [
            'however', 'therefore', 'furthermore', 'additionally',
            'consequently', 'moreover', 'nevertheless', 'meanwhile'
        ]
        
        transition_count = sum(1 for word in transitions if word in response.lower())
        transition_bonus = min(0.2, transition_count * 0.05)
        
        return min(1.0, length_score + transition_bonus)
    
    # Summarization domain assessments
    def assess_coverage(self, prompt: str, response: str, context: Dict) -> float:
        """Assess content coverage for summarization."""
        
        source_text = context.get('source_text', '')
        if not source_text:
            return 0.5  # Can't assess without source
        
        # Extract key terms from source
        source_words = set(word.lower() for word in source_text.split() if len(word) > 3)
        summary_words = set(word.lower() for word in response.split())
        
        # Calculate coverage ratio
        coverage_ratio = len(source_words.intersection(summary_words)) / len(source_words) if source_words else 0
        
        return min(1.0, coverage_ratio * 2)  # Scale to 0-1
    
    def assess_conciseness(self, prompt: str, response: str, context: Dict) -> float:
        """Assess conciseness of summary."""
        
        source_text = context.get('source_text', '')
        if not source_text:
            return 0.5
        
        source_length = len(source_text.split())
        summary_length = len(response.split())
        
        compression_ratio = summary_length / source_length if source_length > 0 else 1
        
        # Optimal compression ratio is 0.1-0.3 (10-30% of original)
        if 0.1 <= compression_ratio <= 0.3:
            return 1.0
        elif compression_ratio < 0.1:
            return 0.7  # Too brief
        elif compression_ratio < 0.5:
            return 0.8  # Good compression
        else:
            return max(0.3, 1.0 - compression_ratio)  # Too long

class QualityTrackingWorkflow:
    """Workflow that integrates custom quality assessment with monitoring."""
    
    def __init__(self, domain: str = "general"):
        self.assessor = CustomQualityAssessor(domain)
    
    def process_with_quality_tracking(self, prompt_id: str, prompt_text: str, 
                                    model_config: Dict, context: Dict = None):
        """Process prompt with custom quality tracking."""
        
        with track_prompt(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            model_config=model_config,
            tags=[f"domain-{self.assessor.domain}", "custom-quality"]
        ) as execution:
            
            # Simulate LLM processing
            response = self.simulate_llm_response(prompt_text, model_config)
            
            # Assess quality using custom metrics
            metric_scores = self.assessor.assess_quality(prompt_text, response, context)
            overall_score = self.assessor.calculate_overall_score(metric_scores)
            
            # Update execution with results
            execution.response_text = response
            execution.response_quality_score = overall_score
            execution.success = True
            
            # Store detailed metric scores in metadata
            execution.metadata.update({
                'custom_metrics': metric_scores,
                'domain': self.assessor.domain,
                'metric_weights': {name: metric.weight for name, metric in self.assessor.metrics.items()}
            })
            
            return {
                'response': response,
                'overall_score': overall_score,
                'metric_scores': metric_scores,
                'execution_id': execution.execution_id
            }
    
    def simulate_llm_response(self, prompt: str, config: Dict) -> str:
        """Simulate LLM response (replace with actual LLM call)."""
        
        # This is a placeholder - replace with actual LLM integration
        if self.assessor.domain == "research":
            return f"Based on recent studies, the research on '{prompt[:30]}...' shows significant progress. According to Smith et al. (2023), the methodology involves systematic analysis of variables. The findings indicate strong evidence for the proposed hypothesis."
        elif self.assessor.domain == "summarization":
            return f"Summary: The main points regarding '{prompt[:30]}...' include three key aspects: methodology, results, and implications. The approach demonstrates effectiveness in addressing the core requirements."
        else:
            return f"Response to '{prompt[:30]}...': This addresses the key aspects of your question with relevant information and practical insights."

# Usage examples
def main():
    print("🎯 Custom Quality Metrics Examples")
    print("=" * 40)
    
    # Example 1: Research domain
    research_workflow = QualityTrackingWorkflow("research")
    
    research_result = research_workflow.process_with_quality_tracking(
        prompt_id="research_methodology_analysis",
        prompt_text="Analyze the effectiveness of machine learning approaches in healthcare diagnostics",
        model_config={"model": "gpt-4", "temperature": 0.3},
        context={"research_area": "healthcare", "methodology_focus": True}
    )
    
    print("📊 Research Quality Assessment:")
    print(f"  Overall Score: {research_result['overall_score']:.3f}")
    for metric, score in research_result['metric_scores'].items():
        print(f"  {metric}: {score:.3f}")
    
    # Example 2: Summarization domain
    summarization_workflow = QualityTrackingWorkflow("summarization")
    
    source_text = "Machine learning has revolutionized healthcare diagnostics by enabling more accurate and faster analysis of medical images, patient data, and clinical records. Recent studies show that ML algorithms can achieve diagnostic accuracy comparable to or exceeding human experts in specific domains such as radiology and pathology. However, challenges remain in terms of data quality, algorithm bias, and regulatory approval processes."
    
    summarization_result = summarization_workflow.process_with_quality_tracking(
        prompt_id="healthcare_ml_summary",
        prompt_text="Summarize the impact of machine learning on healthcare diagnostics",
        model_config={"model": "gpt-3.5-turbo", "temperature": 0.2},
        context={"source_text": source_text}
    )
    
    print(f"\n📊 Summarization Quality Assessment:")
    print(f"  Overall Score: {summarization_result['overall_score']:.3f}")
    for metric, score in summarization_result['metric_scores'].items():
        print(f"  {metric}: {score:.3f}")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Customize metrics for your specific domain")
    print(f"  2. Implement domain-specific assessment methods")
    print(f"  3. Use custom metrics in A/B testing")
    print(f"  4. Analyze metric correlations and weights")

if __name__ == "__main__":
    main()
```

## 🔍 Performance Deep-Dive Analytics

### Execution Pattern Analysis

Analyze detailed execution patterns for optimization:

```python
# File: examples/performance_deep_dive.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from src.monitoring import get_prompt_tracker

class PerformanceAnalyzer:
    """Deep-dive performance analysis and optimization recommendations."""
    
    def __init__(self):
        self.tracker = get_prompt_tracker()
    
    def analyze_execution_patterns(self, days_back=7, min_executions=10):
        """Analyze execution patterns for performance optimization."""
        
        print("🔍 Execution Pattern Analysis")
        print("=" * 30)
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        executions = self.tracker.get_executions_by_date_range(start_date, end_date)
        
        if len(executions) < min_executions:
            print(f"❌ Insufficient data: {len(executions)} executions (need {min_executions}+)")
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(executions)
        
        # Basic statistics
        successful = df[df['success'] == True]
        failed = df[df['success'] == False]
        
        print(f"📊 Dataset Overview:")
        print(f"  Total executions: {len(df)}")
        print(f"  Successful: {len(successful)} ({len(successful)/len(df):.1%})")
        print(f"  Failed: {len(failed)} ({len(failed)/len(df):.1%})")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        
        # Performance analysis
        if len(successful) > 0:
            execution_times = successful['execution_time_ms'].dropna()
            
            if len(execution_times) > 0:
                print(f"\n⚡ Performance Statistics:")
                print(f"  Mean: {execution_times.mean():.0f}ms")
                print(f"  Median: {execution_times.median():.0f}ms")
                print(f"  P95: {execution_times.quantile(0.95):.0f}ms")
                print(f"  P99: {execution_times.quantile(0.99):.0f}ms")
                print(f"  Min: {execution_times.min():.0f}ms")
                print(f"  Max: {execution_times.max():.0f}ms")
                
                # Identify slow executions
                slow_threshold = execution_times.quantile(0.9)
                slow_executions = successful[successful['execution_time_ms'] > slow_threshold]
                
                print(f"\n🐌 Slow Executions (>{slow_threshold:.0f}ms):")
                print(f"  Count: {len(slow_executions)}")
                
                if len(slow_executions) > 0:
                    # Analyze slow execution patterns
                    slow_prompts = slow_executions['prompt_id'].value_counts().head(5)
                    print(f"  Top slow prompt IDs:")
                    for prompt_id, count in slow_prompts.items():
                        avg_time = slow_executions[slow_executions['prompt_id'] == prompt_id]['execution_time_ms'].mean()
                        print(f"    {prompt_id}: {count} executions, {avg_time:.0f}ms avg")
        
        # Quality analysis
        quality_scores = successful['response_quality_score'].dropna()
        if len(quality_scores) > 0:
            print(f"\n🎯 Quality Statistics:")
            print(f"  Mean quality: {quality_scores.mean():.3f}")
            print(f"  Median quality: {quality_scores.median():.3f}")
            print(f"  Min quality: {quality_scores.min():.3f}")
            print(f"  Max quality: {quality_scores.max():.3f}")
            
            # Identify low-quality executions
            low_quality_threshold = 0.5
            low_quality = successful[successful['response_quality_score'] < low_quality_threshold]
            
            if len(low_quality) > 0:
                print(f"\n📉 Low Quality Executions (<{low_quality_threshold}):")
                print(f"  Count: {len(low_quality)} ({len(low_quality)/len(successful):.1%})")
                
                low_quality_prompts = low_quality['prompt_id'].value_counts().head(3)
                for prompt_id, count in low_quality_prompts.items():
                    avg_quality = low_quality[low_quality['prompt_id'] == prompt_id]['response_quality_score'].mean()
                    print(f"    {prompt_id}: {count} executions, {avg_quality:.3f} avg quality")
        
        # Token usage analysis
        token_usage = successful['tokens_used'].dropna()
        if len(token_usage) > 0:
            print(f"\n🔢 Token Usage Statistics:")
            print(f"  Mean tokens: {token_usage.mean():.0f}")
            print(f"  Median tokens: {token_usage.median():.0f}")
            print(f"  Total tokens: {token_usage.sum():,}")
            
            # Token efficiency (quality per token)
            if len(quality_scores) > 0:
                efficiency_data = successful[['tokens_used', 'response_quality_score']].dropna()
                if len(efficiency_data) > 0:
                    efficiency_data['efficiency'] = efficiency_data['response_quality_score'] / efficiency_data['tokens_used']
                    print(f"  Quality per token: {efficiency_data['efficiency'].mean():.6f}")
        
        return {
            'total_executions': len(df),
            'success_rate': len(successful) / len(df),
            'performance_stats': execution_times.describe().to_dict() if len(execution_times) > 0 else {},
            'quality_stats': quality_scores.describe().to_dict() if len(quality_scores) > 0 else {},
            'token_stats': token_usage.describe().to_dict() if len(token_usage) > 0 else {}
        }
    
    def model_performance_comparison(self):
        """Compare performance across different models."""
        
        print("\n🤖 Model Performance Comparison")
        print("=" * 30)
        
        # Get recent executions grouped by model
        executions = self.tracker.get_recent_executions(limit=500)
        
        model_stats = defaultdict(lambda: {
            'count': 0,
            'success_count': 0,
            'total_time': 0,
            'total_tokens': 0,
            'quality_scores': []
        })
        
        for execution in executions:
            model_config = execution.get('model_config', {})
            model = model_config.get('model', 'unknown')
            
            stats = model_stats[model]
            stats['count'] += 1
            
            if execution.get('success', False):
                stats['success_count'] += 1
                stats['total_time'] += execution.get('execution_time_ms', 0)
                stats['total_tokens'] += execution.get('tokens_used', 0)
                
                quality = execution.get('response_quality_score')
                if quality is not None:
                    stats['quality_scores'].append(quality)
        
        # Calculate averages and display results
        for model, stats in model_stats.items():
            if stats['count'] < 5:  # Skip models with too few executions
                continue
            
            success_rate = stats['success_count'] / stats['count']
            avg_time = stats['total_time'] / stats['success_count'] if stats['success_count'] > 0 else 0
            avg_tokens = stats['total_tokens'] / stats['success_count'] if stats['success_count'] > 0 else 0
            avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores']) if stats['quality_scores'] else 0
            
            print(f"\n📊 {model}:")
            print(f"  Executions: {stats['count']}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Avg time: {avg_time:.0f}ms")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg quality: {avg_quality:.3f}")
            
            # Calculate efficiency metrics
            if avg_time > 0 and avg_tokens > 0:
                tokens_per_second = (avg_tokens / avg_time) * 1000
                quality_per_second = (avg_quality / avg_time) * 1000
                print(f"  Tokens/sec: {tokens_per_second:.1f}")
                print(f"  Quality/sec: {quality_per_second:.4f}")
    
    def identify_optimization_opportunities(self):
        """Identify specific optimization opportunities."""
        
        print("\n🚀 Optimization Opportunities")
        print("=" * 30)
        
        opportunities = []
        
        # Get recent execution data
        executions = self.tracker.get_recent_executions(limit=200)
        
        if not executions:
            print("❌ No data available for optimization analysis")
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(executions)
        successful = df[df['success'] == True]
        
        if len(successful) == 0:
            print("❌ No successful executions for analysis")
            return []
        
        # Opportunity 1: High-latency prompts
        execution_times = successful['execution_time_ms'].dropna()
        if len(execution_times) > 0:
            p95_time = execution_times.quantile(0.95)
            slow_prompts = successful[successful['execution_time_ms'] > p95_time]['prompt_id'].value_counts()
            
            if len(slow_prompts) > 0:
                opportunities.append({
                    'type': 'High Latency',
                    'description': f"Optimize {len(slow_prompts)} prompt types with >P95 latency ({p95_time:.0f}ms)",
                    'impact': 'High',
                    'prompts': slow_prompts.head(3).to_dict()
                })
        
        # Opportunity 2: Low-quality prompts
        quality_scores = successful['response_quality_score'].dropna()
        if len(quality_scores) > 0:
            low_quality_threshold = quality_scores.quantile(0.25)
            low_quality_prompts = successful[successful['response_quality_score'] < low_quality_threshold]['prompt_id'].value_counts()
            
            if len(low_quality_prompts) > 0:
                opportunities.append({
                    'type': 'Low Quality',
                    'description': f"Improve {len(low_quality_prompts)} prompt types with <Q25 quality ({low_quality_threshold:.3f})",
                    'impact': 'Medium',
                    'prompts': low_quality_prompts.head(3).to_dict()
                })
        
        # Opportunity 3: Token inefficiency
        if 'tokens_used' in successful.columns:
            token_usage = successful['tokens_used'].dropna()
            if len(token_usage) > 0 and len(quality_scores) > 0:
                # Calculate efficiency for prompts with both metrics
                efficiency_data = successful[['prompt_id', 'tokens_used', 'response_quality_score']].dropna()
                if len(efficiency_data) > 0:
                    efficiency_data['efficiency'] = efficiency_data['response_quality_score'] / efficiency_data['tokens_used']
                    
                    # Find prompts with low efficiency
                    prompt_efficiency = efficiency_data.groupby('prompt_id')['efficiency'].mean()
                    low_efficiency = prompt_efficiency[prompt_efficiency < prompt_efficiency.quantile(0.25)]
                    
                    if len(low_efficiency) > 0:
                        opportunities.append({
                            'type': 'Token Inefficiency',
                            'description': f"Optimize {len(low_efficiency)} prompts with low quality-per-token ratio",
                            'impact': 'Medium',
                            'prompts': low_efficiency.head(3).to_dict()
                        })
        
        # Opportunity 4: Error-prone prompts
        failed = df[df['success'] == False]
        if len(failed) > 0:
            error_prone_prompts = failed['prompt_id'].value_counts()
            
            if len(error_prone_prompts) > 0:
                opportunities.append({
                    'type': 'Error Prone',
                    'description': f"Fix {len(error_prone_prompts)} prompt types with high error rates",
                    'impact': 'High',
                    'prompts': error_prone_prompts.head(3).to_dict()
                })
        
        # Display opportunities
        if opportunities:
            for i, opp in enumerate(opportunities, 1):
                print(f"\n{i}. {opp['type']} ({opp['impact']} Impact)")
                print(f"   {opp['description']}")
                print(f"   Top prompts:")
                for prompt_id, count in list(opp['prompts'].items())[:3]:
                    print(f"     - {prompt_id}: {count} occurrences")
        else:
            print("✅ No major optimization opportunities identified")
        
        return opportunities
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        
        print("📋 Performance Optimization Report")
        print("=" * 40)
        
        # Run all analyses
        pattern_analysis = self.analyze_execution_patterns()
        self.model_performance_comparison()
        opportunities = self.identify_optimization_opportunities()
        
        # Generate recommendations
        print(f"\n💡 Optimization Recommendations")
        print("-" * 30)
        
        recommendations = []
        
        if pattern_analysis and pattern_analysis.get('success_rate', 1) < 0.95:
            recommendations.append("🔧 Investigate and fix high error rate prompts")
        
        if opportunities:
            high_impact_opps = [opp for opp in opportunities if opp['impact'] == 'High']
            if high_impact_opps:
                recommendations.append("⚡ Prioritize high-impact optimizations (latency & errors)")
            
            medium_impact_opps = [opp for opp in opportunities if opp['impact'] == 'Medium']
            if medium_impact_opps:
                recommendations.append("🎯 Address quality and efficiency improvements")
        
        if pattern_analysis:
            performance_stats = pattern_analysis.get('performance_stats', {})
            if performance_stats and performance_stats.get('75%', 0) > 2000:  # P75 > 2 seconds
                recommendations.append("🚀 Implement caching for slow prompts")
        
        recommendations.append("📊 Set up automated performance monitoring")
        recommendations.append("🔄 Establish regular optimization review cycles")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return {
            'pattern_analysis': pattern_analysis,
            'opportunities': opportunities,
            'recommendations': recommendations
        }

def main():
    analyzer = PerformanceAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_optimization_report()
    
    print(f"\n🎉 Performance analysis complete!")
    print(f"📄 Found {len(report['opportunities'])} optimization opportunities")
    print(f"📝 Generated {len(report['recommendations'])} recommendations")

if __name__ == "__main__":
    main()
```

This advanced usage guide provides sophisticated analytics capabilities that go far beyond basic monitoring. The examples show how to perform rigorous statistical comparisons, create domain-specific quality metrics, and conduct deep performance analysis.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create tutorial directory structure with organized learning materials", "status": "completed", "activeForm": "Creating tutorial directory structure with organized learning materials"}, {"content": "Write Quick Start Tutorial with installation and basic usage", "status": "completed", "activeForm": "Writing Quick Start Tutorial with installation and basic usage"}, {"content": "Create Integration Tutorials for Lethe workflows and MLflow", "status": "completed", "activeForm": "Creating Integration Tutorials for Lethe workflows and MLflow"}, {"content": "Develop Advanced Usage Examples with statistics and custom analytics", "status": "completed", "activeForm": "Developing Advanced Usage Examples with statistics and custom analytics"}, {"content": "Build Dashboard Tutorial with navigation and features guide", "status": "in_progress", "activeForm": "Building Dashboard Tutorial with navigation and features guide"}, {"content": "Write CLI Reference Guide with command examples and automation", "status": "pending", "activeForm": "Writing CLI Reference Guide with command examples and automation"}, {"content": "Create Development Examples for extending the system", "status": "pending", "activeForm": "Creating Development Examples for extending the system"}]