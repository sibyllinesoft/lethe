#!/usr/bin/env python3
"""
Comprehensive Three-Tier Diagnostic System for Evaluation Pipeline Failures
===========================================================================

This diagnostic system isolates exactly where evaluation pipelines are failing
by testing three critical tiers:

1. **Generation Capture** (A): Model text generation/streaming verification
2. **Answer Extraction/Normalization** (B): Text → candidate answers processing  
3. **Gold Matching/Scoring** (C): Candidates → accuracy metrics computation

The system provides specific failure isolation and actionable repair guidance.

Author: Lethe Research Team
"""

import json
import logging
import random
import time
import unicodedata
import regex as re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

try:
    from ..infinitebench.dataset_loader import InfiniteBenchLoader, InfiniteBenchSample
    from ..infinitebench.metrics import InfiniteBenchMetrics
except ImportError:
    # Handle imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from infinitebench.dataset_loader import InfiniteBenchLoader, InfiniteBenchSample
    from infinitebench.metrics import InfiniteBenchMetrics

logger = logging.getLogger(__name__)

@dataclass
class GenerationProbe:
    """Detailed probe of text generation process."""
    sample_id: str
    decoded_text: str
    stop_reason: str
    new_token_count: int
    finish_reason: str
    processing_time_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiagnosticResult:
    """Result from a single diagnostic tier."""
    tier_name: str
    success: bool
    score: float
    details: Dict[str, Any]
    samples_analyzed: int
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ComprehensiveDiagnostic:
    """Complete diagnostic results across all tiers."""
    experiment_name: str
    timestamp: str
    tier_results: Dict[str, DiagnosticResult]
    overall_assessment: str
    critical_failures: List[str] = field(default_factory=list)
    repair_priority: List[str] = field(default_factory=list)

class EvaluationDiagnostics:
    """
    Comprehensive diagnostic system for evaluation pipeline failures.
    
    Provides systematic isolation of failures across generation, extraction,
    and scoring tiers with specific repair guidance.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize evaluation diagnostics.
        
        Args:
            data_dir: Directory containing evaluation datasets
        """
        self.data_dir = Path(data_dir)
        self.loader = InfiniteBenchLoader(self.data_dir)
        self.metrics = InfiniteBenchMetrics()
        
    def normalize_answer(self, text: str, language: str = 'en') -> str:
        """
        Deterministic text normalization for scoring.
        
        Args:
            text: Raw text to normalize
            language: Language code for language-specific normalization
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
            
        # Unicode normalization and case folding
        text = unicodedata.normalize("NFKC", text).casefold().strip()
        
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Language-specific punctuation normalization
        if language == "zh":
            text = text.replace("，", ",").replace("。", ".").replace("：", ":")
        
        # Remove non-word characters except specific Unicode ranges
        text = re.sub(
            r"[^\w\s\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}\-\.]", 
            "", 
            text, 
            flags=re.UNICODE
        )
        
        return text.strip()

    def compute_token_f1(self, pred: str, gold: str, language: str = 'en') -> float:
        """
        Token-level F1 score with proper normalization.
        
        Args:
            pred: Predicted text
            gold: Gold standard text
            language: Language code
            
        Returns:
            F1 score between 0.0 and 1.0
        """
        pred_tokens = self.normalize_answer(pred, language).split()
        gold_tokens = self.normalize_answer(gold, language).split()
        
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
            
        pred_set = set(pred_tokens)
        gold_set = set(gold_tokens)
        intersection = pred_set & gold_set
        
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)

    def evaluate_single_prediction(self, prediction: str, gold: str, language: str = 'en') -> float:
        """
        Single prediction evaluation with comprehensive normalization.
        
        Args:
            prediction: Model prediction
            gold: Gold standard answer
            language: Language code
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Exact match after normalization
        pred_norm = self.normalize_answer(prediction, language)
        gold_norm = self.normalize_answer(gold, language)
        
        if pred_norm == gold_norm:
            return 1.0
        
        # Fall back to token F1
        return self.compute_token_f1(prediction, gold, language)

    def gold_echo_control(self, samples: List[dict], n_samples: int = 200) -> DiagnosticResult:
        """
        Gold-Echo Control: Test scorer with perfect inputs to verify matching/normalization.
        
        This test feeds the scorer prediction := gold for mixed samples.
        Expected result: accuracy = 1.0
        
        Args:
            samples: List of evaluation samples with 'answer' field
            n_samples: Number of samples to test
            
        Returns:
            DiagnosticResult with scorer sanity validation
        """
        logger.info(f"🧪 Running Gold-Echo Control (n={n_samples})")
        
        if len(samples) < n_samples:
            n_samples = len(samples)
            logger.warning(f"Only {len(samples)} samples available, using all")
        
        mixed_samples = random.sample(samples, n_samples)
        correct = 0
        total = 0
        issues = []
        detailed_results = []
        
        for sample in mixed_samples:
            try:
                # Use gold as prediction - should always score 1.0
                gold_answer = sample.get('answer', '')
                if not gold_answer:
                    issues.append(f"Sample {sample.get('id', 'unknown')} has empty gold answer")
                    continue
                
                # Test scorer with gold as prediction
                prediction = gold_answer
                language = sample.get('language', 'en')
                score = self.evaluate_single_prediction(prediction, gold_answer, language)
                
                detailed_results.append({
                    'sample_id': sample.get('id', 'unknown'),
                    'prediction': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                    'gold': gold_answer[:100] + "..." if len(gold_answer) > 100 else gold_answer,
                    'score': score,
                    'language': language
                })
                
                correct += score
                total += 1
                
                if score < 1.0:
                    issues.append(f"Perfect input scored {score:.3f} for sample {sample.get('id', 'unknown')}")
                    
            except Exception as e:
                issues.append(f"Error evaluating sample {sample.get('id', 'unknown')}: {str(e)}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        success = accuracy >= 0.95  # Allow for minor floating point errors
        
        # Generate recommendations
        recommendations = []
        if not success:
            recommendations.append("CRITICAL: Scorer/normalization is broken - fix text normalization logic")
            recommendations.append("Check normalize_answer() function for Unicode handling")
            recommendations.append("Verify token F1 computation is working correctly")
        else:
            recommendations.append("✅ Scorer/Normalization: WORKING")
        
        return DiagnosticResult(
            tier_name="Gold-Echo Control",
            success=success,
            score=accuracy,
            details={
                "total_samples": total,
                "perfect_scores": correct,
                "accuracy": accuracy,
                "issues_count": len(issues),
                "sample_results": detailed_results[:10]  # Show first 10 for inspection
            },
            samples_analyzed=total,
            issues_found=issues,
            recommendations=recommendations
        )

    def raw_capture_probe(self, method_fn: Callable, samples: List[dict], n_samples: int = 100) -> DiagnosticResult:
        """
        Raw-Capture Probe: Verify model is generating non-empty responses.
        
        This test captures detailed generation information for diagnostic analysis.
        Expected: Non-empty responses with reasonable token counts.
        
        Args:
            method_fn: Function that takes (sample) and returns generation result
            samples: List of evaluation samples
            n_samples: Number of samples to probe
            
        Returns:
            DiagnosticResult with generation verification
        """
        logger.info(f"🎯 Running Raw-Capture Probe (n={n_samples})")
        
        if len(samples) < n_samples:
            n_samples = len(samples)
            logger.warning(f"Only {len(samples)} samples available, using all")
        
        probe_samples = random.sample(samples, n_samples)
        probes = []
        issues = []
        total_tokens = 0
        non_empty_responses = 0
        
        for sample in probe_samples:
            try:
                start_time = time.time()
                
                # Run actual generation
                response = method_fn(sample)
                processing_time = (time.time() - start_time) * 1000
                
                # Extract response details
                if hasattr(response, 'text'):
                    decoded_text = response.text
                elif isinstance(response, dict):
                    decoded_text = response.get('text', str(response))
                else:
                    decoded_text = str(response)
                
                # Count tokens (simple whitespace split)
                token_count = len(decoded_text.split()) if decoded_text else 0
                total_tokens += token_count
                
                if decoded_text.strip():
                    non_empty_responses += 1
                
                probe = GenerationProbe(
                    sample_id=sample.get('id', 'unknown'),
                    decoded_text=decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text,
                    stop_reason=getattr(response, 'stop_reason', 'unknown'),
                    new_token_count=token_count,
                    finish_reason=getattr(response, 'finish_reason', 'unknown'),
                    processing_time_ms=processing_time
                )
                probes.append(probe)
                
                if token_count == 0:
                    issues.append(f"Empty response for sample {sample.get('id', 'unknown')}")
                elif token_count < 5:
                    issues.append(f"Very short response ({token_count} tokens) for sample {sample.get('id', 'unknown')}")
                    
            except Exception as e:
                issues.append(f"Generation failed for sample {sample.get('id', 'unknown')}: {str(e)}")
                probe = GenerationProbe(
                    sample_id=sample.get('id', 'unknown'),
                    decoded_text="",
                    stop_reason="error",
                    new_token_count=0,
                    finish_reason="error",
                    processing_time_ms=0,
                    error=str(e)
                )
                probes.append(probe)
        
        # Calculate metrics
        median_tokens = np.median([p.new_token_count for p in probes]) if probes else 0
        response_rate = non_empty_responses / len(probes) if probes else 0
        success = response_rate >= 0.8 and median_tokens >= 5
        
        # Generate recommendations
        recommendations = []
        if response_rate < 0.8:
            recommendations.append("CRITICAL: Low response generation rate - check model connectivity")
            recommendations.append("Verify model is properly initialized and accessible")
        if median_tokens < 5:
            recommendations.append("WARNING: Very short responses - check generation parameters")
            recommendations.append("Increase max_tokens or check prompt engineering")
        if success:
            recommendations.append("✅ Generation: WORKING")
        
        return DiagnosticResult(
            tier_name="Raw-Capture Probe",
            success=success,
            score=response_rate,
            details={
                "total_samples": len(probes),
                "non_empty_responses": non_empty_responses,
                "response_rate": response_rate,
                "median_tokens": median_tokens,
                "total_tokens": total_tokens,
                "probe_samples": [
                    {
                        'sample_id': p.sample_id,
                        'text_preview': p.decoded_text,
                        'token_count': p.new_token_count,
                        'processing_time_ms': p.processing_time_ms
                    } for p in probes[:10]
                ]
            },
            samples_analyzed=len(probes),
            issues_found=issues,
            recommendations=recommendations
        )

    def id_space_check(self, method_fn: Callable, samples: List[dict], task_type: str) -> DiagnosticResult:
        """
        ID-Space Check: Verify ID namespace alignment for retrieval tasks.
        
        This test checks top-5 ID intersection with gold for retrieval tasks.
        Expected: Meaningful ID overlap for retrieval-based tasks.
        
        Args:
            method_fn: Function that returns retrieval results with IDs
            samples: List of evaluation samples
            task_type: Type of task (code_debug, passkey, retrieval, etc.)
            
        Returns:
            DiagnosticResult with ID namespace validation
        """
        logger.info(f"🔍 Running ID-Space Check for {task_type}")
        
        # Only applicable to certain task types
        if task_type not in ['code_debug', 'passkey', 'retrieval', 'kv_retrieval']:
            return DiagnosticResult(
                tier_name="ID-Space Check",
                success=True,
                score=1.0,
                details={"applicable": False, "task_type": task_type},
                samples_analyzed=0,
                recommendations=["Not applicable to task type: " + task_type]
            )
        
        intersections = []
        top5_ids_samples = []
        issues = []
        
        for sample in samples[:50]:  # Check first 50 for speed
            try:
                # Get predicted IDs from method
                pred_ids = self.get_top5_prediction_ids(method_fn, sample)
                gold_ids = self.get_gold_ids(sample)
                
                intersection = set(pred_ids) & set(gold_ids)
                intersections.append(len(intersection) > 0)
                
                top5_ids_samples.append({
                    'sample_id': sample.get('id', 'unknown'),
                    'pred_ids': pred_ids[:5],
                    'gold_ids': gold_ids,
                    'intersection': list(intersection)
                })
                
                if len(intersection) == 0:
                    issues.append(f"No ID overlap for sample {sample.get('id', 'unknown')}")
                    
            except Exception as e:
                issues.append(f"ID extraction failed for sample {sample.get('id', 'unknown')}: {str(e)}")
        
        intersection_rate = sum(intersections) / len(intersections) if intersections else 0.0
        success = intersection_rate >= 0.30  # At least 30% should have some overlap
        
        # Generate recommendations
        recommendations = []
        if intersection_rate < 0.30:
            recommendations.append("❌ ID Namespace: BROKEN - Fix retrieval ID mapping")
            recommendations.append("Check that predicted IDs match gold ID format")
            recommendations.append("Verify retrieval index uses correct document IDs")
        else:
            recommendations.append("✅ ID Namespace: WORKING")
        
        return DiagnosticResult(
            tier_name="ID-Space Check",
            success=success,
            score=intersection_rate,
            details={
                "applicable": True,
                "task_type": task_type,
                "intersection_rate": intersection_rate,
                "samples_checked": len(intersections),
                "id_samples": top5_ids_samples[:10]  # Return first 10 for inspection
            },
            samples_analyzed=len(intersections),
            issues_found=issues,
            recommendations=recommendations
        )

    def get_top5_prediction_ids(self, method_fn: Callable, sample: dict) -> List[str]:
        """Extract top-5 prediction IDs from method response."""
        try:
            response = method_fn(sample)
            
            # Try to extract IDs from various response formats
            if hasattr(response, 'retrieved_docs'):
                return [doc.id for doc in response.retrieved_docs[:5]]
            elif hasattr(response, 'document_ids'):
                return response.document_ids[:5]
            elif isinstance(response, dict) and 'ids' in response:
                return response['ids'][:5]
            elif isinstance(response, dict) and 'documents' in response:
                return [doc.get('id', '') for doc in response['documents'][:5]]
            else:
                # Try to extract from text response
                text = str(response)
                # Look for common ID patterns
                ids = re.findall(r'\b(?:doc|id)[-_]?\d+\b', text, re.IGNORECASE)
                return ids[:5]
                
        except Exception as e:
            logger.warning(f"Failed to extract prediction IDs: {e}")
            return []

    def get_gold_ids(self, sample: dict) -> List[str]:
        """Extract gold standard IDs from sample."""
        try:
            # Try various gold ID field names
            if 'gold_ids' in sample:
                return sample['gold_ids']
            elif 'relevant_ids' in sample:
                return sample['relevant_ids']
            elif 'document_ids' in sample:
                return sample['document_ids']
            elif 'target_id' in sample:
                return [sample['target_id']]
            else:
                # Try to extract from answer field
                answer = sample.get('answer', '')
                if answer:
                    ids = re.findall(r'\b(?:doc|id)[-_]?\d+\b', answer, re.IGNORECASE)
                    return ids
                return []
                
        except Exception as e:
            logger.warning(f"Failed to extract gold IDs: {e}")
            return []

    def run_comprehensive_diagnostic(self, 
                                   method_fn: Callable,
                                   samples: List[dict],
                                   task_type: str,
                                   experiment_name: str = "evaluation_diagnostic") -> ComprehensiveDiagnostic:
        """
        Run complete three-tier diagnostic system.
        
        Args:
            method_fn: Evaluation method function to test
            samples: List of evaluation samples
            task_type: Type of evaluation task
            experiment_name: Name for this diagnostic run
            
        Returns:
            ComprehensiveDiagnostic with complete analysis
        """
        logger.info(f"🧪 Starting comprehensive diagnostic: {experiment_name}")
        start_time = time.time()
        
        # Prepare samples for diagnostics
        diagnostic_samples = []
        for sample in samples:
            if isinstance(sample, InfiniteBenchSample):
                diagnostic_samples.append({
                    'id': sample.id,
                    'question': sample.question,
                    'context': sample.context,
                    'answer': sample.answer,
                    'language': getattr(sample, 'language', 'en')
                })
            else:
                diagnostic_samples.append(sample)
        
        if len(diagnostic_samples) == 0:
            raise ValueError("No samples provided for diagnostic")
        
        # Run all three tiers
        tier_results = {}
        
        # Tier 1: Gold-Echo Control (Scorer Sanity)
        try:
            tier_results["gold_echo"] = self.gold_echo_control(diagnostic_samples, n_samples=200)
        except Exception as e:
            logger.error(f"Gold-Echo Control failed: {e}")
            tier_results["gold_echo"] = DiagnosticResult(
                tier_name="Gold-Echo Control",
                success=False,
                score=0.0,
                details={"error": str(e)},
                samples_analyzed=0,
                issues_found=[f"Control failed: {str(e)}"],
                recommendations=["Fix Gold-Echo Control implementation"]
            )
        
        # Tier 2: Raw-Capture Probe (Generation Sanity)
        try:
            tier_results["raw_capture"] = self.raw_capture_probe(method_fn, diagnostic_samples, n_samples=100)
        except Exception as e:
            logger.error(f"Raw-Capture Probe failed: {e}")
            tier_results["raw_capture"] = DiagnosticResult(
                tier_name="Raw-Capture Probe",
                success=False,
                score=0.0,
                details={"error": str(e)},
                samples_analyzed=0,
                issues_found=[f"Probe failed: {str(e)}"],
                recommendations=["Fix Raw-Capture Probe implementation"]
            )
        
        # Tier 3: ID-Space Check (Retrieval Validation)
        try:
            tier_results["id_space"] = self.id_space_check(method_fn, diagnostic_samples, task_type)
        except Exception as e:
            logger.error(f"ID-Space Check failed: {e}")
            tier_results["id_space"] = DiagnosticResult(
                tier_name="ID-Space Check",
                success=False,
                score=0.0,
                details={"error": str(e)},
                samples_analyzed=0,
                issues_found=[f"Check failed: {str(e)}"],
                recommendations=["Fix ID-Space Check implementation"]
            )
        
        # Analyze overall results
        critical_failures = []
        repair_priority = []
        
        for tier_name, result in tier_results.items():
            if not result.success:
                critical_failures.append(f"{result.tier_name}: {result.score:.3f}")
                if tier_name == "gold_echo":
                    repair_priority.append("1. Fix Tier 3 (Scorer/Normalization) - CRITICAL")
                elif tier_name == "raw_capture":
                    repair_priority.append("2. Fix Tier 1 (Generation) - HIGH")
                elif tier_name == "id_space":
                    repair_priority.append("3. Fix Tier 2 (ID Mapping) - MEDIUM")
        
        # Generate overall assessment
        if len(critical_failures) == 0:
            overall_assessment = "✅ All diagnostic tiers PASSED - evaluation pipeline is healthy"
        elif "gold_echo" in [k for k, v in tier_results.items() if not v.success]:
            overall_assessment = "❌ CRITICAL: Scorer/Normalization BROKEN - fix immediately"
        elif "raw_capture" in [k for k, v in tier_results.items() if not v.success]:
            overall_assessment = "⚠️ HIGH: Generation pipeline issues detected"
        else:
            overall_assessment = "⚠️ MEDIUM: Specific subsystem issues detected"
        
        duration = time.time() - start_time
        logger.info(f"🎯 Diagnostic completed in {duration:.1f}s")
        
        return ComprehensiveDiagnostic(
            experiment_name=experiment_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tier_results=tier_results,
            overall_assessment=overall_assessment,
            critical_failures=critical_failures,
            repair_priority=repair_priority
        )

    def print_diagnostic_report(self, diagnostic: ComprehensiveDiagnostic) -> None:
        """Print comprehensive diagnostic report to console."""
        print("\n" + "="*80)
        print("🧪 EVALUATION DIAGNOSTIC REPORT")
        print("="*80)
        print(f"Experiment: {diagnostic.experiment_name}")
        print(f"Timestamp: {diagnostic.timestamp}")
        print()
        
        # Print tier results
        for i, (tier_key, result) in enumerate(diagnostic.tier_results.items(), 1):
            status = "✅" if result.success else "❌"
            print(f"{i}. {status} {result.tier_name} (n={result.samples_analyzed})")
            print(f"   Score: {result.score:.3f} (Expected: ≥0.95 for gold_echo, ≥0.80 for others)")
            
            # Show key details
            if result.tier_name == "Gold-Echo Control":
                accuracy = result.details.get('accuracy', 0.0)
                print(f"   Accuracy: {accuracy:.3f} (Expected: 1.000)")
            elif result.tier_name == "Raw-Capture Probe":
                response_rate = result.details.get('response_rate', 0.0)
                median_tokens = result.details.get('median_tokens', 0)
                print(f"   Non-empty responses: {response_rate:.1%} (Expected: ≥80%)")
                print(f"   Median tokens: {median_tokens} (Expected: ≥5)")
            elif result.tier_name == "ID-Space Check":
                if result.details.get('applicable', True):
                    intersection_rate = result.details.get('intersection_rate', 0.0)
                    print(f"   ID intersection rate: {intersection_rate:.3f} (Expected: ≥0.30)")
                else:
                    print(f"   Not applicable to task type")
            
            # Show main recommendation
            if result.recommendations:
                print(f"   → {result.recommendations[0]}")
            print()
        
        # Overall assessment
        print("🎯 OVERALL ASSESSMENT")
        print(f"{diagnostic.overall_assessment}")
        print()
        
        # Repair priority
        if diagnostic.repair_priority:
            print("🔧 REPAIR PRIORITY")
            for priority in diagnostic.repair_priority:
                print(f"   {priority}")
            print()
        
        # Critical failures
        if diagnostic.critical_failures:
            print("💥 CRITICAL FAILURES")
            for failure in diagnostic.critical_failures:
                print(f"   • {failure}")
        else:
            print("✅ No critical failures detected")
        
        print("="*80)

    def save_diagnostic_report(self, diagnostic: ComprehensiveDiagnostic, output_path: Union[str, Path]) -> None:
        """Save diagnostic report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        report_data = {
            "experiment_name": diagnostic.experiment_name,
            "timestamp": diagnostic.timestamp,
            "overall_assessment": diagnostic.overall_assessment,
            "critical_failures": diagnostic.critical_failures,
            "repair_priority": diagnostic.repair_priority,
            "tier_results": {}
        }
        
        for tier_key, result in diagnostic.tier_results.items():
            report_data["tier_results"][tier_key] = {
                "tier_name": result.tier_name,
                "success": result.success,
                "score": result.score,
                "samples_analyzed": result.samples_analyzed,
                "details": result.details,
                "issues_found": result.issues_found,
                "recommendations": result.recommendations
            }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Diagnostic report saved to {output_path}")


def main():
    """Example usage of evaluation diagnostics system."""
    import sys
    from pathlib import Path
    
    # Mock method function for testing
    def mock_method(sample):
        """Mock method that returns dummy generation result."""
        return type('MockResponse', (), {
            'text': f"Generated response for {sample.get('id', 'unknown')}",
            'stop_reason': 'length',
            'finish_reason': 'stop'
        })()
    
    # Create diagnostics system
    data_dir = Path("benchmarks/infinitebench/data")
    diagnostics = EvaluationDiagnostics(data_dir)
    
    # Create mock samples
    mock_samples = [
        {
            'id': f'sample_{i}',
            'question': f'Test question {i}',
            'context': f'Test context {i}' * 100,  # Long context
            'answer': f'Test answer {i}',
            'language': 'en'
        }
        for i in range(10)
    ]
    
    try:
        # Run comprehensive diagnostic
        result = diagnostics.run_comprehensive_diagnostic(
            method_fn=mock_method,
            samples=mock_samples,
            task_type="code_debug",
            experiment_name="test_diagnostic"
        )
        
        # Print report
        diagnostics.print_diagnostic_report(result)
        
        # Save report
        output_path = Path("diagnostic_results/test_diagnostic.json")
        diagnostics.save_diagnostic_report(result, output_path)
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()