"""
Cross-Encoder Synthetic Test Suite
==================================

Mandatory synthetic pair tests that must pass before any full evaluation run.
Tests cross-encoder with carefully constructed pairs to validate:

1. Identical pairs score highest (perfect match)
2. Completely disjoint pairs score lowest  
3. Partial overlap pairs score in between
4. Score distribution has meaningful variance (std > 0.2)

These synthetic tests catch flat scoring issues immediately without requiring
real evaluation data. All tests must pass before proceeding to real data.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SyntheticTestResult:
    """Results from synthetic pair testing."""
    test_passed: bool
    score_variance: float
    score_range: float
    ranking_correct: bool
    flat_scoring_detected: bool
    scores: List[float]
    test_pairs: List[Tuple[str, str]]
    issues_found: List[str]
    fix_recommendations: List[str]

class CrossEncoderSyntheticTester:
    """
    Synthetic test suite for cross-encoder validation.
    
    Uses carefully constructed test pairs to validate that the cross-encoder
    produces meaningful, differentiated scores before running on real data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize synthetic tester.
        
        Args:
            config: Configuration for test thresholds
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for synthetic tests."""
        return {
            'min_score_std': 0.2,     # Minimum standard deviation for scores
            'min_score_range': 0.3,   # Minimum range between max and min scores
            'identical_score_threshold': 0.7,  # Minimum score for identical pairs
            'disjoint_score_threshold': 0.4,   # Maximum score for disjoint pairs
            'require_correct_ranking': True,   # Require identical > partial > disjoint
            'num_test_iterations': 3,  # Number of test runs for consistency
            'floating_point_tolerance': 1e-6  # Tolerance for floating point comparisons
        }
    
    def run_synthetic_tests(self, 
                          cross_encoder: Any,
                          tokenizer: Any = None) -> SyntheticTestResult:
        """
        Run complete synthetic test suite.
        
        Args:
            cross_encoder: Cross-encoder model to test
            tokenizer: Optional tokenizer (will try to get from model)
            
        Returns:
            SyntheticTestResult with pass/fail status and diagnostics
        """
        self.logger.info("🧪 Starting Cross-Encoder Synthetic Tests")
        self.logger.info("=" * 50)
        
        issues = []
        fixes = []
        
        # Generate test pairs
        test_pairs = self._generate_test_pairs()
        
        # Run scoring tests
        scores, scoring_issues = self._run_scoring_tests(cross_encoder, test_pairs, tokenizer)
        issues.extend(scoring_issues)
        
        if not scores:
            return SyntheticTestResult(
                test_passed=False,
                score_variance=0.0,
                score_range=0.0,
                ranking_correct=False,
                flat_scoring_detected=True,
                scores=[],
                test_pairs=test_pairs,
                issues_found=["CRITICAL: No scores generated from cross-encoder"],
                fix_recommendations=["Check cross-encoder implementation and tokenizer"]
            )
        
        # Analyze score distribution
        variance_issues = self._test_score_variance(scores)
        issues.extend(variance_issues)
        
        # Test ranking correctness
        ranking_correct, ranking_issues = self._test_ranking_correctness(scores)
        issues.extend(ranking_issues)
        
        # Test for flat scoring
        flat_scoring = self._detect_flat_scoring(scores)
        if flat_scoring:
            issues.append("CRITICAL: Flat scoring detected - cross-encoder producing identical scores")
        
        # Calculate metrics
        score_variance = float(np.std(scores))
        score_range = float(max(scores) - min(scores))
        
        # Generate fix recommendations
        fixes = self._generate_synthetic_test_fixes(issues, scores, test_pairs)
        
        # Determine overall pass/fail
        critical_issues = [i for i in issues if 'CRITICAL' in i]
        test_passed = len(critical_issues) == 0 and ranking_correct and not flat_scoring
        
        # Log results
        self._log_synthetic_test_results(test_pairs, scores, issues, test_passed)
        
        return SyntheticTestResult(
            test_passed=test_passed,
            score_variance=score_variance,
            score_range=score_range,
            ranking_correct=ranking_correct,
            flat_scoring_detected=flat_scoring,
            scores=scores,
            test_pairs=test_pairs,
            issues_found=issues,
            fix_recommendations=fixes
        )
    
    def _generate_test_pairs(self) -> List[Tuple[str, str]]:
        """Generate synthetic test pairs for validation."""
        pairs = [
            # Identical pairs (should score highest)
            ("the quick brown fox", "the quick brown fox"),
            ("machine learning algorithms", "machine learning algorithms"), 
            ("data structure implementation", "data structure implementation"),
            
            # Completely disjoint pairs (should score lowest)
            ("abc def", "xyz uvw"),
            ("red blue green", "seven eight nine"),
            ("programming language", "kitchen utensils"),
            
            # Partial overlap pairs (should score in between)
            ("sum of squares", "sum of squares formula a^2 + b^2"),
            ("machine learning", "machine learning models and algorithms"),
            ("the quick brown fox", "the brown fox jumped quickly"),
            
            # Edge cases
            ("", "empty string test"),  # Empty query
            ("single", ""),             # Empty document
            ("a", "b"),                 # Single characters
            ("very long text with many words that should test tokenizer limits", "short text"),
        ]
        
        self.logger.info(f"Generated {len(pairs)} synthetic test pairs")
        return pairs
    
    def _run_scoring_tests(self, 
                          cross_encoder: Any,
                          test_pairs: List[Tuple[str, str]],
                          tokenizer: Any) -> Tuple[List[float], List[str]]:
        """Run scoring on test pairs."""
        issues = []
        scores = []
        
        self.logger.info("Running synthetic pair scoring...")
        
        try:
            # Try multiple scoring iterations for consistency
            for iteration in range(self.config['num_test_iterations']):
                iteration_scores = []
                
                for i, (query, doc) in enumerate(test_pairs):
                    try:
                        score = self._score_pair(cross_encoder, query, doc, tokenizer)
                        if score is not None:
                            iteration_scores.append(float(score))
                        else:
                            issues.append(f"WARNING: Pair {i} returned None score")
                            iteration_scores.append(0.0)  # Default score
                    except Exception as e:
                        issues.append(f"ERROR: Scoring pair {i} failed: {str(e)}")
                        iteration_scores.append(0.0)
                
                scores.append(iteration_scores)
                self.logger.debug(f"Iteration {iteration + 1} scores: {iteration_scores}")
            
            # Average scores across iterations
            if scores:
                avg_scores = [float(np.mean([scores[iter][i] for iter in range(len(scores))])) 
                             for i in range(len(test_pairs))]
                
                # Check score consistency across iterations
                score_stds = [float(np.std([scores[iter][i] for iter in range(len(scores))]))
                             for i in range(len(test_pairs))]
                
                max_score_std = max(score_stds) if score_stds else 0.0
                if max_score_std > 0.1:
                    issues.append(f"WARNING: Inconsistent scores across iterations (max std: {max_score_std:.3f})")
                
                return avg_scores, issues
            else:
                issues.append("CRITICAL: No scores generated in any iteration")
                return [], issues
                
        except Exception as e:
            issues.append(f"CRITICAL: Synthetic scoring failed: {str(e)}")
            return [], issues
    
    def _score_pair(self, cross_encoder: Any, query: str, doc: str, tokenizer: Any) -> Optional[float]:
        """Score a single query-document pair."""
        try:
            # Try different scoring methods
            if hasattr(cross_encoder, 'score_pairs'):
                # Use score_pairs method if available
                scores = cross_encoder.score_pairs(query, ['doc_0'], {'doc_0': doc})
                return scores.get('doc_0') if scores else None
                
            elif hasattr(cross_encoder, 'predict'):
                # Use predict method
                result = cross_encoder.predict([[query, doc]])
                if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                    return float(result[0])
                return float(result) if result is not None else None
                
            elif hasattr(cross_encoder, '__call__'):
                # Use direct call
                result = cross_encoder(query, doc)
                return float(result) if result is not None else None
                
            else:
                # Try manual tokenization and inference
                return self._manual_score_pair(cross_encoder, query, doc, tokenizer)
                
        except Exception as e:
            self.logger.warning(f"Pair scoring failed: {e}")
            return None
    
    def _manual_score_pair(self, model: Any, query: str, doc: str, tokenizer: Any) -> Optional[float]:
        """Manually score pair using tokenizer and model."""
        try:
            if tokenizer is None:
                # Try to get tokenizer from model
                if hasattr(model, 'tokenizer'):
                    tokenizer = model.tokenizer
                else:
                    self.logger.warning("No tokenizer available for manual scoring")
                    return None
            
            # Tokenize input pair
            inputs = tokenizer(
                query, doc,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to same device as model
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Extract score from outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Handle different output formats
                if logits.shape[-1] == 1:
                    # Regression output
                    score = logits.squeeze(-1).item()
                else:
                    # Classification output - use positive class
                    scores = torch.softmax(logits, dim=-1)
                    score = scores[0, -1].item()  # Last class (usually positive)
                
                return float(score)
                
        except Exception as e:
            self.logger.warning(f"Manual scoring failed: {e}")
            return None
    
    def _test_score_variance(self, scores: List[float]) -> List[str]:
        """Test if scores have sufficient variance."""
        issues = []
        
        if not scores:
            issues.append("CRITICAL: No scores to test variance")
            return issues
        
        score_std = np.std(scores)
        score_range = max(scores) - min(scores)
        
        if score_std < self.config['min_score_std']:
            issues.append(f"CRITICAL: Score std too low: {score_std:.3f} < {self.config['min_score_std']}")
        
        if score_range < self.config['min_score_range']:
            issues.append(f"CRITICAL: Score range too narrow: {score_range:.3f} < {self.config['min_score_range']}")
        
        # Check for floating point precision issues
        unique_scores = len(set(np.round(scores, 6)))  # Round to avoid fp precision issues
        if unique_scores <= 2:
            issues.append(f"CRITICAL: Only {unique_scores} unique score values")
        
        return issues
    
    def _test_ranking_correctness(self, scores: List[float]) -> Tuple[bool, List[str]]:
        """Test if identical pairs score higher than partial overlap, which score higher than disjoint."""
        issues = []
        
        if len(scores) < 6:  # Need at least 6 scores for 3 categories
            issues.append("WARNING: Not enough scores to test ranking")
            return False, issues
        
        try:
            # Expected score order: identical (0-2) > partial (6-8) > disjoint (3-5) 
            identical_scores = scores[0:3]   # First 3 are identical pairs
            disjoint_scores = scores[3:6]    # Next 3 are disjoint pairs  
            partial_scores = scores[6:9] if len(scores) > 8 else scores[6:len(scores)]  # Partial overlap pairs
            
            avg_identical = np.mean(identical_scores) if identical_scores else 0.0
            avg_partial = np.mean(partial_scores) if partial_scores else 0.0
            avg_disjoint = np.mean(disjoint_scores) if disjoint_scores else 0.0
            
            self.logger.info(f"Average scores - Identical: {avg_identical:.3f}, Partial: {avg_partial:.3f}, Disjoint: {avg_disjoint:.3f}")
            
            ranking_correct = True
            
            # Test identical > partial  
            if avg_identical <= avg_partial:
                issues.append(f"RANKING ERROR: Identical pairs ({avg_identical:.3f}) should score > partial overlap ({avg_partial:.3f})")
                ranking_correct = False
            
            # Test partial > disjoint
            if avg_partial <= avg_disjoint:
                issues.append(f"RANKING ERROR: Partial overlap ({avg_partial:.3f}) should score > disjoint pairs ({avg_disjoint:.3f})")
                ranking_correct = False
            
            # Test identical > disjoint (should be strongest signal)
            if avg_identical <= avg_disjoint:
                issues.append(f"RANKING ERROR: Identical pairs ({avg_identical:.3f}) should score > disjoint pairs ({avg_disjoint:.3f})")
                ranking_correct = False
            
            # Additional thresholds
            if avg_identical < self.config['identical_score_threshold']:
                issues.append(f"WARNING: Identical pairs score low: {avg_identical:.3f} < {self.config['identical_score_threshold']}")
                
            if avg_disjoint > self.config['disjoint_score_threshold']:
                issues.append(f"WARNING: Disjoint pairs score high: {avg_disjoint:.3f} > {self.config['disjoint_score_threshold']}")
            
            return ranking_correct, issues
            
        except Exception as e:
            issues.append(f"ERROR: Ranking test failed: {str(e)}")
            return False, issues
    
    def _detect_flat_scoring(self, scores: List[float]) -> bool:
        """Detect if all scores are essentially identical (flat scoring)."""
        if not scores:
            return True
        
        # Check if all scores are within floating point tolerance
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        # Consider flat if range is smaller than tolerance
        flat_threshold = self.config['floating_point_tolerance'] * 10  # 10x tolerance
        return score_range < flat_threshold
    
    def _generate_synthetic_test_fixes(self, 
                                     issues: List[str],
                                     scores: List[float],
                                     test_pairs: List[Tuple[str, str]]) -> List[str]:
        """Generate fix recommendations for synthetic test failures."""
        fixes = []
        
        # Critical issue fixes
        for issue in issues:
            if "No scores generated" in issue:
                fixes.append("Check cross-encoder model is loaded and inference method is correct")
            elif "Score std too low" in issue or "Score range too narrow" in issue:
                fixes.append("Cross-encoder producing flat scores - check model weights, tokenizer, and input formatting")
            elif "Only" in issue and "unique score values" in issue:
                fixes.append("Cross-encoder returning constant scores - verify model is not broken")
            elif "RANKING ERROR" in issue:
                fixes.append("Cross-encoder not distinguishing between different similarity levels - check model training")
            elif "Inconsistent scores" in issue:
                fixes.append("Cross-encoder producing non-deterministic results - ensure model.eval() and disable dropout")
        
        # General recommendations based on score analysis
        if scores:
            unique_scores = len(set(np.round(scores, 3)))
            if unique_scores == 1:
                fixes.append("All scores identical - check tokenizer special tokens ([CLS], [SEP]) and input format")
            elif unique_scores == 2:
                fixes.append("Only 2 unique scores - check model head (binary vs regression) and output processing")
        
        # Add specific diagnostic recommendations
        if any("flat" in issue.lower() for issue in issues):
            fixes.extend([
                "Run model.eval() to disable dropout",
                "Check tokenizer truncation strategy (longest_first vs only_second)",
                "Verify special tokens are correctly positioned in input",
                "Test with different precision (fp32 vs fp16)",
                "Validate model checkpoint was loaded correctly"
            ])
        
        if not fixes:
            fixes.append("Synthetic tests passed - cross-encoder producing meaningful score distributions")
        
        return fixes
    
    def _log_synthetic_test_results(self, 
                                  test_pairs: List[Tuple[str, str]],
                                  scores: List[float], 
                                  issues: List[str],
                                  test_passed: bool):
        """Log detailed synthetic test results."""
        self.logger.info("🧪 SYNTHETIC TEST RESULTS:")
        self.logger.info("-" * 40)
        
        # Log test pairs and scores
        for i, ((query, doc), score) in enumerate(zip(test_pairs[:6], scores[:6])):
            pair_type = "IDENTICAL" if i < 3 else "DISJOINT"
            self.logger.info(f"  {i+1}. [{pair_type}] '{query[:30]}...' | '{doc[:30]}...' → {score:.3f}")
        
        if len(scores) > 6:
            self.logger.info(f"  ... and {len(scores) - 6} more pairs")
        
        # Score statistics
        if scores:
            self.logger.info(f"Score Statistics:")
            self.logger.info(f"  Mean: {np.mean(scores):.3f}")
            self.logger.info(f"  Std:  {np.std(scores):.3f}")
            self.logger.info(f"  Range: {max(scores) - min(scores):.3f}")
            self.logger.info(f"  Unique values: {len(set(np.round(scores, 3)))}")
        
        # Issues
        if issues:
            self.logger.warning(f"Issues Found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                level = "🚨" if "CRITICAL" in issue else "⚠️" if "WARNING" in issue else "ℹ️"
                self.logger.warning(f"  {i}. {level} {issue}")
        
        # Overall result
        if test_passed:
            self.logger.info("✅ SYNTHETIC TESTS PASSED - Cross-encoder producing differentiated scores")
        else:
            self.logger.error("❌ SYNTHETIC TESTS FAILED - Cross-encoder not working correctly")
        
        self.logger.info("=" * 50)