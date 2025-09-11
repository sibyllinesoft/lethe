#!/usr/bin/env python3
"""
Standalone Cross-Encoder Debugging CLI
=====================================

Self-contained diagnostic script for cross-encoder debugging without complex
import dependencies. Focuses on the core debugging functionality.

Usage:
    python debug_cross_encoder_standalone.py --model MODEL_NAME [options]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import hashlib

# Check for required libraries
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import transformers or torch: {e}")
    print("Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

class StandaloneCrossEncoderDebugger:
    """
    Standalone cross-encoder debugger that runs all essential diagnostics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run_complete_diagnosis(self, 
                             model_name: str,
                             device: str = 'cpu',
                             max_seq_len: int = 512) -> Dict[str, Any]:
        """
        Run complete standalone diagnosis.
        
        Args:
            model_name: HuggingFace model identifier
            device: Target device
            max_seq_len: Maximum sequence length
            
        Returns:
            Complete diagnostic results
        """
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "Transformers library not available"}
            
        self.logger.info("🔍 STANDALONE CROSS-ENCODER DIAGNOSIS")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Device: {device}")
        self.logger.info("=" * 50)
        
        results = {
            'model_name': model_name,
            'device': device,
            'max_seq_len': max_seq_len,
            'diagnosis_timestamp': time.time(),
            'stages': {}
        }
        
        try:
            # Stage 1: Load model
            self.logger.info("📦 STAGE 1: Loading Model and Tokenizer")
            model, tokenizer = self._load_model(model_name, device)
            if model is None or tokenizer is None:
                results['stages']['loading'] = {'success': False, 'error': 'Model loading failed'}
                return results
            results['stages']['loading'] = {'success': True}
            
            # Stage 2: Basic attestation
            self.logger.info("🔐 STAGE 2: Basic Attestation")
            attestation = self._basic_attestation(model, tokenizer, model_name, device, max_seq_len)
            results['stages']['attestation'] = attestation
            
            # Stage 3: Synthetic tests
            self.logger.info("🧪 STAGE 3: Synthetic Tests")
            synthetic = self._synthetic_tests(model, tokenizer)
            results['stages']['synthetic_tests'] = synthetic
            
            # Stage 4: Input debugging
            self.logger.info("🔍 STAGE 4: Input Debugging")
            input_debug = self._input_debugging(model, tokenizer, max_seq_len)
            results['stages']['input_debugging'] = input_debug
            
            # Stage 5: Safe mode recommendations
            self.logger.info("🛡️ STAGE 5: Parameter Recommendations")
            recommendations = self._parameter_recommendations(results)
            results['recommendations'] = recommendations
            
            # Overall status
            overall_status = self._determine_status(results)
            results['overall_status'] = overall_status
            
            self._log_summary(results)
            
        except Exception as e:
            self.logger.error(f"Diagnosis failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _load_model(self, model_name: str, device: str):
        """Load model and tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
            self.logger.info("✅ Model and tokenizer loaded successfully")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return None, None
    
    def _basic_attestation(self, model, tokenizer, model_name, device, max_seq_len):
        """Basic attestation checks."""
        issues = []
        
        # Check basic properties
        if not model.training:
            self.logger.info("✅ Model in evaluation mode")
        else:
            issues.append("Model in training mode")
            self.logger.warning("⚠️ Model in training mode - should be eval()")
        
        # Check config
        if hasattr(model, 'config'):
            config = model.config
            num_labels = getattr(config, 'num_labels', None)
            if num_labels == 1:
                self.logger.info("✅ Regression head detected (1 output)")
            elif num_labels == 2:
                self.logger.info("✅ Binary classification head detected (2 outputs)")
            else:
                issues.append(f"Unexpected num_labels: {num_labels}")
                self.logger.warning(f"⚠️ Unexpected num_labels: {num_labels}")
        else:
            issues.append("No config found")
        
        # Check tokenizer
        if tokenizer:
            vocab_size = getattr(tokenizer, 'vocab_size', None)
            max_length = getattr(tokenizer, 'model_max_length', None)
            self.logger.info(f"✅ Tokenizer vocab_size: {vocab_size}, max_length: {max_length}")
        else:
            issues.append("No tokenizer")
        
        return {
            'success': len(issues) == 0,
            'issues': issues,
            'model_name': model_name,
            'device': device,
            'max_seq_len': max_seq_len
        }
    
    def _synthetic_tests(self, model, tokenizer):
        """Run synthetic test pairs."""
        test_pairs = [
            ("the quick brown fox", "the quick brown fox"),      # identical
            ("abc def", "xyz uvw"),                              # disjoint  
            ("sum of squares", "sum of squares formula a^2"),    # partial overlap
            ("machine learning", "deep learning algorithms"),    # related
            ("programming", "cooking recipes"),                  # unrelated
        ]
        
        scores = []
        issues = []
        
        try:
            for i, (query, doc) in enumerate(test_pairs):
                score = self._score_pair(model, tokenizer, query, doc)
                if score is not None:
                    scores.append(score)
                    pair_type = ["identical", "disjoint", "partial", "related", "unrelated"][i]
                    self.logger.debug(f"  {pair_type}: {score:.3f}")
                else:
                    issues.append(f"Failed to score pair {i}")
            
            if scores:
                score_std = np.std(scores)
                score_range = max(scores) - min(scores)
                unique_scores = len(set(np.round(scores, 3)))
                
                self.logger.info(f"📊 Score statistics:")
                self.logger.info(f"  Mean: {np.mean(scores):.3f}")
                self.logger.info(f"  Std:  {score_std:.3f}")
                self.logger.info(f"  Range: {score_range:.3f}")
                self.logger.info(f"  Unique: {unique_scores}")
                
                # Check for flat scoring
                flat_scoring = score_std < 0.05 or score_range < 0.1 or unique_scores <= 2
                
                if flat_scoring:
                    issues.append("CRITICAL: Flat scoring detected")
                    self.logger.error("🚨 FLAT SCORING DETECTED")
                else:
                    self.logger.info("✅ Scores show reasonable variance")
                
                # Check ranking (identical should score highest)
                if len(scores) >= 2:
                    identical_score = scores[0]
                    disjoint_score = scores[1]
                    if identical_score > disjoint_score:
                        self.logger.info("✅ Ranking appears correct (identical > disjoint)")
                    else:
                        issues.append("Ranking incorrect: identical <= disjoint")
                        self.logger.warning("⚠️ Ranking appears incorrect")
                
                return {
                    'success': not flat_scoring and len(issues) == 0,
                    'flat_scoring_detected': flat_scoring,
                    'score_std': score_std,
                    'score_range': score_range,
                    'unique_scores': unique_scores,
                    'scores': scores,
                    'issues': issues
                }
            else:
                issues.append("No valid scores generated")
                return {'success': False, 'issues': issues, 'scores': []}
                
        except Exception as e:
            issues.append(f"Synthetic test failed: {str(e)}")
            return {'success': False, 'issues': issues, 'error': str(e)}
    
    def _score_pair(self, model, tokenizer, query, doc):
        """Score a single query-document pair."""
        try:
            inputs = tokenizer(
                query, doc,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move to same device as model
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Handle different output formats
                if logits.shape[-1] == 1:
                    # Regression
                    score = logits.squeeze(-1).item()
                else:
                    # Classification - use positive class
                    scores = torch.softmax(logits, dim=-1)
                    score = scores[0, -1].item()
                
                return float(score)
                
        except Exception as e:
            self.logger.warning(f"Scoring failed: {e}")
            return None
    
    def _input_debugging(self, model, tokenizer, max_seq_len):
        """Debug input formatting."""
        issues = []
        sample_inputs = []
        
        try:
            test_cases = [
                ("short query", "short document"),
                ("", "empty query test"),
                ("long query " * 20, "long document " * 30)
            ]
            
            for i, (query, doc) in enumerate(test_cases):
                try:
                    inputs = tokenizer(
                        query, doc,
                        truncation=True,
                        padding=True,
                        max_length=max_seq_len,
                        return_tensors="pt"
                    )
                    
                    # Analyze tokenization
                    input_ids = inputs['input_ids'][0].tolist()
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    
                    sample_input = {
                        'case': i,
                        'query_length': len(query),
                        'doc_length': len(doc),
                        'total_tokens': len(input_ids),
                        'first_10_tokens': tokens[:10],
                        'last_10_tokens': tokens[-10:] if len(tokens) > 10 else tokens
                    }
                    
                    sample_inputs.append(sample_input)
                    
                    if i == 0:  # Log details for first case
                        self.logger.info(f"📝 Sample tokenization:")
                        self.logger.info(f"  Query: '{query}'")
                        self.logger.info(f"  Document: '{doc}'")
                        self.logger.info(f"  Total tokens: {len(input_ids)}")
                        self.logger.info(f"  First 10: {tokens[:10]}")
                    
                    # Check for issues
                    if len(input_ids) == 0:
                        issues.append(f"Case {i}: Empty tokenization")
                    elif len(input_ids) > max_seq_len:
                        issues.append(f"Case {i}: Exceeds max length")
                    
                    # Check for special tokens
                    if '[CLS]' not in tokens and '<s>' not in tokens:
                        issues.append(f"Case {i}: Missing start token")
                    if '[SEP]' not in tokens and '</s>' not in tokens:
                        issues.append(f"Case {i}: Missing separator token")
                    
                except Exception as e:
                    issues.append(f"Case {i} tokenization failed: {str(e)}")
            
            return {
                'success': len(issues) == 0,
                'issues': issues,
                'sample_inputs': sample_inputs
            }
            
        except Exception as e:
            return {
                'success': False,
                'issues': [f"Input debugging failed: {str(e)}"],
                'error': str(e)
            }
    
    def _parameter_recommendations(self, results):
        """Generate parameter recommendations."""
        recommendations = {
            'immediate_adjustments': {
                'K1': 5000,
                'K2': 1200,
                'dims': 768,
                'diversity_delta': 0.0,
                'facility_gamma': 0.8
            },
            'safe_mode_config': {
                'bi_encoder_weight': 0.6,
                'bm25_weight': 0.4,
                'k1_candidate_pool': 5000,
                'k2_rerank_budget': 1200
            },
            'fixes': []
        }
        
        # Analyze results and add specific fixes
        stages = results.get('stages', {})
        
        if stages.get('synthetic_tests', {}).get('flat_scoring_detected'):
            recommendations['fixes'].extend([
                "CRITICAL: Cross-encoder producing flat scores",
                "Activate safe mode immediately",
                "Check model weights and tokenizer configuration",
                "Consider retraining or replacing the cross-encoder"
            ])
        
        if stages.get('attestation', {}).get('issues'):
            recommendations['fixes'].append("Resolve attestation issues first")
        
        if stages.get('input_debugging', {}).get('issues'):
            recommendations['fixes'].append("Fix input formatting and tokenization issues")
        
        return recommendations
    
    def _determine_status(self, results):
        """Determine overall diagnostic status."""
        stages = results.get('stages', {})
        
        critical_issues = 0
        total_stages = 0
        
        for stage_name, stage_data in stages.items():
            if isinstance(stage_data, dict):
                total_stages += 1
                if not stage_data.get('success', True):
                    critical_issues += 1
                
                # Check for flat scoring specifically
                if stage_name == 'synthetic_tests' and stage_data.get('flat_scoring_detected'):
                    critical_issues += 2  # Double weight for flat scoring
        
        if critical_issues == 0:
            return 'HEALTHY'
        elif critical_issues >= 3:
            return 'CRITICAL'
        else:
            return 'DEGRADED'
    
    def _log_summary(self, results):
        """Log final summary."""
        status = results.get('overall_status', 'UNKNOWN')
        
        self.logger.info("📋 DIAGNOSIS SUMMARY")
        self.logger.info("-" * 30)
        
        if status == 'HEALTHY':
            self.logger.info("✅ Cross-encoder appears healthy")
        elif status == 'DEGRADED':
            self.logger.warning("⚠️ Cross-encoder has issues but may be functional")
        else:
            self.logger.error("❌ Cross-encoder has critical issues")
        
        # Log recommendations
        recommendations = results.get('recommendations', {})
        if recommendations.get('fixes'):
            self.logger.info("🔧 Immediate actions needed:")
            for fix in recommendations['fixes']:
                self.logger.info(f"  • {fix}")
        
        # Log parameter adjustments
        params = recommendations.get('immediate_adjustments', {})
        if params:
            self.logger.info("🎯 Parameter adjustments:")
            for param, value in params.items():
                self.logger.info(f"  {param} = {value}")
        
        self.logger.info("-" * 30)

def setup_logging(level='INFO'):
    """Setup logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Standalone Cross-Encoder Debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model identifier')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library not available")
        print("Install with: pip install transformers torch")
        sys.exit(1)
    
    try:
        # Run diagnosis
        debugger = StandaloneCrossEncoderDebugger()
        results = debugger.run_complete_diagnosis(
            args.model, args.device, args.max_length
        )
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        status = results.get('overall_status')
        if status == 'CRITICAL':
            sys.exit(1)
        elif status == 'DEGRADED':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()