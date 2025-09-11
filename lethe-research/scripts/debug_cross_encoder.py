#!/usr/bin/env python3
"""
Cross-Encoder Debugging CLI
==========================

Command-line interface for the comprehensive cross-encoder debugging system.
Runs all diagnostic components in sequence to systematically identify and 
resolve cross-encoder issues causing flat scoring.

Usage:
    python debug_cross_encoder.py --model MODEL_NAME [options]
    python debug_cross_encoder.py --config config.json [options]

Features:
- Complete CE attestation and validation
- Synthetic test suite execution  
- Input formatting debugging
- Head architecture validation
- Safe mode activation on failure
- Automated fix recommendations
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diagnostics.ce_attestation import CrossEncoderAttestationSystem
from diagnostics.ce_synthetic_tests import CrossEncoderSyntheticTester
from diagnostics.ce_input_debugging import CrossEncoderInputDebugger
from diagnostics.ce_head_validation import CrossEncoderHeadValidator
from diagnostics.ce_safe_mode import CrossEncoderSafeMode, SafeModeConfig

# Import cross-encoder and related components
try:
    from rerank.cross_encoder import CrossEncoderReranker
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required libraries: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CrossEncoderDebugger:
    """
    Main coordinator for cross-encoder debugging system.
    
    Runs comprehensive diagnostic suite to identify and resolve
    cross-encoder issues causing flat scoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize debugger.
        
        Args:
            config: Configuration for debugging components
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize diagnostic components
        self.attestation_system = CrossEncoderAttestationSystem(
            config.get('attestation', {})
        )
        self.synthetic_tester = CrossEncoderSyntheticTester(
            config.get('synthetic_tests', {})
        )
        self.input_debugger = CrossEncoderInputDebugger(
            config.get('input_debugging', {})
        )
        self.head_validator = CrossEncoderHeadValidator(
            config.get('head_validation', {})
        )
        self.safe_mode = CrossEncoderSafeMode(
            SafeModeConfig(**config.get('safe_mode', {}))
        )
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.cross_encoder = None
        
    def run_complete_diagnosis(self, 
                             model_name: str,
                             device: str = 'cpu',
                             max_seq_len: int = 512,
                             truncation: str = 'longest_first') -> Dict[str, Any]:
        """
        Run complete cross-encoder diagnostic suite.
        
        Args:
            model_name: HuggingFace model identifier
            device: Target device
            max_seq_len: Maximum sequence length
            truncation: Truncation strategy
            
        Returns:
            Complete diagnostic results
        """
        self.logger.info("🔍 CROSS-ENCODER COMPREHENSIVE DIAGNOSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Max length: {max_seq_len}")
        self.logger.info(f"Truncation: {truncation}")
        self.logger.info("=" * 60)
        
        total_start_time = time.time()
        results = {
            'model_name': model_name,
            'device': device,
            'max_seq_len': max_seq_len,
            'truncation': truncation,
            'diagnosis_timestamp': time.time(),
            'stages': {}
        }
        
        try:
            # Stage 1: Load model and tokenizer
            self.logger.info("📦 STAGE 1: Loading Model and Tokenizer")
            load_success = self._load_model_components(model_name, device)
            results['stages']['model_loading'] = {
                'success': load_success,
                'model_loaded': self.model is not None,
                'tokenizer_loaded': self.tokenizer is not None
            }
            
            if not load_success:
                self.logger.error("❌ Model loading failed - cannot proceed with diagnosis")
                return results
            
            # Stage 2: Cross-encoder attestation
            self.logger.info("🔐 STAGE 2: Cross-Encoder Attestation")
            attestation_result = self.attestation_system.attest_cross_encoder(
                self.model, self.tokenizer, model_name, device, max_seq_len, truncation
            )
            results['stages']['attestation'] = {
                'success': not attestation_result.abort_required,
                'issues_count': len(attestation_result.issues_found),
                'issues': attestation_result.issues_found,
                'fixes': attestation_result.fix_recommendations,
                'details': attestation_result.attestation_log
            }
            
            if attestation_result.abort_required:
                self.logger.error("🛑 Attestation failed - critical issues found")
                self._activate_safe_mode("Attestation failure")
                return results
            
            # Stage 3: Synthetic test suite
            self.logger.info("🧪 STAGE 3: Synthetic Test Suite")
            synthetic_result = self.synthetic_tester.run_synthetic_tests(
                self.cross_encoder or self.model, self.tokenizer
            )
            results['stages']['synthetic_tests'] = {
                'success': synthetic_result.test_passed,
                'flat_scoring_detected': synthetic_result.flat_scoring_detected,
                'score_variance': synthetic_result.score_variance,
                'score_range': synthetic_result.score_range,
                'ranking_correct': synthetic_result.ranking_correct,
                'issues': synthetic_result.issues_found,
                'fixes': synthetic_result.fix_recommendations,
                'scores': synthetic_result.scores
            }
            
            if synthetic_result.flat_scoring_detected:
                self.logger.error("🚨 FLAT SCORING DETECTED - Activating safe mode")
                self._activate_safe_mode("Synthetic tests failed - flat scoring")
            
            # Stage 4: Input format debugging
            self.logger.info("🔍 STAGE 4: Input Format Debugging")
            input_result = self.input_debugger.debug_input_formatting(
                self.model, self.tokenizer, None, max_seq_len, truncation
            )
            results['stages']['input_debugging'] = {
                'success': input_result.formatting_correct,
                'truncation_working': input_result.truncation_working,
                'special_tokens_present': input_result.special_tokens_present,
                'token_type_ids_correct': input_result.token_type_ids_correct,
                'attention_mask_valid': input_result.attention_mask_valid,
                'issues': input_result.issues_found,
                'fixes': input_result.fix_recommendations,
                'sample_count': len(input_result.sample_inputs)
            }
            
            # Stage 5: Head architecture validation
            self.logger.info("🔧 STAGE 5: Head Architecture Validation")
            head_result = self.head_validator.validate_head_architecture(
                self.model, self.tokenizer, device
            )
            results['stages']['head_validation'] = {
                'success': head_result.head_architecture_correct,
                'precision_stable': head_result.precision_stable,
                'evaluation_mode_active': head_result.evaluation_mode_active,
                'output_processing_correct': head_result.output_processing_correct,
                'checkpoint_aligned': head_result.checkpoint_aligned,
                'issues': head_result.issues_found,
                'fixes': head_result.fix_recommendations,
                'diagnostics': head_result.head_diagnostics
            }
            
            # Stage 6: Overall analysis and recommendations
            self.logger.info("📊 STAGE 6: Analysis and Recommendations")
            overall_analysis = self._analyze_overall_results(results)
            results['overall_analysis'] = overall_analysis
            
            # Stage 7: Parameter recommendations
            param_recommendations = self._generate_parameter_recommendations(results)
            results['parameter_recommendations'] = param_recommendations
            
            # Log parameter recommendations immediately
            self._log_parameter_recommendations(param_recommendations)
            
        except Exception as e:
            self.logger.error(f"Diagnosis failed with error: {str(e)}")
            results['error'] = str(e)
            self._activate_safe_mode(f"Diagnosis error: {str(e)}")
        
        total_time = time.time() - total_start_time
        results['total_execution_time_seconds'] = total_time
        
        # Final summary
        self._log_final_summary(results, total_time)
        
        return results
    
    def _load_model_components(self, model_name: str, device: str) -> bool:
        """Load model and tokenizer components."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available")
            return False
        
        try:
            self.logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.logger.info(f"Loading model: {model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            # Create cross-encoder wrapper if possible
            try:
                self.cross_encoder = CrossEncoderReranker(model_name, device)
                self.logger.info("Cross-encoder wrapper created successfully")
            except Exception as e:
                self.logger.warning(f"Could not create cross-encoder wrapper: {e}")
                self.cross_encoder = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return False
    
    def _activate_safe_mode(self, reason: str):
        """Activate safe mode with given reason."""
        self.safe_mode.activate_safe_mode(reason)
        
        # Log safe mode activation details
        self.logger.warning("🛡️ SAFE MODE PARAMETERS:")
        stats = self.safe_mode.get_safe_mode_stats()
        for key, value in stats['config'].items():
            self.logger.warning(f"  {key}: {value}")
    
    def _analyze_overall_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall diagnostic results."""
        stages = results['stages']
        
        # Count successes and failures
        stage_results = {
            'total_stages': len(stages),
            'passed_stages': 0,
            'failed_stages': 0,
            'critical_issues': [],
            'warning_issues': [],
            'all_fixes': []
        }
        
        for stage_name, stage_data in stages.items():
            if isinstance(stage_data, dict) and 'success' in stage_data:
                if stage_data['success']:
                    stage_results['passed_stages'] += 1
                else:
                    stage_results['failed_stages'] += 1
                
                # Collect issues and fixes
                issues = stage_data.get('issues', [])
                fixes = stage_data.get('fixes', [])
                
                for issue in issues:
                    if 'CRITICAL' in issue:
                        stage_results['critical_issues'].append(f"{stage_name}: {issue}")
                    else:
                        stage_results['warning_issues'].append(f"{stage_name}: {issue}")
                
                stage_results['all_fixes'].extend([f"{stage_name}: {fix}" for fix in fixes])
        
        # Determine overall health
        if stage_results['failed_stages'] == 0 and not stage_results['critical_issues']:
            stage_results['overall_status'] = 'HEALTHY'
        elif stage_results['critical_issues']:
            stage_results['overall_status'] = 'CRITICAL'
        else:
            stage_results['overall_status'] = 'DEGRADED'
        
        return stage_results
    
    def _generate_parameter_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter adjustment recommendations."""
        recommendations = {
            'immediate_adjustments': {},
            'safe_mode_parameters': {},
            'long_term_fixes': []
        }
        
        # Immediate parameter adjustments
        recommendations['immediate_adjustments'] = {
            'K1': 5000,  # Increase candidate pool
            'K2': 1200,  # Increase reranking budget
            'dims': 768, # Use full dimensionality for code
            'diversity_delta': 0.0,  # Disable DPP temporarily
            'facility_gamma': 0.8,   # Emphasize facility-location
            'reasoning': 'Compensate for CE issues with larger candidate pools and disabled diversity'
        }
        
        # Safe mode parameters
        if self.safe_mode.is_safe_mode_active():
            stats = self.safe_mode.get_safe_mode_stats()
            recommendations['safe_mode_parameters'] = {
                'active': True,
                'bi_encoder_weight': stats['config']['bi_encoder_weight'],
                'bm25_weight': stats['config']['bm25_weight'],
                'k1_safe': stats['config']['k1_candidate_pool'],
                'k2_safe': stats['config']['k2_rerank_budget']
            }
        
        # Long-term fixes based on diagnostic results
        stages = results.get('stages', {})
        
        if stages.get('synthetic_tests', {}).get('flat_scoring_detected'):
            recommendations['long_term_fixes'].append(
                "Retrain or replace cross-encoder model - current model produces flat scores"
            )
        
        if not stages.get('input_debugging', {}).get('special_tokens_present'):
            recommendations['long_term_fixes'].append(
                "Fix tokenizer configuration - special tokens not properly formatted"
            )
        
        if not stages.get('head_validation', {}).get('precision_stable'):
            recommendations['long_term_fixes'].append(
                "Address precision issues - consider fp32 or model recalibration"
            )
        
        if not stages.get('attestation', {}).get('success'):
            recommendations['long_term_fixes'].append(
                "Resolve attestation failures - critical configuration issues detected"
            )
        
        return recommendations
    
    def _log_parameter_recommendations(self, recommendations: Dict[str, Any]):
        """Log parameter recommendations immediately."""
        self.logger.info("🎯 IMMEDIATE PARAMETER ADJUSTMENTS:")
        self.logger.info("-" * 40)
        
        immediate = recommendations['immediate_adjustments']
        for param, value in immediate.items():
            if param != 'reasoning':
                self.logger.info(f"  {param} = {value}")
        
        if 'reasoning' in immediate:
            self.logger.info(f"  Reasoning: {immediate['reasoning']}")
        
        # Safe mode parameters
        if recommendations.get('safe_mode_parameters', {}).get('active'):
            self.logger.info("🛡️ SAFE MODE ACTIVE - Use these parameters:")
            safe_params = recommendations['safe_mode_parameters']
            for param, value in safe_params.items():
                if param != 'active':
                    self.logger.info(f"  {param} = {value}")
        
        # Long-term fixes
        if recommendations['long_term_fixes']:
            self.logger.info("🔧 LONG-TERM FIXES NEEDED:")
            for i, fix in enumerate(recommendations['long_term_fixes'], 1):
                self.logger.info(f"  {i}. {fix}")
        
        self.logger.info("-" * 40)
    
    def _log_final_summary(self, results: Dict[str, Any], total_time: float):
        """Log final diagnostic summary."""
        self.logger.info("📋 FINAL DIAGNOSIS SUMMARY:")
        self.logger.info("=" * 50)
        
        overall = results.get('overall_analysis', {})
        status = overall.get('overall_status', 'UNKNOWN')
        
        if status == 'HEALTHY':
            self.logger.info("✅ Cross-encoder appears to be functioning correctly")
        elif status == 'DEGRADED':
            self.logger.warning("⚠️ Cross-encoder has issues but may be functional")  
        else:
            self.logger.error("❌ Cross-encoder has critical issues")
        
        self.logger.info(f"Total stages: {overall.get('total_stages', 0)}")
        self.logger.info(f"Passed: {overall.get('passed_stages', 0)}")
        self.logger.info(f"Failed: {overall.get('failed_stages', 0)}")
        self.logger.info(f"Critical issues: {len(overall.get('critical_issues', []))}")
        self.logger.info(f"Total execution time: {total_time:.1f} seconds")
        
        if self.safe_mode.is_safe_mode_active():
            self.logger.warning("🛡️ Safe mode is ACTIVE - using fallback scoring")
        
        self.logger.info("=" * 50)

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging format
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    # Suppress transformer warnings for cleaner output
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('tokenizers').setLevel(logging.WARNING)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'attestation': {
            'abort_on_critical_issues': True,
            'precision_checks': True,
            'head_validation': True
        },
        'synthetic_tests': {
            'min_score_std': 0.2,
            'min_score_range': 0.3,
            'num_test_iterations': 3
        },
        'input_debugging': {
            'max_samples_to_log': 5,
            'log_token_details': True,
            'check_token_type_ids': True
        },
        'head_validation': {
            'precision_test_iterations': 5,
            'require_eval_mode': True,
            'checkpoint_validation': True
        },
        'safe_mode': {
            'bi_encoder_weight': 0.6,
            'bm25_weight': 0.4,
            'facility_gamma': 0.8,
            'diversity_delta': 0.0,
            'k1_candidate_pool': 5000,
            'k2_rerank_budget': 1200
        }
    }

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-Encoder Debugging CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug a specific model
  python debug_cross_encoder.py --model cross-encoder/ms-marco-MiniLM-L-6-v2
  
  # Use custom configuration
  python debug_cross_encoder.py --model MODEL_NAME --config debug_config.json
  
  # Enable GPU and detailed logging
  python debug_cross_encoder.py --model MODEL_NAME --device cuda --log-level DEBUG
  
  # Save results to file
  python debug_cross_encoder.py --model MODEL_NAME --output results.json
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model identifier')
    parser.add_argument('--config', type=str,
                       help='Configuration JSON file path')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for model inference')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--truncation', type=str, default='longest_first',
                       choices=['longest_first', 'only_first', 'only_second'],
                       help='Truncation strategy')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str,
                       help='Log output file path')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results')
    parser.add_argument('--safe-mode-only', action='store_true',
                       help='Skip diagnostics and activate safe mode immediately')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
        
        # Initialize debugger
        debugger = CrossEncoderDebugger(config)
        
        if args.safe_mode_only:
            logger.info("Safe mode only - skipping diagnostics")
            debugger.safe_mode.activate_safe_mode("Safe mode requested via CLI")
            
            results = {
                'safe_mode_only': True,
                'safe_mode_stats': debugger.safe_mode.get_safe_mode_stats()
            }
        else:
            # Run complete diagnosis
            results = debugger.run_complete_diagnosis(
                args.model, args.device, args.max_length, args.truncation
            )
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        if not args.safe_mode_only:
            overall_status = results.get('overall_analysis', {}).get('overall_status')
            if overall_status == 'CRITICAL':
                sys.exit(1)
            elif overall_status == 'DEGRADED':
                sys.exit(2)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Debugging interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Debugging failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()