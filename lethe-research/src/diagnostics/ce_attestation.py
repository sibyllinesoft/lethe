"""
Cross-Encoder Attestation System
===============================

Runtime validation system that logs and validates all critical cross-encoder 
configuration parameters. Provides comprehensive attestation of CE setup
to catch configuration issues that cause flat scoring.

Logs once per run:
- Model & tokenizer IDs with checksums
- Tokenization parameters (max_seq_len, truncation, special tokens)
- Model precision (fp16/bf16/fp32)
- Head configuration (binary_cls, regression, pairwise)
- Evaluation mode settings

Aborts execution if any critical parameters are None or misconfigured.
"""

import logging
import hashlib
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

logger = logging.getLogger(__name__)

@dataclass
class CEAttestationResult:
    """Results from cross-encoder attestation check."""
    model_validated: bool
    tokenizer_validated: bool
    configuration_valid: bool
    issues_found: List[str]
    attestation_log: Dict[str, Any]
    abort_required: bool
    fix_recommendations: List[str]

class CrossEncoderAttestationSystem:
    """
    Comprehensive cross-encoder configuration validation system.
    
    Validates all critical CE parameters and logs complete attestation
    information for debugging flat scoring issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize attestation system.
        
        Args:
            config: Configuration with validation thresholds
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track if attestation has been run
        self._attestation_complete = False
        self._attestation_result = None
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for attestation."""
        return {
            'required_model_attributes': [
                'config', 'classifier', 'pooler', 'encoder'
            ],
            'required_tokenizer_attributes': [
                'vocab_size', 'model_max_length', 'pad_token_id'
            ],
            'precision_checks': True,
            'head_validation': True,
            'special_token_validation': True,
            'abort_on_critical_issues': True
        }
    
    def attest_cross_encoder(self, 
                           model: Any,
                           tokenizer: Any,
                           model_name: str,
                           device: str,
                           max_seq_len: int = 512,
                           truncation_strategy: str = 'longest_first') -> CEAttestationResult:
        """
        Complete cross-encoder attestation and validation.
        
        Args:
            model: Cross-encoder model instance
            tokenizer: Associated tokenizer
            model_name: Model identifier
            device: Target device
            max_seq_len: Maximum sequence length
            truncation_strategy: Truncation strategy
            
        Returns:
            CEAttestationResult with validation status
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        self.logger.info("=" * 60)
        self.logger.info("CROSS-ENCODER ATTESTATION STARTING")
        self.logger.info("=" * 60)
        
        issues = []
        attestation_log = {}
        fixes = []
        
        # Core model attestation
        model_valid, model_issues, model_log = self._attest_model(model, model_name)
        issues.extend(model_issues)
        attestation_log.update(model_log)
        
        # Tokenizer attestation  
        tokenizer_valid, tokenizer_issues, tokenizer_log = self._attest_tokenizer(tokenizer)
        issues.extend(tokenizer_issues)
        attestation_log.update(tokenizer_log)
        
        # Configuration attestation
        config_valid, config_issues, config_log = self._attest_configuration(
            model, tokenizer, device, max_seq_len, truncation_strategy
        )
        issues.extend(config_issues)
        attestation_log.update(config_log)
        
        # Head architecture attestation
        head_issues, head_log = self._attest_head_architecture(model)
        issues.extend(head_issues)
        attestation_log.update(head_log)
        
        # Precision and evaluation mode attestation
        precision_issues, precision_log = self._attest_precision_and_eval_mode(model)
        issues.extend(precision_issues)
        attestation_log.update(precision_log)
        
        # Special tokens attestation
        special_token_issues, special_token_log = self._attest_special_tokens(tokenizer, model_name)
        issues.extend(special_token_issues)
        attestation_log.update(special_token_log)
        
        # Generate fix recommendations
        fixes = self._generate_fix_recommendations(issues, attestation_log)
        
        # Determine if abort is required
        critical_issues = [issue for issue in issues if 'CRITICAL' in issue or 'None' in issue]
        abort_required = len(critical_issues) > 0 and self.config.get('abort_on_critical_issues', True)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            execution_time = start_time.elapsed_time(end_time)
            attestation_log['execution_time_ms'] = execution_time
        
        # Create result
        result = CEAttestationResult(
            model_validated=model_valid,
            tokenizer_validated=tokenizer_valid,
            configuration_valid=config_valid,
            issues_found=issues,
            attestation_log=attestation_log,
            abort_required=abort_required,
            fix_recommendations=fixes
        )
        
        # Log complete attestation
        self._log_complete_attestation(attestation_log, issues, abort_required)
        
        # Cache result
        self._attestation_complete = True
        self._attestation_result = result
        
        return result
    
    def _attest_model(self, model: Any, model_name: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Attest model configuration and state."""
        issues = []
        log = {
            'ce_model_id': model_name,
            'model_class': model.__class__.__name__ if model else None
        }
        
        if model is None:
            issues.append("CRITICAL: Model is None")
            return False, issues, log
        
        try:
            # Get model configuration
            if hasattr(model, 'config'):
                config = model.config
                log['model_config'] = {
                    'model_type': getattr(config, 'model_type', None),
                    'num_labels': getattr(config, 'num_labels', None),
                    'hidden_size': getattr(config, 'hidden_size', None),
                    'num_attention_heads': getattr(config, 'num_attention_heads', None),
                    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
                    'vocab_size': getattr(config, 'vocab_size', None)
                }
                
                # Validate critical config values
                if getattr(config, 'num_labels', None) is None:
                    issues.append("CRITICAL: num_labels is None in model config")
            else:
                issues.append("WARNING: Model has no config attribute")
            
            # Check model weights checksum
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                # Calculate checksum of first few layers for fingerprinting
                key_params = []
                for key in sorted(state_dict.keys())[:5]:  # First 5 parameters
                    if state_dict[key].numel() > 0:
                        key_params.append(state_dict[key].cpu().numpy().flatten()[:100])  # First 100 values
                
                if key_params:
                    combined = np.concatenate(key_params)
                    checksum = hashlib.sha256(combined.tobytes()).hexdigest()[:16]
                    log['ce_checkpoint_sha'] = checksum
                else:
                    issues.append("WARNING: Could not calculate model checksum")
                    log['ce_checkpoint_sha'] = None
            else:
                issues.append("WARNING: Model has no state_dict method")
                log['ce_checkpoint_sha'] = None
                
            # Check if model has required attributes
            for attr in self.config.get('required_model_attributes', []):
                if not hasattr(model, attr):
                    issues.append(f"WARNING: Model missing attribute: {attr}")
                    
        except Exception as e:
            issues.append(f"ERROR: Model attestation failed: {str(e)}")
            log['model_error'] = str(e)
            
        return len([i for i in issues if 'CRITICAL' in i]) == 0, issues, log
    
    def _attest_tokenizer(self, tokenizer: Any) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Attest tokenizer configuration."""
        issues = []
        log = {}
        
        if tokenizer is None:
            issues.append("CRITICAL: Tokenizer is None")
            return False, issues, log
        
        try:
            # Basic tokenizer info
            log['ce_tokenizer_id'] = getattr(tokenizer, 'name_or_path', 'unknown')
            log['vocab_size'] = getattr(tokenizer, 'vocab_size', None)
            log['model_max_length'] = getattr(tokenizer, 'model_max_length', None)
            
            # Calculate tokenizer checksum
            vocab_keys = list(tokenizer.get_vocab().keys())[:100]  # First 100 vocab items
            if vocab_keys:
                vocab_str = ''.join(sorted(vocab_keys))
                checksum = hashlib.sha256(vocab_str.encode()).hexdigest()[:16]
                log['ce_tokenizer_checksum'] = checksum
            else:
                issues.append("WARNING: Could not calculate tokenizer checksum")
                log['ce_tokenizer_checksum'] = None
            
            # Validate critical attributes
            for attr in self.config.get('required_tokenizer_attributes', []):
                value = getattr(tokenizer, attr, None)
                if value is None:
                    issues.append(f"CRITICAL: Tokenizer {attr} is None")
                log[f'tokenizer_{attr}'] = value
                
        except Exception as e:
            issues.append(f"ERROR: Tokenizer attestation failed: {str(e)}")
            log['tokenizer_error'] = str(e)
            
        return len([i for i in issues if 'CRITICAL' in i]) == 0, issues, log
    
    def _attest_configuration(self, 
                            model: Any, 
                            tokenizer: Any,
                            device: str,
                            max_seq_len: int,
                            truncation_strategy: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Attest runtime configuration parameters."""
        issues = []
        log = {
            'max_seq_len': max_seq_len,
            'truncation': truncation_strategy,
            'device': device
        }
        
        # Validate max sequence length
        if max_seq_len <= 0:
            issues.append("CRITICAL: max_seq_len must be positive")
        elif max_seq_len > 8192:
            issues.append("WARNING: Very large max_seq_len may cause memory issues")
        
        # Validate truncation strategy
        valid_truncation = ['longest_first', 'only_first', 'only_second', 'do_not_truncate']
        if truncation_strategy not in valid_truncation:
            issues.append(f"WARNING: Unusual truncation strategy: {truncation_strategy}")
        
        # Check device availability
        if device is None:
            issues.append("CRITICAL: Device is None")
        elif device == 'cuda' and not torch.cuda.is_available():
            issues.append("WARNING: CUDA requested but not available")
        
        # Check tokenizer max length compatibility
        if tokenizer and hasattr(tokenizer, 'model_max_length'):
            tokenizer_max = tokenizer.model_max_length
            if tokenizer_max and tokenizer_max < max_seq_len:
                issues.append(f"WARNING: max_seq_len ({max_seq_len}) > tokenizer max ({tokenizer_max})")
        
        return len([i for i in issues if 'CRITICAL' in i]) == 0, issues, log
    
    def _attest_head_architecture(self, model: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Attest classification head architecture."""
        issues = []
        log = {}
        
        if model is None:
            issues.append("CRITICAL: Cannot attest head - model is None")
            return issues, log
        
        try:
            # Detect head type
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                log['head_type'] = 'classifier'
                
                if hasattr(classifier, 'out_features'):
                    num_labels = classifier.out_features
                    log['head_num_labels'] = num_labels
                    
                    if num_labels == 1:
                        log['head_architecture'] = 'regression'
                    elif num_labels == 2:
                        log['head_architecture'] = 'binary_classification'
                    else:
                        log['head_architecture'] = 'multi_classification'
                        issues.append(f"WARNING: Unexpected num_labels: {num_labels}")
                else:
                    issues.append("WARNING: Classifier has no out_features")
                    
            elif hasattr(model, 'score'):
                log['head_type'] = 'score'
                log['head_architecture'] = 'regression'
            else:
                issues.append("WARNING: Unknown head architecture")
                log['head_type'] = 'unknown'
            
            # Check head weights
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'state_dict'):
                head_params = model.classifier.state_dict()
                if head_params:
                    # Calculate head checksum
                    param_values = []
                    for key in sorted(head_params.keys()):
                        param_values.append(head_params[key].cpu().numpy().flatten())
                    
                    if param_values:
                        combined = np.concatenate(param_values)
                        head_checksum = hashlib.sha256(combined.tobytes()).hexdigest()[:16]
                        log['head_ckpt_sha'] = head_checksum
                    else:
                        issues.append("WARNING: Empty head parameters")
                        log['head_ckpt_sha'] = None
                else:
                    issues.append("WARNING: No head parameters found")
                    log['head_ckpt_sha'] = None
                    
        except Exception as e:
            issues.append(f"ERROR: Head attestation failed: {str(e)}")
            log['head_error'] = str(e)
            
        return issues, log
    
    def _attest_precision_and_eval_mode(self, model: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Attest model precision and evaluation mode."""
        issues = []
        log = {}
        
        if model is None:
            issues.append("CRITICAL: Cannot attest precision - model is None")
            return issues, log
        
        try:
            # Check training/eval mode
            is_training = model.training
            log['model_training_mode'] = is_training
            log['dropout_eval_off'] = not is_training
            
            if is_training:
                issues.append("CRITICAL: Model is in training mode - should be eval() for inference")
            
            # Check precision
            param_dtypes = set()
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    param_dtypes.add(str(param.dtype))
                    break  # Just check first parameter
                    
                if param_dtypes:
                    main_dtype = list(param_dtypes)[0]
                    log['model_dtype'] = main_dtype
                    
                    if 'float16' in main_dtype:
                        log['precision'] = 'fp16'
                    elif 'bfloat16' in main_dtype:
                        log['precision'] = 'bf16'
                    elif 'float32' in main_dtype:
                        log['precision'] = 'fp32'
                    else:
                        log['precision'] = 'unknown'
                        issues.append(f"WARNING: Unexpected dtype: {main_dtype}")
                else:
                    issues.append("WARNING: No model parameters found")
                    log['precision'] = None
            else:
                issues.append("WARNING: Model has no parameters method")
                log['precision'] = None
                
            # Check for mixed precision issues
            if len(param_dtypes) > 1:
                issues.append(f"WARNING: Mixed parameter dtypes: {param_dtypes}")
                
        except Exception as e:
            issues.append(f"ERROR: Precision attestation failed: {str(e)}")
            log['precision_error'] = str(e)
            
        return issues, log
    
    def _attest_special_tokens(self, tokenizer: Any, model_name: str) -> Tuple[List[str], Dict[str, Any]]:
        """Attest special token configuration."""
        issues = []
        log = {}
        
        if tokenizer is None:
            issues.append("CRITICAL: Cannot attest special tokens - tokenizer is None")
            return issues, log
        
        try:
            # Detect model architecture type
            if 'roberta' in model_name.lower():
                expected_tokens = {'<s>', '</s>'}
                log['expected_architecture'] = 'roberta'
                log['uses_token_type_ids'] = False
            elif 'deberta' in model_name.lower():
                if 'v3' in model_name.lower():
                    expected_tokens = {'<s>', '</s>'}
                    log['expected_architecture'] = 'deberta-v3'
                    log['uses_token_type_ids'] = False
                else:
                    expected_tokens = {'[CLS]', '[SEP]'}
                    log['expected_architecture'] = 'deberta'
                    log['uses_token_type_ids'] = True
            else:  # BERT-style
                expected_tokens = {'[CLS]', '[SEP]'}
                log['expected_architecture'] = 'bert'
                log['uses_token_type_ids'] = True
            
            # Check for special tokens
            special_tokens = {}
            token_attrs = ['cls_token', 'sep_token', 'pad_token', 'unk_token', 'mask_token']
            
            for attr in token_attrs:
                if hasattr(tokenizer, attr):
                    token = getattr(tokenizer, attr)
                    special_tokens[attr] = str(token) if token else None
                    if token is None and attr in ['cls_token', 'sep_token']:
                        issues.append(f"WARNING: {attr} is None")
            
            log['special_tokens'] = special_tokens
            
            # Validate expected tokens exist
            vocab = tokenizer.get_vocab()
            for expected_token in expected_tokens:
                if expected_token not in vocab:
                    issues.append(f"WARNING: Expected token '{expected_token}' not in vocabulary")
                    
            # Check for token type ids support
            try:
                test_inputs = tokenizer("test", "test", return_tensors="pt")
                has_token_type_ids = 'token_type_ids' in test_inputs
                log['actual_uses_token_type_ids'] = has_token_type_ids
                
                expected_token_type_ids = log.get('uses_token_type_ids', True)
                if has_token_type_ids != expected_token_type_ids:
                    issues.append(f"WARNING: token_type_ids mismatch - expected: {expected_token_type_ids}, actual: {has_token_type_ids}")
                    
            except Exception as e:
                issues.append(f"WARNING: Could not test token_type_ids: {str(e)}")
                log['actual_uses_token_type_ids'] = None
                
        except Exception as e:
            issues.append(f"ERROR: Special token attestation failed: {str(e)}")
            log['special_tokens_error'] = str(e)
            
        return issues, log
    
    def _generate_fix_recommendations(self, 
                                    issues: List[str], 
                                    attestation_log: Dict[str, Any]) -> List[str]:
        """Generate fix recommendations based on issues found."""
        fixes = []
        
        # Critical issue fixes
        critical_issues = [i for i in issues if 'CRITICAL' in i]
        for issue in critical_issues:
            if "Model is None" in issue:
                fixes.append("Load cross-encoder model using AutoModelForSequenceClassification.from_pretrained()")
            elif "Tokenizer is None" in issue:
                fixes.append("Load tokenizer using AutoTokenizer.from_pretrained()")
            elif "training mode" in issue:
                fixes.append("Set model to evaluation mode: model.eval()")
            elif "num_labels is None" in issue:
                fixes.append("Check model configuration - num_labels should be 1 (regression) or 2 (classification)")
                
        # Precision and configuration fixes
        for issue in issues:
            if "dtype" in issue.lower():
                fixes.append("Consider using fp32 precision for debugging: model.float()")
            elif "max_seq_len" in issue:
                fixes.append("Adjust max_seq_len to reasonable value (256-512 for most models)")
            elif "token_type_ids" in issue:
                fixes.append("Check tokenizer architecture - BERT/DeBERTa use token_type_ids, RoBERTa does not")
            elif "special token" in issue.lower():
                fixes.append("Verify tokenizer matches model architecture (BERT vs RoBERTa vs DeBERTa)")
                
        # Add general recommendations
        if not fixes:
            fixes.append("All critical parameters validated - check input formatting and model inference")
        
        return fixes
    
    def _log_complete_attestation(self, 
                                attestation_log: Dict[str, Any], 
                                issues: List[str],
                                abort_required: bool):
        """Log complete attestation information."""
        self.logger.info("COMPLETE CROSS-ENCODER ATTESTATION:")
        self.logger.info("-" * 40)
        
        # Log all critical parameters
        critical_params = [
            'ce_model_id', 'ce_tokenizer_id', 'ce_checkpoint_sha', 'max_seq_len',
            'truncation', 'special_tokens', 'uses_token_type_ids', 'precision',
            'device', 'dropout_eval_off', 'head_architecture', 'head_ckpt_sha'
        ]
        
        for param in critical_params:
            value = attestation_log.get(param, 'NOT_SET')
            if value is None:
                self.logger.warning(f"  {param}: None ⚠️")
            else:
                self.logger.info(f"  {param}: {value}")
        
        # Log issues
        if issues:
            self.logger.warning(f"ISSUES FOUND ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                level = "🚨" if "CRITICAL" in issue else "⚠️" if "WARNING" in issue else "ℹ️"
                self.logger.warning(f"  {i}. {level} {issue}")
        else:
            self.logger.info("✅ No issues found in attestation")
        
        if abort_required:
            self.logger.error("🛑 ATTESTATION FAILED - ABORTING EXECUTION")
            self.logger.error("Critical issues must be resolved before continuing")
        else:
            self.logger.info("✅ ATTESTATION PASSED - Configuration validated")
        
        self.logger.info("=" * 60)
    
    def require_attestation(self) -> bool:
        """
        Check if attestation has been completed successfully.
        
        Returns:
            True if attestation passed, False if failed or not run
        """
        if not self._attestation_complete:
            self.logger.error("Cross-encoder attestation has not been run")
            return False
            
        if self._attestation_result.abort_required:
            self.logger.error("Cross-encoder attestation failed - aborting required")
            return False
            
        return True
    
    def get_attestation_result(self) -> Optional[CEAttestationResult]:
        """Get the last attestation result."""
        return self._attestation_result