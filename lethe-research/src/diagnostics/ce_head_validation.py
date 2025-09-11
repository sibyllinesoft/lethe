"""
Cross-Encoder Head Wiring & Precision Validation
===============================================

Validates cross-encoder head architecture and precision settings to catch
issues that cause flat scoring:

1. Head wiring validation (binary classifier vs regression head)
2. Output logit processing (softmax vs direct regression)
3. Precision testing (fp16 vs fp32 underflow/saturation)
4. Checkpoint alignment (head layer keys and shapes)
5. Evaluation mode enforcement (dropout frozen)
6. Attention mask and token type ID shape validation

Ensures the classification/regression head is correctly wired and configured.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class HeadValidationResult:
    """Results from cross-encoder head validation."""
    head_architecture_correct: bool
    precision_stable: bool
    evaluation_mode_active: bool
    output_processing_correct: bool
    checkpoint_aligned: bool
    issues_found: List[str]
    head_diagnostics: Dict[str, Any]
    fix_recommendations: List[str]

class CrossEncoderHeadValidator:
    """
    Validate cross-encoder head architecture and precision settings.
    
    Ensures the classification head is properly configured and catches
    common issues that lead to flat or incorrect scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize head validator.
        
        Args:
            config: Configuration for validation thresholds
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for head validation."""
        return {
            'precision_test_iterations': 5,   # Iterations for precision stability testing
            'underflow_threshold': 1e-7,      # Threshold for detecting underflow
            'saturation_threshold': 1e3,      # Threshold for detecting saturation
            'output_variance_threshold': 1e-8, # Minimum variance in outputs
            'dropout_tolerance': 1e-6,        # Tolerance for dropout variation
            'checkpoint_validation': True,     # Validate checkpoint integrity
            'require_eval_mode': True,        # Require model.eval() mode
            'test_precision_types': ['fp32', 'fp16']  # Precision types to test
        }
    
    def validate_head_architecture(self, 
                                 model: Any,
                                 tokenizer: Any,
                                 device: str = 'cpu') -> HeadValidationResult:
        """
        Comprehensive head architecture and precision validation.
        
        Args:
            model: Cross-encoder model
            tokenizer: Associated tokenizer
            device: Target device for testing
            
        Returns:
            HeadValidationResult with validation status
        """
        self.logger.info("🔧 Starting Cross-Encoder Head Validation")
        self.logger.info("=" * 50)
        
        issues = []
        fixes = []
        head_diagnostics = {}
        
        if model is None:
            return HeadValidationResult(
                head_architecture_correct=False,
                precision_stable=False,
                evaluation_mode_active=False,
                output_processing_correct=False,
                checkpoint_aligned=False,
                issues_found=["CRITICAL: Model is None"],
                head_diagnostics={},
                fix_recommendations=["Provide valid cross-encoder model instance"]
            )
        
        # 1. Analyze head architecture
        arch_issues, arch_diagnostics = self._analyze_head_architecture(model)
        issues.extend(arch_issues)
        head_diagnostics.update(arch_diagnostics)
        
        # 2. Validate evaluation mode
        eval_issues, eval_diagnostics = self._validate_evaluation_mode(model)
        issues.extend(eval_issues)
        head_diagnostics.update(eval_diagnostics)
        
        # 3. Test precision stability
        precision_issues, precision_diagnostics = self._test_precision_stability(model, tokenizer, device)
        issues.extend(precision_issues)
        head_diagnostics.update(precision_diagnostics)
        
        # 4. Validate output processing
        output_issues, output_diagnostics = self._validate_output_processing(model, tokenizer, device)
        issues.extend(output_issues)
        head_diagnostics.update(output_diagnostics)
        
        # 5. Check checkpoint alignment
        checkpoint_issues, checkpoint_diagnostics = self._validate_checkpoint_alignment(model)
        issues.extend(checkpoint_issues)
        head_diagnostics.update(checkpoint_diagnostics)
        
        # 6. Test input tensor shapes and masks
        shape_issues, shape_diagnostics = self._validate_tensor_shapes(model, tokenizer, device)
        issues.extend(shape_issues)
        head_diagnostics.update(shape_diagnostics)
        
        # Generate fix recommendations
        fixes = self._generate_head_validation_fixes(issues, head_diagnostics)
        
        # Determine validation status
        critical_issues = [i for i in issues if 'CRITICAL' in i]
        arch_correct = len([i for i in arch_issues if 'CRITICAL' in i]) == 0
        precision_stable = len([i for i in precision_issues if 'CRITICAL' in i or 'underflow' in i.lower() or 'saturation' in i.lower()]) == 0
        eval_mode = len([i for i in eval_issues if 'CRITICAL' in i]) == 0
        output_correct = len([i for i in output_issues if 'CRITICAL' in i]) == 0
        checkpoint_aligned = len([i for i in checkpoint_issues if 'CRITICAL' in i]) == 0
        
        result = HeadValidationResult(
            head_architecture_correct=arch_correct,
            precision_stable=precision_stable,
            evaluation_mode_active=eval_mode,
            output_processing_correct=output_correct,
            checkpoint_aligned=checkpoint_aligned,
            issues_found=issues,
            head_diagnostics=head_diagnostics,
            fix_recommendations=fixes
        )
        
        # Log validation summary
        self._log_validation_summary(result)
        
        return result
    
    def _analyze_head_architecture(self, model: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Analyze classification head architecture."""
        issues = []
        diagnostics = {}
        
        try:
            # Get model configuration
            if hasattr(model, 'config'):
                config = model.config
                num_labels = getattr(config, 'num_labels', None)
                diagnostics['num_labels'] = num_labels
                
                if num_labels is None:
                    issues.append("CRITICAL: num_labels is None in model config")
                elif num_labels == 1:
                    diagnostics['head_type'] = 'regression'
                    self.logger.info("Detected regression head (1 output)")
                elif num_labels == 2:
                    diagnostics['head_type'] = 'binary_classification'
                    self.logger.info("Detected binary classification head (2 outputs)")
                else:
                    diagnostics['head_type'] = 'multi_classification'
                    issues.append(f"WARNING: Unexpected num_labels for cross-encoder: {num_labels}")
            else:
                issues.append("WARNING: Model config not accessible")
                diagnostics['num_labels'] = None
                diagnostics['head_type'] = 'unknown'
            
            # Inspect classifier layer
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                diagnostics['has_classifier'] = True
                
                # Get classifier properties
                if hasattr(classifier, 'in_features'):
                    diagnostics['classifier_in_features'] = classifier.in_features
                if hasattr(classifier, 'out_features'):
                    diagnostics['classifier_out_features'] = classifier.out_features
                if hasattr(classifier, 'bias'):
                    has_bias = classifier.bias is not None
                    diagnostics['classifier_has_bias'] = has_bias
                    if not has_bias:
                        issues.append("WARNING: Classifier has no bias term")
                
                # Check classifier weights
                if hasattr(classifier, 'weight'):
                    weight = classifier.weight
                    if weight is not None:
                        weight_stats = {
                            'shape': list(weight.shape),
                            'mean': float(weight.mean().item()),
                            'std': float(weight.std().item()),
                            'min': float(weight.min().item()),
                            'max': float(weight.max().item())
                        }
                        diagnostics['classifier_weights'] = weight_stats
                        
                        # Check for degenerate weights
                        if weight_stats['std'] < 1e-8:
                            issues.append("CRITICAL: Classifier weights have zero variance (all identical)")
                        if abs(weight_stats['mean']) < 1e-8 and weight_stats['std'] < 1e-6:
                            issues.append("WARNING: Classifier weights appear to be zero-initialized")
                    else:
                        issues.append("CRITICAL: Classifier weight is None")
                else:
                    issues.append("WARNING: Classifier has no weight attribute")
                    
            elif hasattr(model, 'score'):
                diagnostics['has_classifier'] = False
                diagnostics['has_score_layer'] = True
                issues.append("INFO: Model uses score layer instead of classifier")
            else:
                issues.append("CRITICAL: No classifier or score layer found")
                diagnostics['has_classifier'] = False
                
        except Exception as e:
            issues.append(f"ERROR: Head architecture analysis failed: {str(e)}")
            diagnostics['analysis_error'] = str(e)
        
        return issues, diagnostics
    
    def _validate_evaluation_mode(self, model: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Validate model is in evaluation mode."""
        issues = []
        diagnostics = {}
        
        try:
            is_training = model.training
            diagnostics['model_training_mode'] = is_training
            
            if is_training:
                issues.append("CRITICAL: Model is in training mode - must call model.eval()")
            
            # Check dropout modules specifically
            dropout_modules = []
            for name, module in model.named_modules():
                if 'dropout' in name.lower() or isinstance(module, torch.nn.Dropout):
                    dropout_modules.append({
                        'name': name,
                        'training': module.training,
                        'p': getattr(module, 'p', None)
                    })
            
            diagnostics['dropout_modules'] = dropout_modules
            
            # Verify dropout is disabled
            active_dropout = [m for m in dropout_modules if m['training']]
            if active_dropout:
                issues.append(f"WARNING: {len(active_dropout)} dropout modules still in training mode")
                diagnostics['active_dropout_count'] = len(active_dropout)
            
        except Exception as e:
            issues.append(f"ERROR: Evaluation mode validation failed: {str(e)}")
            diagnostics['eval_mode_error'] = str(e)
        
        return issues, diagnostics
    
    def _test_precision_stability(self, model: Any, tokenizer: Any, device: str) -> Tuple[List[str], Dict[str, Any]]:
        """Test precision stability across different inputs."""
        issues = []
        diagnostics = {}
        
        if tokenizer is None:
            issues.append("WARNING: No tokenizer provided for precision testing")
            return issues, diagnostics
        
        try:
            # Create test inputs
            test_query = "test query for precision"
            test_doc = "test document for precision stability validation"
            
            # Test different precisions if model supports it
            precision_results = {}
            
            for precision in self.config['test_precision_types']:
                try:
                    if precision == 'fp16' and device == 'cpu':
                        continue  # Skip fp16 on CPU
                    
                    # Convert model precision
                    if precision == 'fp16':
                        model = model.half()
                    elif precision == 'fp32':
                        model = model.float()
                    
                    # Run multiple iterations
                    outputs = []
                    for i in range(self.config['precision_test_iterations']):
                        inputs = tokenizer(
                            test_query, test_doc,
                            truncation=True,
                            padding=True,
                            max_length=256,
                            return_tensors="pt"
                        )
                        
                        if device != 'cpu':
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            output = model(**inputs)
                            
                            # Extract logits
                            if hasattr(output, 'logits'):
                                logits = output.logits
                            else:
                                logits = output[0]
                            
                            outputs.append(logits.cpu().numpy())
                    
                    # Analyze outputs for stability
                    outputs_array = np.array(outputs)
                    variance = np.var(outputs_array)
                    mean_output = np.mean(outputs_array)
                    std_output = np.std(outputs_array)
                    
                    precision_results[precision] = {
                        'variance': float(variance),
                        'mean': float(mean_output),
                        'std': float(std_output),
                        'min': float(np.min(outputs_array)),
                        'max': float(np.max(outputs_array))
                    }
                    
                    # Check for precision issues
                    if variance > self.config['dropout_tolerance'] and not model.training:
                        issues.append(f"WARNING: High output variance in {precision} ({variance:.2e}) - possible non-deterministic behavior")
                    
                    if np.any(np.abs(outputs_array) < self.config['underflow_threshold']):
                        issues.append(f"WARNING: Potential underflow detected in {precision}")
                    
                    if np.any(np.abs(outputs_array) > self.config['saturation_threshold']):
                        issues.append(f"WARNING: Potential saturation detected in {precision}")
                    
                    if std_output < self.config['output_variance_threshold']:
                        issues.append(f"CRITICAL: Outputs have zero variance in {precision} - model producing constant values")
                    
                except Exception as e:
                    issues.append(f"ERROR: Precision testing failed for {precision}: {str(e)}")
                    precision_results[precision] = {'error': str(e)}
            
            diagnostics['precision_test_results'] = precision_results
            
        except Exception as e:
            issues.append(f"ERROR: Precision stability testing failed: {str(e)}")
            diagnostics['precision_test_error'] = str(e)
        
        return issues, diagnostics
    
    def _validate_output_processing(self, model: Any, tokenizer: Any, device: str) -> Tuple[List[str], Dict[str, Any]]:
        """Validate output processing and score extraction."""
        issues = []
        diagnostics = {}
        
        if tokenizer is None:
            issues.append("WARNING: No tokenizer provided for output validation")
            return issues, diagnostics
        
        try:
            # Create test input
            test_inputs = tokenizer(
                "test query", "test document",
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            
            if device != 'cpu':
                test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**test_inputs)
                
                # Analyze raw outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    diagnostics['output_has_logits_attr'] = True
                else:
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    diagnostics['output_has_logits_attr'] = False
                
                logits_np = logits.cpu().numpy()
                diagnostics['raw_logits_shape'] = list(logits.shape)
                diagnostics['raw_logits_stats'] = {
                    'mean': float(np.mean(logits_np)),
                    'std': float(np.std(logits_np)),
                    'min': float(np.min(logits_np)),
                    'max': float(np.max(logits_np))
                }
                
                # Test different output processing methods
                processing_methods = {}
                
                # Method 1: Regression (direct logit)
                if logits.shape[-1] == 1:
                    regression_score = logits.squeeze(-1).item()
                    processing_methods['regression'] = regression_score
                    diagnostics['recommended_processing'] = 'regression'
                
                # Method 2: Binary classification (softmax)
                if logits.shape[-1] == 2:
                    softmax_scores = torch.softmax(logits, dim=-1)
                    positive_score = softmax_scores[0, -1].item()  # Last class (positive)
                    processing_methods['binary_softmax'] = positive_score
                    diagnostics['recommended_processing'] = 'binary_classification'
                    
                    # Also test logit differences
                    logit_diff = logits[0, 1].item() - logits[0, 0].item()
                    processing_methods['logit_difference'] = logit_diff
                
                # Method 3: Raw logits (no processing)
                if logits.numel() == 1:
                    raw_score = logits.item()
                    processing_methods['raw_logit'] = raw_score
                elif logits.shape[-1] > 1:
                    raw_score = logits[0, -1].item()  # Last dimension
                    processing_methods['raw_last_logit'] = raw_score
                
                diagnostics['processing_methods'] = processing_methods
                
                # Validate processing method consistency
                if len(processing_methods) == 0:
                    issues.append("CRITICAL: No valid output processing method found")
                elif len(set(np.round(list(processing_methods.values()), 3))) == 1:
                    issues.append("WARNING: All processing methods yield identical scores")
                
                # Check for NaN or infinite outputs
                if torch.any(torch.isnan(logits)):
                    issues.append("CRITICAL: Model outputs contain NaN values")
                if torch.any(torch.isinf(logits)):
                    issues.append("CRITICAL: Model outputs contain infinite values")
                
                # Validate output ranges
                logit_range = float(torch.max(logits) - torch.min(logits))
                if logit_range < 1e-6:
                    issues.append("CRITICAL: Output logits have zero range - model not learning")
                
        except Exception as e:
            issues.append(f"ERROR: Output processing validation failed: {str(e)}")
            diagnostics['output_processing_error'] = str(e)
        
        return issues, diagnostics
    
    def _validate_checkpoint_alignment(self, model: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Validate checkpoint integrity and layer alignment."""
        issues = []
        diagnostics = {}
        
        if not self.config['checkpoint_validation']:
            return issues, diagnostics
        
        try:
            # Get model state dict
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                
                # Analyze layer keys
                all_keys = list(state_dict.keys())
                classifier_keys = [k for k in all_keys if 'classifier' in k or 'score' in k]
                
                diagnostics['total_parameters'] = len(all_keys)
                diagnostics['classifier_parameters'] = len(classifier_keys)
                diagnostics['classifier_keys'] = classifier_keys
                
                # Check classifier parameters specifically
                if classifier_keys:
                    for key in classifier_keys:
                        param = state_dict[key]
                        param_stats = {
                            'shape': list(param.shape),
                            'dtype': str(param.dtype),
                            'requires_grad': param.requires_grad if hasattr(param, 'requires_grad') else None,
                            'mean': float(param.mean().item()) if param.numel() > 0 else 0.0,
                            'std': float(param.std().item()) if param.numel() > 1 else 0.0
                        }
                        diagnostics[f'param_{key}'] = param_stats
                        
                        # Check for parameter issues
                        if param.numel() == 0:
                            issues.append(f"WARNING: Parameter {key} is empty")
                        elif param_stats['std'] < 1e-8:
                            issues.append(f"WARNING: Parameter {key} has zero variance")
                
                # Calculate checkpoint fingerprint
                key_params = []
                for key in sorted(classifier_keys):
                    if key in state_dict:
                        param = state_dict[key]
                        if param.numel() > 0:
                            # Use first few values for fingerprinting
                            flat_param = param.flatten()
                            sample_size = min(10, flat_param.numel())
                            key_params.append(flat_param[:sample_size].cpu().numpy())
                
                if key_params:
                    combined = np.concatenate(key_params)
                    checksum = hashlib.sha256(combined.tobytes()).hexdigest()[:16]
                    diagnostics['head_checkpoint_fingerprint'] = checksum
                else:
                    issues.append("WARNING: No classifier parameters found for fingerprinting")
                    diagnostics['head_checkpoint_fingerprint'] = None
            else:
                issues.append("WARNING: Model has no state_dict method")
            
        except Exception as e:
            issues.append(f"ERROR: Checkpoint validation failed: {str(e)}")
            diagnostics['checkpoint_validation_error'] = str(e)
        
        return issues, diagnostics
    
    def _validate_tensor_shapes(self, model: Any, tokenizer: Any, device: str) -> Tuple[List[str], Dict[str, Any]]:
        """Validate input tensor shapes and attention masks."""
        issues = []
        diagnostics = {}
        
        if tokenizer is None:
            issues.append("WARNING: No tokenizer provided for shape validation")
            return issues, diagnostics
        
        try:
            # Test with different input combinations
            test_cases = [
                ("short query", "short doc"),
                ("", "empty query test"),
                ("empty doc test", ""),
                ("medium length query with several words", "medium length document with multiple sentences for testing")
            ]
            
            shape_results = []
            
            for i, (query, doc) in enumerate(test_cases):
                try:
                    # Tokenize
                    inputs = tokenizer(
                        query, doc,
                        truncation=True,
                        padding=True,
                        max_length=256,
                        return_tensors="pt"
                    )
                    
                    # Analyze input shapes
                    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
                    
                    # Validate shapes are consistent
                    input_ids_shape = inputs['input_ids'].shape
                    attention_mask_shape = inputs['attention_mask'].shape
                    
                    if input_ids_shape != attention_mask_shape:
                        issues.append(f"ERROR: Shape mismatch - input_ids: {input_ids_shape}, attention_mask: {attention_mask_shape}")
                    
                    # Check token type IDs if present
                    if 'token_type_ids' in inputs:
                        token_type_ids_shape = inputs['token_type_ids'].shape
                        if input_ids_shape != token_type_ids_shape:
                            issues.append(f"ERROR: token_type_ids shape mismatch: {token_type_ids_shape}")
                    
                    # Move to device and test inference
                    if device != 'cpu':
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                        if hasattr(outputs, 'logits'):
                            output_shape = list(outputs.logits.shape)
                        else:
                            output_shape = list(outputs[0].shape) if isinstance(outputs, (tuple, list)) else list(outputs.shape)
                    
                    shape_result = {
                        'test_case': i,
                        'query_length': len(query),
                        'doc_length': len(doc),
                        'input_shapes': input_shapes,
                        'output_shape': output_shape,
                        'success': True
                    }
                    
                except Exception as e:
                    shape_result = {
                        'test_case': i,
                        'query_length': len(query),
                        'doc_length': len(doc),
                        'error': str(e),
                        'success': False
                    }
                    issues.append(f"ERROR: Shape validation failed for test case {i}: {str(e)}")
                
                shape_results.append(shape_result)
            
            diagnostics['shape_validation_results'] = shape_results
            
            # Check consistency across test cases
            successful_cases = [r for r in shape_results if r['success']]
            if successful_cases:
                output_shapes = [r['output_shape'] for r in successful_cases]
                if len(set(tuple(shape) for shape in output_shapes)) > 1:
                    issues.append("WARNING: Inconsistent output shapes across test cases")
            
        except Exception as e:
            issues.append(f"ERROR: Tensor shape validation failed: {str(e)}")
            diagnostics['shape_validation_error'] = str(e)
        
        return issues, diagnostics
    
    def _generate_head_validation_fixes(self, 
                                      issues: List[str],
                                      diagnostics: Dict[str, Any]) -> List[str]:
        """Generate fix recommendations for head validation issues."""
        fixes = []
        
        # Critical issue fixes
        for issue in issues:
            if "training mode" in issue:
                fixes.append("Call model.eval() to set model to evaluation mode")
            elif "num_labels is None" in issue:
                fixes.append("Check model configuration - set num_labels to 1 (regression) or 2 (classification)")
            elif "zero variance" in issue and "weights" in issue:
                fixes.append("Classifier weights are degenerate - reload model or check checkpoint")
            elif "constant values" in issue:
                fixes.append("Model producing constant outputs - check weights, precision, or model loading")
            elif "NaN" in issue:
                fixes.append("Model outputs contain NaN - check for gradient explosion or numerical instability")
            elif "infinite" in issue:
                fixes.append("Model outputs contain infinite values - check for overflow or division by zero")
            elif "No valid output processing" in issue:
                fixes.append("Cannot extract scores from model outputs - check output format and processing")
        
        # Architecture-specific fixes
        head_type = diagnostics.get('head_type')
        if head_type == 'regression':
            fixes.append("Using regression head - extract scores with logits.squeeze(-1)")
        elif head_type == 'binary_classification':
            fixes.append("Using binary classification head - extract scores with softmax(logits)[:, -1]")
        
        # Precision fixes
        precision_results = diagnostics.get('precision_test_results', {})
        if any('underflow' in issue.lower() for issue in issues):
            fixes.append("Underflow detected - try fp32 precision or gradient clipping")
        if any('saturation' in issue.lower() for issue in issues):
            fixes.append("Saturation detected - check input scaling or model configuration")
        
        # General recommendations
        if any('dropout' in issue.lower() for issue in issues):
            fixes.append("Disable dropout for inference: model.eval() and set dropout.training = False")
        
        if any('shape' in issue.lower() for issue in issues):
            fixes.append("Fix tensor shape mismatches - check tokenizer output format")
        
        if not fixes:
            fixes.append("Head validation passed - architecture and precision appear correct")
        
        return fixes
    
    def _log_validation_summary(self, result: HeadValidationResult):
        """Log summary of head validation results."""
        self.logger.info("🔧 HEAD VALIDATION SUMMARY:")
        self.logger.info("-" * 40)
        
        # Overall status
        all_passed = (result.head_architecture_correct and 
                     result.precision_stable and 
                     result.evaluation_mode_active and 
                     result.output_processing_correct and 
                     result.checkpoint_aligned)
        
        status_emoji = "✅" if all_passed else "❌"
        self.logger.info(f"  Overall validation: {status_emoji} {'PASSED' if all_passed else 'ISSUES FOUND'}")
        
        # Individual validations
        validations = [
            ("Head architecture", result.head_architecture_correct),
            ("Precision stable", result.precision_stable),
            ("Evaluation mode", result.evaluation_mode_active),
            ("Output processing", result.output_processing_correct),
            ("Checkpoint aligned", result.checkpoint_aligned)
        ]
        
        for validation_name, status in validations:
            emoji = "✅" if status else "❌"
            self.logger.info(f"  {validation_name}: {emoji}")
        
        # Key diagnostics
        diagnostics = result.head_diagnostics
        if 'head_type' in diagnostics:
            self.logger.info(f"  Head type: {diagnostics['head_type']}")
        if 'num_labels' in diagnostics:
            self.logger.info(f"  Output labels: {diagnostics['num_labels']}")
        if 'recommended_processing' in diagnostics:
            self.logger.info(f"  Processing method: {diagnostics['recommended_processing']}")
        
        # Issues
        if result.issues_found:
            self.logger.warning(f"Issues found ({len(result.issues_found)}):")
            for issue in result.issues_found:
                level = "🚨" if "CRITICAL" in issue else "⚠️" if "WARNING" in issue else "ℹ️"
                self.logger.warning(f"  • {level} {issue}")
        
        self.logger.info("=" * 50)