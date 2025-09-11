"""
Cross-Encoder Input Format & Truncation Debugging
================================================

Validates cross-encoder input formatting and tokenization to catch common
issues that lead to flat scoring:

1. Correct token format (BERT vs RoBERTa vs DeBERTa styles)
2. Proper special token positioning ([CLS], [SEP] or <s>, </s>)  
3. Token type IDs handling
4. Truncation behavior (longest_first vs only_second)
5. Attention mask validation
6. Sequence length analysis

Logs actual tokenized inputs for first 5 pairs to enable manual inspection.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class InputDebuggingResult:
    """Results from input format debugging."""
    formatting_correct: bool
    truncation_working: bool
    special_tokens_present: bool
    token_type_ids_correct: bool
    attention_mask_valid: bool
    issues_found: List[str]
    sample_inputs: List[Dict[str, Any]]
    fix_recommendations: List[str]

class CrossEncoderInputDebugger:
    """
    Debug cross-encoder input formatting and tokenization.
    
    Validates that query-document pairs are correctly formatted for the
    specific model architecture and catches tokenization issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize input debugger.
        
        Args:
            config: Configuration for debugging parameters
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for input debugging."""
        return {
            'max_samples_to_log': 5,          # Number of sample inputs to log
            'expected_max_length': 512,       # Expected maximum sequence length
            'min_attention_ratio': 0.5,       # Minimum ratio of non-padding tokens
            'truncation_test_length': 1000,   # Length for truncation testing
            'special_token_validation': True,  # Validate special tokens
            'log_token_details': True,        # Log detailed tokenization
            'check_token_type_ids': True      # Validate token type IDs
        }
    
    def debug_input_formatting(self, 
                             cross_encoder: Any,
                             tokenizer: Any,
                             test_pairs: Optional[List[Tuple[str, str]]] = None,
                             max_seq_len: int = 512,
                             truncation: str = 'longest_first') -> InputDebuggingResult:
        """
        Debug cross-encoder input formatting and tokenization.
        
        Args:
            cross_encoder: Cross-encoder model
            tokenizer: Tokenizer instance  
            test_pairs: Optional test pairs (will generate if None)
            max_seq_len: Maximum sequence length
            truncation: Truncation strategy
            
        Returns:
            InputDebuggingResult with validation status
        """
        self.logger.info("🔍 Starting Cross-Encoder Input Debugging")
        self.logger.info("=" * 50)
        
        issues = []
        fixes = []
        
        # Generate test pairs if not provided
        if test_pairs is None:
            test_pairs = self._generate_debug_test_pairs()
        
        # Get tokenizer if not provided
        if tokenizer is None:
            tokenizer = self._extract_tokenizer(cross_encoder)
            if tokenizer is None:
                return InputDebuggingResult(
                    formatting_correct=False,
                    truncation_working=False,
                    special_tokens_present=False,
                    token_type_ids_correct=False,
                    attention_mask_valid=False,
                    issues_found=["CRITICAL: No tokenizer available for debugging"],
                    sample_inputs=[],
                    fix_recommendations=["Provide tokenizer instance or ensure cross_encoder has tokenizer attribute"]
                )
        
        # Detect model architecture
        model_arch = self._detect_model_architecture(tokenizer, cross_encoder)
        self.logger.info(f"Detected model architecture: {model_arch}")
        
        # Tokenize test pairs and analyze
        sample_inputs = []
        tokenization_issues = []
        
        for i, (query, doc) in enumerate(test_pairs[:self.config['max_samples_to_log']]):
            try:
                # Tokenize the pair
                tokenized = self._tokenize_and_analyze(
                    tokenizer, query, doc, max_seq_len, truncation, model_arch
                )
                
                sample_inputs.append(tokenized)
                
                # Log detailed tokenization for first few pairs
                if i < 3:
                    self._log_tokenization_details(tokenized, i)
                
                # Validate this tokenization
                pair_issues = self._validate_tokenization(tokenized, model_arch)
                tokenization_issues.extend([f"Pair {i}: {issue}" for issue in pair_issues])
                
            except Exception as e:
                issues.append(f"ERROR: Tokenization failed for pair {i}: {str(e)}")
        
        issues.extend(tokenization_issues)
        
        # Run specific validation tests
        special_tokens_ok = self._validate_special_tokens(sample_inputs, model_arch)
        if not special_tokens_ok:
            issues.append("WARNING: Special token positioning may be incorrect")
        
        token_type_ids_ok = self._validate_token_type_ids(sample_inputs, model_arch)
        if not token_type_ids_ok:
            issues.append("WARNING: Token type IDs validation failed")
        
        attention_mask_ok = self._validate_attention_masks(sample_inputs)
        if not attention_mask_ok:
            issues.append("WARNING: Attention mask validation failed")
        
        truncation_ok = self._test_truncation_behavior(tokenizer, max_seq_len, truncation)
        if not truncation_ok:
            issues.append("WARNING: Truncation behavior unexpected")
        
        # Generate fix recommendations
        fixes = self._generate_input_debugging_fixes(issues, model_arch, sample_inputs)
        
        # Determine overall status
        critical_issues = [i for i in issues if 'CRITICAL' in i or 'ERROR' in i]
        formatting_correct = len(critical_issues) == 0
        
        result = InputDebuggingResult(
            formatting_correct=formatting_correct,
            truncation_working=truncation_ok,
            special_tokens_present=special_tokens_ok,
            token_type_ids_correct=token_type_ids_ok,
            attention_mask_valid=attention_mask_ok,
            issues_found=issues,
            sample_inputs=sample_inputs,
            fix_recommendations=fixes
        )
        
        # Log summary
        self._log_debugging_summary(result)
        
        return result
    
    def _generate_debug_test_pairs(self) -> List[Tuple[str, str]]:
        """Generate test pairs for input debugging."""
        return [
            # Normal length pairs
            ("machine learning algorithms", "supervised learning techniques for classification"),
            
            # Very short pairs
            ("a", "b"),
            
            # Long pairs (for truncation testing)
            ("very long query with many words that should definitely exceed the maximum token limit when combined with document", 
             "very long document text with extensive content that will test the tokenizer truncation behavior and ensure proper handling of sequence length limits in cross-encoder models"),
            
            # Empty/edge cases
            ("", "empty query test"),
            ("empty document test", ""),
            
            # Special characters
            ("query with [special] tokens", "document with <special> formatting"),
        ]
    
    def _extract_tokenizer(self, cross_encoder: Any) -> Optional[Any]:
        """Extract tokenizer from cross-encoder model."""
        if hasattr(cross_encoder, 'tokenizer'):
            return cross_encoder.tokenizer
        elif hasattr(cross_encoder, 'model') and hasattr(cross_encoder.model, 'tokenizer'):
            return cross_encoder.model.tokenizer
        else:
            self.logger.warning("Could not extract tokenizer from cross-encoder")
            return None
    
    def _detect_model_architecture(self, tokenizer: Any, model: Any) -> str:
        """Detect model architecture from tokenizer and model."""
        try:
            # Try to get model name
            model_name = getattr(tokenizer, 'name_or_path', '').lower()
            
            if 'roberta' in model_name:
                return 'roberta'
            elif 'deberta-v3' in model_name:
                return 'deberta-v3'
            elif 'deberta' in model_name:
                return 'deberta'
            elif 'bert' in model_name:
                return 'bert'
            else:
                # Try to infer from tokenizer properties
                if hasattr(tokenizer, 'cls_token'):
                    cls_token = str(tokenizer.cls_token) if tokenizer.cls_token else None
                    if cls_token == '<s>':
                        return 'roberta'
                    elif cls_token == '[CLS]':
                        return 'bert'
                
                self.logger.warning(f"Could not detect architecture from: {model_name}")
                return 'unknown'
                
        except Exception as e:
            self.logger.warning(f"Architecture detection failed: {e}")
            return 'unknown'
    
    def _tokenize_and_analyze(self, 
                            tokenizer: Any,
                            query: str, 
                            doc: str,
                            max_seq_len: int,
                            truncation: str,
                            model_arch: str) -> Dict[str, Any]:
        """Tokenize pair and analyze the result."""
        try:
            # Tokenize the pair
            inputs = tokenizer(
                query, doc,
                truncation=True,
                padding=True,
                max_length=max_seq_len,
                return_tensors="pt",
                return_attention_mask=True,
                truncation_strategy=truncation
            )
            
            # Extract token information
            input_ids = inputs['input_ids'][0].tolist()  # Remove batch dimension
            attention_mask = inputs['attention_mask'][0].tolist()
            
            # Decode tokens for analysis
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Find special token positions
            sep_positions = []
            cls_position = None
            
            for i, token in enumerate(tokens):
                if token in ['[CLS]', '<s>']:
                    cls_position = i
                elif token in ['[SEP]', '</s>']:
                    sep_positions.append(i)
            
            # Calculate sequence statistics
            total_length = len(input_ids)
            attention_sum = sum(attention_mask)
            padding_tokens = total_length - attention_sum
            
            # Split into query and document parts (approximate)
            if len(sep_positions) >= 1:
                if cls_position is not None:
                    query_length = sep_positions[0] - cls_position - 1
                else:
                    query_length = sep_positions[0]
                
                if len(sep_positions) >= 2:
                    doc_length = sep_positions[1] - sep_positions[0] - 1
                else:
                    doc_length = total_length - sep_positions[0] - 1 - padding_tokens
            else:
                query_length = 0
                doc_length = 0
            
            result = {
                'query_text': query,
                'doc_text': doc,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'tokens': tokens,
                'total_length': total_length,
                'query_length': query_length,
                'doc_length': doc_length,
                'padding_tokens': padding_tokens,
                'attention_sum': attention_sum,
                'cls_position': cls_position,
                'sep_positions': sep_positions,
                'first_30_tokens': tokens[:30],
                'last_30_tokens': tokens[-30:] if len(tokens) > 30 else tokens,
                'model_arch': model_arch
            }
            
            # Add token type IDs if present
            if 'token_type_ids' in inputs:
                result['token_type_ids'] = inputs['token_type_ids'][0].tolist()
            else:
                result['token_type_ids'] = None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tokenization analysis failed: {e}")
            return {
                'query_text': query,
                'doc_text': doc,
                'error': str(e),
                'tokens': [],
                'total_length': 0
            }
    
    def _log_tokenization_details(self, tokenized: Dict[str, Any], pair_index: int):
        """Log detailed tokenization information."""
        self.logger.info(f"📝 TOKENIZATION DETAILS - Pair {pair_index}:")
        self.logger.info(f"  Query: '{tokenized['query_text'][:50]}...'")
        self.logger.info(f"  Document: '{tokenized['doc_text'][:50]}...'")
        self.logger.info(f"  Total length: {tokenized['total_length']}")
        self.logger.info(f"  Query tokens: {tokenized['query_length']}")
        self.logger.info(f"  Document tokens: {tokenized['doc_length']}")
        self.logger.info(f"  Padding tokens: {tokenized['padding_tokens']}")
        self.logger.info(f"  Attention sum: {tokenized['attention_sum']}")
        
        if tokenized.get('cls_position') is not None:
            self.logger.info(f"  CLS position: {tokenized['cls_position']}")
        
        if tokenized.get('sep_positions'):
            self.logger.info(f"  SEP positions: {tokenized['sep_positions']}")
        
        # Log first and last tokens
        self.logger.info(f"  First 30 tokens: {tokenized['first_30_tokens']}")
        self.logger.info(f"  Last 30 tokens: {tokenized['last_30_tokens']}")
        
        # Log token type IDs if present
        if tokenized.get('token_type_ids'):
            token_types = tokenized['token_type_ids']
            type_0_count = token_types.count(0)
            type_1_count = token_types.count(1)
            self.logger.info(f"  Token type IDs: {type_0_count} type-0, {type_1_count} type-1")
        else:
            self.logger.info("  Token type IDs: Not present")
        
        self.logger.info("-" * 40)
    
    def _validate_tokenization(self, tokenized: Dict[str, Any], model_arch: str) -> List[str]:
        """Validate tokenization for specific issues."""
        issues = []
        
        if 'error' in tokenized:
            issues.append(f"Tokenization failed: {tokenized['error']}")
            return issues
        
        tokens = tokenized.get('tokens', [])
        if not tokens:
            issues.append("No tokens generated")
            return issues
        
        # Architecture-specific validation
        if model_arch == 'bert':
            # BERT should have [CLS] and [SEP] tokens
            if '[CLS]' not in tokens:
                issues.append("Missing [CLS] token for BERT model")
            if '[SEP]' not in tokens:
                issues.append("Missing [SEP] token for BERT model")
            if tokenized.get('token_type_ids') is None:
                issues.append("Missing token_type_ids for BERT model")
                
        elif model_arch == 'roberta':
            # RoBERTa should have <s> and </s> tokens
            if '<s>' not in tokens:
                issues.append("Missing <s> token for RoBERTa model")
            if '</s>' not in tokens:
                issues.append("Missing </s> token for RoBERTa model")
            if tokenized.get('token_type_ids') is not None:
                issues.append("Unexpected token_type_ids for RoBERTa model")
        
        # Check sequence length
        total_length = tokenized.get('total_length', 0)
        if total_length == 0:
            issues.append("Zero total sequence length")
        elif total_length > self.config['expected_max_length']:
            issues.append(f"Sequence length ({total_length}) exceeds expected max ({self.config['expected_max_length']})")
        
        # Check attention mask ratio
        attention_sum = tokenized.get('attention_sum', 0)
        if total_length > 0:
            attention_ratio = attention_sum / total_length
            if attention_ratio < self.config['min_attention_ratio']:
                issues.append(f"Low attention ratio: {attention_ratio:.2f} (too much padding)")
        
        return issues
    
    def _validate_special_tokens(self, sample_inputs: List[Dict[str, Any]], model_arch: str) -> bool:
        """Validate special token positioning across samples."""
        try:
            issues = 0
            
            for sample in sample_inputs:
                if 'error' in sample:
                    continue
                    
                tokens = sample.get('tokens', [])
                if not tokens:
                    issues += 1
                    continue
                
                # Check for expected special tokens
                if model_arch in ['bert', 'deberta']:
                    if '[CLS]' not in tokens or '[SEP]' not in tokens:
                        issues += 1
                elif model_arch in ['roberta', 'deberta-v3']:
                    if '<s>' not in tokens or '</s>' not in tokens:
                        issues += 1
                
                # Check special token positioning
                cls_pos = sample.get('cls_position')
                sep_positions = sample.get('sep_positions', [])
                
                if model_arch in ['bert', 'deberta']:
                    # [CLS] should be at position 0
                    if cls_pos != 0:
                        issues += 1
                    # Should have at least one [SEP]
                    if not sep_positions:
                        issues += 1
                elif model_arch in ['roberta', 'deberta-v3']:
                    # <s> should be at position 0
                    if cls_pos != 0:
                        issues += 1
                    # Should have </s> tokens
                    if not sep_positions:
                        issues += 1
            
            # Consider valid if less than half have issues
            return issues < len(sample_inputs) / 2
            
        except Exception as e:
            self.logger.warning(f"Special token validation failed: {e}")
            return False
    
    def _validate_token_type_ids(self, sample_inputs: List[Dict[str, Any]], model_arch: str) -> bool:
        """Validate token type IDs are correct for model architecture."""
        try:
            for sample in sample_inputs:
                if 'error' in sample:
                    continue
                
                token_type_ids = sample.get('token_type_ids')
                
                if model_arch in ['bert', 'deberta']:
                    # Should have token type IDs
                    if token_type_ids is None:
                        return False
                    
                    # Should have both 0s and 1s (query and document)
                    if 0 not in token_type_ids or 1 not in token_type_ids:
                        return False
                        
                elif model_arch in ['roberta', 'deberta-v3']:
                    # Should NOT have token type IDs
                    if token_type_ids is not None:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Token type ID validation failed: {e}")
            return False
    
    def _validate_attention_masks(self, sample_inputs: List[Dict[str, Any]]) -> bool:
        """Validate attention masks are reasonable."""
        try:
            for sample in sample_inputs:
                if 'error' in sample:
                    continue
                
                attention_mask = sample.get('attention_mask', [])
                if not attention_mask:
                    return False
                
                # Should have some non-padding tokens
                attention_sum = sum(attention_mask)
                if attention_sum == 0:
                    return False
                
                # Should not be all padding
                if attention_sum < len(attention_mask) * 0.1:  # Less than 10% real tokens
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Attention mask validation failed: {e}")
            return False
    
    def _test_truncation_behavior(self, tokenizer: Any, max_seq_len: int, truncation: str) -> bool:
        """Test truncation behavior with long sequences."""
        try:
            # Create long sequences
            long_query = " ".join(["query"] * 200)  # Very long query
            long_doc = " ".join(["document"] * 500)  # Very long document
            
            # Tokenize with truncation
            inputs = tokenizer(
                long_query, long_doc,
                truncation=True,
                max_length=max_seq_len,
                truncation_strategy=truncation
            )
            
            # Check that result fits within max length
            input_ids = inputs['input_ids']
            if isinstance(input_ids, list):
                actual_length = len(input_ids)
            else:  # tensor
                actual_length = input_ids.shape[-1]
            
            if actual_length > max_seq_len:
                self.logger.warning(f"Truncation failed: length {actual_length} > max {max_seq_len}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Truncation test failed: {e}")
            return False
    
    def _generate_input_debugging_fixes(self, 
                                      issues: List[str],
                                      model_arch: str,
                                      sample_inputs: List[Dict[str, Any]]) -> List[str]:
        """Generate fix recommendations for input issues."""
        fixes = []
        
        # Architecture-specific fixes
        for issue in issues:
            if "Missing [CLS]" in issue or "Missing [SEP]" in issue:
                fixes.append("Use correct tokenizer for BERT model - should add [CLS] and [SEP] tokens")
            elif "Missing <s>" in issue or "Missing </s>" in issue:
                fixes.append("Use correct tokenizer for RoBERTa model - should add <s> and </s> tokens")
            elif "token_type_ids" in issue:
                if "Missing" in issue:
                    fixes.append("BERT/DeBERTa models require token_type_ids - ensure tokenizer returns them")
                else:
                    fixes.append("RoBERTa/DeBERTa-v3 models should not use token_type_ids")
            elif "Sequence length" in issue and "exceeds" in issue:
                fixes.append("Reduce max_seq_len or improve truncation strategy")
            elif "Low attention ratio" in issue:
                fixes.append("Too much padding - check input lengths and batch padding strategy")
            elif "Tokenization failed" in issue:
                fixes.append("Fix tokenizer configuration or input text preprocessing")
        
        # General input format fixes
        if any("special token" in issue.lower() for issue in issues):
            fixes.append(f"Verify tokenizer matches model architecture: {model_arch}")
            fixes.append("Check tokenizer.add_special_tokens() configuration")
        
        if any("truncation" in issue.lower() for issue in issues):
            fixes.extend([
                "Try different truncation strategies: 'longest_first', 'only_first', 'only_second'",
                "Validate max_length parameter matches model configuration"
            ])
        
        if not fixes:
            fixes.append("Input formatting appears correct - check model inference and scoring")
        
        return fixes
    
    def _log_debugging_summary(self, result: InputDebuggingResult):
        """Log summary of input debugging results."""
        self.logger.info("🔍 INPUT DEBUGGING SUMMARY:")
        self.logger.info("-" * 40)
        
        status_emoji = "✅" if result.formatting_correct else "❌"
        self.logger.info(f"  Overall formatting: {status_emoji} {'CORRECT' if result.formatting_correct else 'ISSUES FOUND'}")
        
        # Individual checks
        checks = [
            ("Truncation working", result.truncation_working),
            ("Special tokens present", result.special_tokens_present),
            ("Token type IDs correct", result.token_type_ids_correct),
            ("Attention masks valid", result.attention_mask_valid)
        ]
        
        for check_name, status in checks:
            emoji = "✅" if status else "❌"
            self.logger.info(f"  {check_name}: {emoji}")
        
        # Issues
        if result.issues_found:
            self.logger.warning(f"Issues found ({len(result.issues_found)}):")
            for issue in result.issues_found:
                self.logger.warning(f"  • {issue}")
        
        self.logger.info("=" * 50)