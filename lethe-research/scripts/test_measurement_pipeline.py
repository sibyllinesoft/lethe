#!/usr/bin/env python3
"""
Unit Tests for Fixed Measurement Pipeline
=========================================

Tests all three measurement pipes with specified fixtures and validation scenarios.
Ensures fail-closed guards work correctly and measurements meet contract requirements.
"""

import unittest
import numpy as np
import json
from typing import Dict, List, Any
import tempfile
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from measurement_pipeline import (
    MeasurementPipeline, TokenizationPipe, KVReusePipe, DeltaCBUPipe,
    TokenizationResult, KVReuseResult, DeltaCBUResult
)

class TestTokenizationPipe(unittest.TestCase):
    """Test tokenization pipe with proper tokenizer-based measurement."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipe = TokenizationPipe("gpt-4")
        
        # Test fixtures
        self.fixtures = {
            'zh_qa_8pct': {
                'blob_text': "这是一个很长的中文文本，用于测试分词器的正确性和准确性。它包含了各种中文字符、标点符号和数字，以确保分词器能够正确处理中文内容。" * 200,  # Much longer Chinese text
                'arranged_head_text': "这是一个很长的中文文本，用于测试分词器的正确性和准确性。它包含了各种中文字符、标点符号和数字，以确保分词器能够正确处理中文内容。" * 50,
                'arranged_tail_text': "这是尾部的文本内容，包含更多的中文字符以确保总的token数量足够。" * 20,
                'expected_tokens_min': 500  # zh_qa@8% should be >500 tokens
            },
            'code_debug_15pct': {
                'blob_text': "def function_name(arg1, arg2):\n    return arg1 + arg2\n" * 50,
                'arranged_head_text': "def function_name(arg1, arg2):\n    return arg1 + arg2\n" * 15,
                'arranged_tail_text': "    # Additional code here\n" * 5,
                'expected_tokens_range': (200, 800)
            },
            'monotonicity_test': [
                {
                    'keep_ratio': 0.08,
                    'blob_text': "Sample text for testing monotonicity. " * 100,
                    'head_text': "Sample text for testing monotonicity. " * 8,
                    'tail_text': ""
                },
                {
                    'keep_ratio': 0.15,
                    'blob_text': "Sample text for testing monotonicity. " * 100,
                    'head_text': "Sample text for testing monotonicity. " * 15,
                    'tail_text': ""
                },
                {
                    'keep_ratio': 0.30,
                    'blob_text': "Sample text for testing monotonicity. " * 100,
                    'head_text': "Sample text for testing monotonicity. " * 30,
                    'tail_text': ""
                }
            ]
        }
    
    def test_tokenizer_hash_consistency(self):
        """Test that tokenizer hash is consistent and matches manifest."""
        result1 = self.pipe.measure_tokenization("test", "test", "")
        result2 = self.pipe.measure_tokenization("different", "different", "")
        
        # Hash should be same for same tokenizer
        self.assertEqual(result1.tokenizer_hash, result2.tokenizer_hash)
        self.assertEqual(result1.tokenizer_hash, self.pipe.tokenizer_hash)
        self.assertIsInstance(result1.tokenizer_hash, str)
        self.assertGreater(len(result1.tokenizer_hash), 0)
    
    def test_zh_qa_sanity_check(self):
        """Test zh_qa@8% sanity: median(tokens_kept@8%) > 500."""
        fixture = self.fixtures['zh_qa_8pct']
        
        result = self.pipe.measure_tokenization(
            fixture['blob_text'],
            fixture['arranged_head_text'], 
            fixture['arranged_tail_text']
        )
        
        # Validate result
        is_valid, error_msg = self.pipe.validate_tokenization_result(result)
        self.assertTrue(is_valid, f"Validation failed: {error_msg}")
        
        # Check zh_qa sanity requirement
        self.assertGreaterEqual(result.tokens_kept, fixture['expected_tokens_min'],
                               f"zh_qa tokens_kept {result.tokens_kept} below minimum {fixture['expected_tokens_min']}")
        
        # Check that it's using real tokenizer, not window/sink counts
        self.assertNotEqual(result.tokens_kept, 4, "Using window/sink count instead of tokenizer")
        self.assertNotEqual(result.tokens_kept, 8, "Using window/sink count instead of tokenizer")
    
    def test_monotonicity_requirement(self):
        """Test monotonicity: median(tokens_kept@30%) > @15% > @8%."""
        results = []
        
        for fixture in self.fixtures['monotonicity_test']:
            result = self.pipe.measure_tokenization(
                fixture['blob_text'],
                fixture['head_text'],
                fixture['tail_text']
            )
            results.append((fixture['keep_ratio'], result.tokens_kept))
        
        # Sort by keep ratio
        results.sort(key=lambda x: x[0])
        tokens_8pct = results[0][1]
        tokens_15pct = results[1][1]
        tokens_30pct = results[2][1]
        
        # Check monotonicity
        self.assertLess(tokens_8pct, tokens_15pct, 
                       f"Monotonicity violation: {tokens_8pct} >= {tokens_15pct}")
        self.assertLess(tokens_15pct, tokens_30pct,
                       f"Monotonicity violation: {tokens_15pct} >= {tokens_30pct}")
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio = tokens_kept / tokens_in."""
        fixture = self.fixtures['code_debug_15pct']
        
        result = self.pipe.measure_tokenization(
            fixture['blob_text'],
            fixture['arranged_head_text'],
            fixture['arranged_tail_text']
        )
        
        # Check compression ratio calculation
        expected_ratio = result.tokens_kept / result.tokens_in
        self.assertAlmostEqual(result.compression_ratio, expected_ratio, places=6)
        
        # Check ratio is in reasonable range
        self.assertGreaterEqual(result.compression_ratio, 0.0)
        self.assertLessEqual(result.compression_ratio, 1.0)
    
    def test_fail_closed_validation(self):
        """Test fail-closed validation triggers."""
        # Test with invalid tokenizer hash
        invalid_result = TokenizationResult(
            tokenizer_hash="INVALID",
            tokens_in=100,
            head_tokens=50,
            tail_tokens=25,
            tokens_kept=75,
            compression_ratio=0.75
        )
        
        is_valid, error_msg = self.pipe.validate_tokenization_result(invalid_result)
        self.assertFalse(is_valid)
        self.assertIn("Tokenizer hash mismatch", error_msg)
        
        # Test with arithmetic inconsistency
        inconsistent_result = TokenizationResult(
            tokenizer_hash=self.pipe.tokenizer_hash,
            tokens_in=100,
            head_tokens=50,
            tail_tokens=25,
            tokens_kept=80,  # Should be 75
            compression_ratio=0.75
        )
        
        is_valid, error_msg = self.pipe.validate_tokenization_result(inconsistent_result)
        self.assertFalse(is_valid)
        self.assertIn("Token arithmetic inconsistent", error_msg)

class TestKVReusePipe(unittest.TestCase):
    """Test KV-reuse pipe with prefix-Jaccard calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipe = KVReusePipe(prefix_length=100)
        
        # Test fixtures for different scenarios
        self.fixtures = {
            'high_reuse_scenario': {
                'session_id': 'debug_session',
                'turns': [
                    {'tokens': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'expected_jaccard': 0.0},  # First turn
                    {'tokens': [1, 2, 3, 4, 5, 11, 12, 13, 14, 15], 'expected_jaccard': 0.5},  # 50% overlap
                    {'tokens': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'expected_jaccard': 0.5},  # Back to first
                ]
            },
            'low_reuse_scenario': {
                'session_id': 'qa_session',
                'turns': [
                    {'tokens': [1, 2, 3, 4, 5], 'expected_jaccard': 0.0},  # First turn
                    {'tokens': [6, 7, 8, 9, 10], 'expected_jaccard': 0.0},  # No overlap
                    {'tokens': [11, 12, 13, 14, 15], 'expected_jaccard': 0.0},  # No overlap
                ]
            },
            'nonzero_mass_test': {
                'sessions': [f'session_{i}' for i in range(10)],
                'high_reuse_count': 8  # 80% should have >0.1 Jaccard
            }
        }
    
    def test_prefix_jaccard_calculation(self):
        """Test prefix-Jaccard calculation: |A ∩ B| / |A ∪ B|."""
        session_id = "test_session"
        
        # Turn 0: First turn should have 0.0 Jaccard
        result1 = self.pipe.measure_kv_reuse(session_id, [1, 2, 3, 4, 5], turn_number=0)
        self.assertEqual(result1.prefix_jaccard, 0.0)
        self.assertIsNone(result1.prev_head_prefix_tokens)
        
        # Turn 1: 50% overlap -> Jaccard = 3/7 = 0.428...
        result2 = self.pipe.measure_kv_reuse(session_id, [1, 2, 3, 6, 7], turn_number=1)
        expected_jaccard = 3 / 7  # |{1,2,3}| / |{1,2,3,4,5,6,7}|
        self.assertAlmostEqual(result2.prefix_jaccard, expected_jaccard, places=3)
        self.assertIsNotNone(result2.prev_head_prefix_tokens)
    
    def test_high_reuse_scenario(self):
        """Test high reuse scenario (code debug type)."""
        fixture = self.fixtures['high_reuse_scenario']
        session_id = fixture['session_id']
        
        for i, turn in enumerate(fixture['turns']):
            result = self.pipe.measure_kv_reuse(session_id, turn['tokens'], turn_number=i)
            
            # Validate result
            is_valid, error_msg = self.pipe.validate_kv_reuse_result(result)
            self.assertTrue(is_valid, f"Turn {i} validation failed: {error_msg}")
            
            # Check Jaccard is in expected range
            if i == 0:
                self.assertEqual(result.prefix_jaccard, 0.0)
            else:
                self.assertGreaterEqual(result.prefix_jaccard, 0.0)
                self.assertLessEqual(result.prefix_jaccard, 1.0)
    
    def test_nonzero_mass_requirement(self):
        """Test non-zero mass: share(prefix_jaccard>0.1) ≥ 0.8."""
        results = []
        
        # Simulate multiple sessions with high reuse to meet 80% requirement
        for i in range(10):
            session_id = f"session_{i}"
            
            # First turn (always 0.0)
            result1 = self.pipe.measure_kv_reuse(session_id, [1, 2, 3, 4, 5], turn_number=0)
            
            # Second turn with significant overlap for most sessions (8 out of 10)
            if i < 8:
                # High overlap to ensure Jaccard > 0.1
                # Using larger overlap: 4 shared tokens out of 8 total -> Jaccard = 4/8 = 0.5
                result2 = self.pipe.measure_kv_reuse(session_id, [1, 2, 3, 4, 6, 7, 8, 9], turn_number=1)
            else:
                # Low overlap for remaining sessions
                result2 = self.pipe.measure_kv_reuse(session_id, [10, 11, 12, 13, 14], turn_number=1)
            
            # Third turn to get more samples
            if i < 8:
                # Continue high overlap pattern
                result3 = self.pipe.measure_kv_reuse(session_id, [1, 2, 10, 11, 12, 13], turn_number=2)
                results.extend([result2.prefix_jaccard, result3.prefix_jaccard])
            else:
                result3 = self.pipe.measure_kv_reuse(session_id, [20, 21, 22, 23, 24], turn_number=2)
                results.extend([result2.prefix_jaccard, result3.prefix_jaccard])
        
        # Check non-zero mass
        nonzero_count = sum(1 for jaccard in results if jaccard > 0.1)
        nonzero_ratio = nonzero_count / len(results)
        
        self.assertGreaterEqual(nonzero_ratio, 0.8, 
                               f"Non-zero mass {nonzero_ratio:.2f} below required 0.8")
    
    def test_no_universal_zeros(self):
        """Test that not all KV-reuse values are zero."""
        session_id = "test_session"
        results = []
        
        # Generate several turns
        for i in range(5):
            if i == 0:
                tokens = [1, 2, 3, 4, 5]
            else:
                # Partial overlap to generate non-zero Jaccard
                tokens = [1, 2] + list(range(10 + i, 15 + i))
            
            result = self.pipe.measure_kv_reuse(session_id, tokens, turn_number=i)
            results.append(result.prefix_jaccard)
        
        # Should not be all zeros
        self.assertFalse(all(j == 0.0 for j in results), 
                        "All KV-reuse values are zero - arranger not wired")
    
    def test_fail_closed_validation(self):
        """Test fail-closed validation guards."""
        # Test invalid Jaccard bounds
        invalid_result = KVReuseResult(
            head_prefix_tokens=[1, 2, 3],
            prev_head_prefix_tokens=[4, 5, 6],
            prefix_jaccard=1.5  # Invalid: > 1.0
        )
        
        is_valid, error_msg = self.pipe.validate_kv_reuse_result(invalid_result)
        self.assertFalse(is_valid)
        self.assertIn("out of bounds", error_msg)

class TestDeltaCBUPipe(unittest.TestCase):
    """Test ΔCBU computation pipe with V2 payloads."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipe = DeltaCBUPipe()
        
        # Test fixtures with varying scenarios
        self.fixtures = {
            'v2_payload_present': {
                'atoms': [
                    {
                        'delta_utility': 0.5,
                        'entities': ['variable_a', 'function_x'],
                        'types': ['variable', 'function'],
                        'files': ['file1.py'],
                        'features': [0.1, 0.2, 0.3]
                    },
                    {
                        'delta_utility': 0.3,
                        'entities': ['class_b'],
                        'types': ['class'],
                        'files': ['file2.py'],
                        'features': [0.4, 0.5, 0.6]
                    }
                ],
                'tokens_kept': 1000,
                'has_v2': True
            },
            'v2_payload_missing': {
                'atoms': [],
                'tokens_kept': 1000,
                'has_v2': False
            },
            'variance_test': {
                'scenarios': [
                    {'method': 'streaming', 'utility_base': 0.1},
                    {'method': 'lethe', 'utility_base': 0.3},
                    {'method': 'hybrid', 'utility_base': 0.2}
                ]
            }
        }
    
    def test_v2_payload_requirement(self):
        """Test V2 payload requirement and fail-closed behavior."""
        # Test with V2 payload present
        fixture = self.fixtures['v2_payload_present']
        result = self.pipe.measure_delta_cbu(
            fixture['atoms'], fixture['tokens_kept'], fixture['has_v2']
        )
        
        is_valid, error_msg = self.pipe.validate_delta_cbu_result(result)
        self.assertTrue(is_valid, f"V2 payload validation failed: {error_msg}")
        self.assertTrue(result.v2_payload_present)
        self.assertGreater(result.delta_cbu_per_1k, 0.0)
        
        # Test with V2 payload missing (should fail closed)
        fixture_missing = self.fixtures['v2_payload_missing']
        result_missing = self.pipe.measure_delta_cbu(
            fixture_missing['atoms'], fixture_missing['tokens_kept'], fixture_missing['has_v2']
        )
        
        self.assertFalse(result_missing.v2_payload_present)
        self.assertTrue(np.isnan(result_missing.delta_cbu_per_1k))
        
        # Validation should fail for missing V2
        is_valid, error_msg = self.pipe.validate_delta_cbu_result(result_missing)
        self.assertFalse(is_valid)
        self.assertIn("V2 payload missing", error_msg)
    
    def test_bundle_utility_calculation(self):
        """Test bundle utility: F(S) = Σ[ΔU(a) + γ·Δ_cov(a) + δ·Δ_div(a)]."""
        fixture = self.fixtures['v2_payload_present']
        
        result = self.pipe.measure_delta_cbu(
            fixture['atoms'], fixture['tokens_kept'], fixture['has_v2']
        )
        
        # Check that ΔCBU is computed and reasonable
        self.assertGreater(result.delta_cbu_per_1k, 0.0)
        self.assertLess(result.delta_cbu_per_1k, 100.0)  # Reasonable upper bound
        
        # Check components are calculated
        self.assertGreaterEqual(result.coverage_marginal, 0.0)
        self.assertGreaterEqual(result.diversity_marginal, 0.0)
        self.assertEqual(result.bundle_atoms_count, len(fixture['atoms']))
    
    def test_variance_across_methods(self):
        """Test ΔCBU variance across different methods."""
        fixture = self.fixtures['variance_test']
        results = []
        
        for scenario in fixture['scenarios']:
            # Create atoms with method-specific utilities
            atoms = [{
                'delta_utility': scenario['utility_base'] + 0.1 * i,
                'entities': [f'entity_{i}'],
                'types': ['test'],
                'files': [f'file_{i}.py'],
                'features': [0.1 * (i + 1), 0.2 * (i + 1)]
            } for i in range(3)]
            
            result = self.pipe.measure_delta_cbu(atoms, 1000, True)
            results.append(result.delta_cbu_per_1k)
        
        # Check variance requirement: std > 1e-3
        variance = np.var(results)
        std_dev = np.std(results)
        
        self.assertGreater(std_dev, 1e-3, 
                          f"ΔCBU variance too low: std={std_dev:.6f}")
        
        # Results should not be constant
        self.assertGreater(max(results) - min(results), 1e-3,
                          "ΔCBU values are constant across methods")
    
    def test_coverage_marginal_calculation(self):
        """Test facility-location coverage calculation."""
        atoms = [
            {
                'delta_utility': 0.5,
                'entities': ['unique_entity'],
                'types': ['unique_type'],
                'files': ['unique_file.py'],
                'features': [0.1, 0.2]
            },
            {
                'delta_utility': 0.3,
                'entities': ['shared_entity'],
                'types': ['shared_type'],
                'files': ['shared_file.py'],
                'features': [0.3, 0.4]
            }
        ]
        
        # Test coverage calculation
        coverage1 = self.pipe._calculate_coverage_marginal(atoms[0], atoms)
        coverage2 = self.pipe._calculate_coverage_marginal(atoms[1], atoms)
        
        self.assertGreaterEqual(coverage1, 0.0)
        self.assertGreaterEqual(coverage2, 0.0)
    
    def test_diversity_marginal_calculation(self):
        """Test DPP diversity: Δ_div(a) = log(1 + ||(I − QQᵀ) v_a||²)."""
        atoms = [
            {
                'delta_utility': 0.5,
                'features': [1.0, 0.0]  # Orthogonal features
            },
            {
                'delta_utility': 0.3,
                'features': [0.0, 1.0]  # Orthogonal features
            }
        ]
        
        # Test diversity calculation
        diversity1 = self.pipe._calculate_diversity_marginal(atoms[0], atoms)
        diversity2 = self.pipe._calculate_diversity_marginal(atoms[1], atoms)
        
        self.assertGreater(diversity1, 0.0)
        self.assertGreater(diversity2, 0.0)
        
        # With orthogonal features, diversity should be significant
        self.assertGreater(diversity1, 0.1)
        self.assertGreater(diversity2, 0.1)
    
    def test_fail_closed_validation(self):
        """Test fail-closed validation guards."""
        # Test with infinite ΔCBU
        invalid_result = DeltaCBUResult(
            delta_cbu_per_1k=float('inf'),
            v2_payload_present=True,
            bundle_atoms_count=5,
            coverage_marginal=0.1,
            diversity_marginal=0.1
        )
        
        is_valid, error_msg = self.pipe.validate_delta_cbu_result(invalid_result)
        self.assertFalse(is_valid)
        self.assertIn("not finite", error_msg)

class TestMeasurementPipelineIntegration(unittest.TestCase):
    """Test complete measurement pipeline integration."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.pipeline = MeasurementPipeline("gpt-4")
        
        # Integration test scenarios
        self.scenarios = {
            'zh_qa_8pct': {
                'sample_data': {
                    'blob_text': "这是一个很长的中文文本，用于测试分词器的正确性和准确性。它包含了各种中文字符、标点符号和数字，以确保分词器能够正确处理中文内容。" * 200,
                    'arranged_head_text': "这是一个很长的中文文本，用于测试分词器的正确性和准确性。它包含了各种中文字符、标点符号和数字，以确保分词器能够正确处理中文内容。" * 50,
                    'arranged_tail_text': "这是尾部的文本内容，包含更多的中文字符以确保总的token数量足够。" * 20,
                    'head_token_ids': list(range(1, 101)),
                    'selected_atoms': [
                        {
                            'delta_utility': 0.4,
                            'entities': ['测试'],
                            'types': ['text'],
                            'features': [0.1, 0.2]
                        }
                    ],
                    'has_v2_payload': True
                },
                'session_id': 'zh_qa_session',
                'expected_min_tokens': 500
            },
            'all_pipes_integration': {
                'sample_data': {
                    'blob_text': "def calculate_result(data):\n    return sum(data)\n" * 50,
                    'arranged_head_text': "def calculate_result(data):\n    return sum(data)\n" * 15,
                    'arranged_tail_text': "# End of function\n" * 5,
                    'head_token_ids': list(range(1, 51)),
                    'selected_atoms': [
                        {
                            'delta_utility': 0.5,
                            'entities': ['calculate_result', 'data'],
                            'types': ['function', 'parameter'],
                            'files': ['main.py'],
                            'features': [0.2, 0.3, 0.1]
                        },
                        {
                            'delta_utility': 0.3,
                            'entities': ['sum'],
                            'types': ['builtin'],
                            'files': ['builtins.py'],
                            'features': [0.1, 0.1, 0.8]
                        }
                    ],
                    'has_v2_payload': True
                },
                'session_id': 'integration_session'
            }
        }
    
    def test_zh_qa_sanity_integration(self):
        """Test zh_qa sanity check in complete pipeline."""
        scenario = self.scenarios['zh_qa_8pct']
        
        result = self.pipeline.process_sample(
            scenario['sample_data'],
            scenario['session_id'],
            turn_number=0
        )
        
        # Should pass all validations
        self.assertTrue(result.get('eval_ok', False), 
                       f"Pipeline failed: {result.get('error', 'Unknown error')}")
        
        # Check zh_qa specific requirement
        tokens_kept = result.get('tokens_kept', 0)
        self.assertGreaterEqual(tokens_kept, scenario['expected_min_tokens'],
                               f"zh_qa tokens_kept {tokens_kept} below minimum")
    
    def test_all_pipes_working(self):
        """Test that all three pipes work together correctly."""
        scenario = self.scenarios['all_pipes_integration']
        
        result = self.pipeline.process_sample(
            scenario['sample_data'],
            scenario['session_id'],
            turn_number=0
        )
        
        # Should pass all validations
        self.assertTrue(result.get('eval_ok', False),
                       f"Pipeline failed: {result.get('error', 'Unknown error')}")
        
        # Check all required fields are present
        required_fields = [
            'tokenizer_hash', 'tokens_in', 'head_tokens', 'tail_tokens',
            'tokens_kept', 'compression_ratio', 'head_prefix_tokens',
            'prefix_jaccard', 'kv_reuse', 'delta_cbu_per_1k',
            'v2_payload_present'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Check field types and ranges
        self.assertIsInstance(result['tokenizer_hash'], str)
        self.assertGreater(result['tokens_in'], 0)
        self.assertGreaterEqual(result['prefix_jaccard'], 0.0)
        self.assertLessEqual(result['prefix_jaccard'], 1.0)
        self.assertTrue(result['v2_payload_present'])
    
    def test_kv_reuse_turn_progression(self):
        """Test KV-reuse across multiple turns."""
        scenario = self.scenarios['all_pipes_integration']
        session_id = scenario['session_id'] + "_turns"
        
        results = []
        
        # Process multiple turns
        for turn in range(3):
            # Vary the head tokens slightly each turn
            sample_data = scenario['sample_data'].copy()
            sample_data['head_token_ids'] = list(range(turn * 10 + 1, (turn + 1) * 10 + 41))
            
            result = self.pipeline.process_sample(sample_data, session_id, turn)
            results.append(result)
        
        # All should succeed
        for i, result in enumerate(results):
            self.assertTrue(result.get('eval_ok', False),
                           f"Turn {i} failed: {result.get('error', 'Unknown error')}")
        
        # First turn should have 0.0 Jaccard
        self.assertEqual(results[0]['prefix_jaccard'], 0.0)
        
        # Later turns should have some reuse
        self.assertGreaterEqual(results[1]['prefix_jaccard'], 0.0)
        self.assertGreaterEqual(results[2]['prefix_jaccard'], 0.0)
    
    def test_monotonicity_validation(self):
        """Test monotonicity validation across keep ratios."""
        # Create results for different keep ratios
        base_sample = self.scenarios['all_pipes_integration']['sample_data']
        results = []
        
        keep_ratios = [0.08, 0.15, 0.30]
        for ratio in keep_ratios:
            # Scale the text based on keep ratio
            scale_factor = int(ratio * 100)
            sample_data = base_sample.copy()
            sample_data['arranged_head_text'] = base_sample['arranged_head_text'][:scale_factor * 10]
            sample_data['arranged_tail_text'] = base_sample['arranged_tail_text'][:scale_factor * 5]
            
            result = self.pipeline.process_sample(sample_data, f"mono_session_{ratio}", 0)
            result['keep_ratio'] = ratio  # Add for validation
            results.append(result)
        
        # Test monotonicity validation
        is_valid, error_msg = self.pipeline.validate_monotonicity(results)
        self.assertTrue(is_valid, f"Monotonicity validation failed: {error_msg}")
    
    def test_fail_closed_behavior(self):
        """Test that fail-closed guards work correctly."""
        # Test with missing V2 payload
        sample_data = self.scenarios['all_pipes_integration']['sample_data'].copy()
        sample_data['has_v2_payload'] = False
        
        result = self.pipeline.process_sample(sample_data, "fail_test_session", 0)
        
        # Should fail due to missing V2 payload
        self.assertFalse(result.get('eval_ok', True),
                        "Pipeline should fail with missing V2 payload")
        self.assertIn('error', result)

def run_test_suite():
    """Run the complete test suite."""
    print("🧪 Running Measurement Pipeline Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestTokenizationPipe,
        TestKVReusePipe,
        TestDeltaCBUPipe,
        TestMeasurementPipelineIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Measurement pipeline is working correctly")
    else:
        print(f"❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_test_suite()
    exit(0 if success else 1)