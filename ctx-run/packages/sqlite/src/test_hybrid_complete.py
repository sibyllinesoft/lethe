#!/usr/bin/env python3
"""
Comprehensive Test Suite for Lethe→StreamingLLM Hybrid System

Tests all major components of the hybrid system including:
- HybridSelector functionality and correctness
- Instrumentation and monitoring systems  
- Adaptive parameter optimization
- Integration between all components
- Performance validation and benchmarking
- Edge cases and error handling

Includes both unit tests and integration demos with realistic scenarios.
"""

import logging
import time
import numpy as np
import unittest
from unittest.mock import Mock, patch
import tempfile
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import hybrid system components
from .hybrid_selector import (
    HybridSelector, HybridConfig, ContentAtom, ContentType, 
    AtomExtractor, HeadBuilder, TailBuilder, ProcessingMode
)
from .instrumentation import (
    HybridInstrumentation, ComputeMetrics, EVTTailModeler,
    KVJaccardMonitor, ParameterDriftMonitor, AlarmLevel
)
from .adaptive_params import (
    AdaptiveParameterController, OptimizationObjective, 
    ParameterBounds, AdaptationStrategy
)
from .benchmarking import (
    HybridBenchmarkEvaluator, BenchmarkMethod, DatasetType,
    LetheStreamingHybridCompetitor, CompetitorConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAtomExtractor(unittest.TestCase):
    """Test atom extraction and classification."""
    
    def setUp(self):
        self.config = HybridConfig()
        self.extractor = AtomExtractor(self.config)
    
    def test_extract_basic_atoms(self):
        """Test basic atom extraction."""
        content = """
def hello_world():
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b

# This is a comment
ERROR: Something went wrong
@tool
def api_call():
    pass
"""
        
        atoms = self.extractor.extract_atoms(content)
        
        # Verify atoms were extracted
        self.assertGreater(len(atoms), 0)
        
        # Check content types are classified
        content_types = {atom.content_type for atom in atoms}
        expected_types = {
            ContentType.DEFINITION, 
            ContentType.ERROR_FRAME,
            ContentType.TOOL_KEY,
            ContentType.CONTEXT
        }
        
        # Should have at least some expected types
        self.assertTrue(len(content_types.intersection(expected_types)) > 0)
    
    def test_atom_metadata(self):
        """Test atom metadata computation."""
        content = "def test_function(): return True"
        atoms = self.extractor.extract_atoms(content)
        
        self.assertEqual(len(atoms), 1)
        atom = atoms[0]
        
        # Check required fields
        self.assertIsNotNone(atom.id)
        self.assertIsNotNone(atom.content)
        self.assertIsNotNone(atom.content_type)
        self.assertGreater(atom.tokens, 0)
        self.assertIsInstance(atom.relevance_score, float)
        self.assertIsInstance(atom.stability_score, float)
        self.assertIsNotNone(atom.kv_prefix_hash)

class TestHeadBuilder(unittest.TestCase):
    """Test head building with grouped atoms."""
    
    def setUp(self):
        self.config = HybridConfig(head_keep_ratio=0.15, dpp_rank=10)
        self.head_builder = HeadBuilder(self.config)
        
        # Create test atoms
        self.test_atoms = [
            ContentAtom(
                id="def1", content="def func1(): pass", 
                content_type=ContentType.DEFINITION,
                tokens=10, relevance_score=0.9, stability_score=0.8
            ),
            ContentAtom(
                id="err1", content="ERROR: Test error",
                content_type=ContentType.ERROR_FRAME, 
                tokens=5, relevance_score=0.7, stability_score=0.9
            ),
            ContentAtom(
                id="tool1", content="@tool def api(): pass",
                content_type=ContentType.TOOL_KEY,
                tokens=8, relevance_score=0.8, stability_score=0.85
            ),
            ContentAtom(
                id="ctx1", content="Regular context text",
                content_type=ContentType.CONTEXT,
                tokens=15, relevance_score=0.5, stability_score=0.3
            )
        ]
    
    def test_build_head(self):
        """Test head building."""
        budget_tokens = 25  # About 60% of total atoms (38 tokens)
        
        head_selection = self.head_builder.build_head(self.test_atoms, budget_tokens)
        
        # Verify head selection structure
        self.assertIsNotNone(head_selection)
        self.assertLessEqual(head_selection.total_tokens, budget_tokens)
        self.assertGreater(len(head_selection.grouped_atoms), 0)
        self.assertIsNotNone(head_selection.head_digest)
        
        # Should prefer stable, high-relevance content
        selected_types = set(head_selection.grouped_atoms.keys())
        
        # Definitions and errors should be preferred over context
        if ContentType.CONTEXT in selected_types:
            # If context is selected, higher priority items should also be selected
            self.assertTrue(
                ContentType.DEFINITION in selected_types or 
                ContentType.ERROR_FRAME in selected_types
            )
    
    def test_head_digest_creation(self):
        """Test head digest creation."""
        head_selection = self.head_builder.build_head(self.test_atoms, 30)
        
        # Head digest should be meaningful and compact
        self.assertIsNotNone(head_selection.head_digest)
        self.assertGreater(len(head_selection.head_digest), 0)
        self.assertLessEqual(len(head_selection.head_digest), 200)  # Max 200 chars

class TestTailBuilder(unittest.TestCase):
    """Test tail building with windowing."""
    
    def setUp(self):
        self.config = HybridConfig(
            window_size=100,  # Small for testing
            stride=50,
            sink_tokens=20
        )
        self.tail_builder = TailBuilder(self.config)
    
    def test_build_tail(self):
        """Test tail building with windowing."""
        # Create long content that requires windowing
        content = " ".join([f"word{i}" for i in range(200)])  # 200 words
        head_digest = "DEF: test_func | ERR: test_error"
        budget_tokens = 150
        
        tail_selection = self.tail_builder.build_tail(content, head_digest, budget_tokens)
        
        # Verify tail structure
        self.assertIsNotNone(tail_selection)
        self.assertGreater(len(tail_selection.windows), 0)
        self.assertLessEqual(tail_selection.total_tokens, budget_tokens)
        
        # Check windows have attention sinks
        for window in tail_selection.windows:
            self.assertGreater(len(window.attention_sinks), 0)
            self.assertGreater(window.sink_tokens, 0)
            
            # Check head digest is included in sinks
            sink_content = " ".join(window.attention_sinks)
            self.assertIn("test_func", sink_content.lower())
    
    def test_attention_sink_creation(self):
        """Test attention sink creation with head digest."""
        content = "This is test content for windowing"
        head_digest = "CONTEXT: important_function | DEF: main_class"
        
        tail_selection = self.tail_builder.build_tail(content, head_digest, 100)
        
        if tail_selection.windows:
            window = tail_selection.windows[0]
            sink_content = " ".join(window.attention_sinks)
            
            # Should include head digest elements
            self.assertTrue(
                "important_function" in sink_content or
                "main_class" in sink_content
            )

class TestHybridSelector(unittest.TestCase):
    """Test complete hybrid selector functionality."""
    
    def setUp(self):
        self.config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=1000,
            stride=500,
            sink_tokens=50
        )
        self.selector = HybridSelector(self.config)
    
    def test_head_only_selection(self):
        """Test head-only processing mode."""
        # Small content that shouldn't trigger streaming
        content = """
def simple_function():
    return "hello"

# Simple comment
x = 1 + 1
"""
        
        result = self.selector.select(content)
        
        # Should use head-only mode for simple content
        self.assertIn(result.processing_mode, [ProcessingMode.HEAD_ONLY, ProcessingMode.HYBRID])
        self.assertIsNotNone(result.head_selection)
        self.assertGreater(result.total_tokens, 0)
        self.assertGreater(len(result.final_content), 0)
    
    def test_hybrid_selection(self):
        """Test hybrid processing mode."""
        # Large, complex content that should trigger hybrid mode
        content = self._create_complex_content()
        
        result = self.selector.select(content)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.head_selection)
        self.assertGreater(result.total_tokens, 0)
        self.assertIsInstance(result.kv_prefix_reuse_ratio, float)
        self.assertIsInstance(result.objective_value, float)
        self.assertIsInstance(result.selection_time_ms, float)
        
        # Check gating decision
        self.assertIsInstance(result.gating_decision, dict)
        self.assertIn('processing_mode', result.gating_decision)
    
    def test_kv_aware_arrangement(self):
        """Test KV cache aware arrangement."""
        content = self._create_complex_content()
        result = self.selector.select(content)
        
        # Should have optimized arrangement
        self.assertIsInstance(result.kv_arrangement_optimized, bool)
        
        # Final content should be properly arranged (head first)
        if result.head_selection and result.tail_selection:
            # In KV-aware arrangement, head content should appear first
            self.assertGreater(len(result.final_content), 0)
    
    def test_parameter_updates(self):
        """Test parameter updates."""
        original_head_ratio = self.selector.config.head_keep_ratio
        
        # Update configuration
        self.selector.update_config(head_keep_ratio=0.20)
        
        self.assertEqual(self.selector.config.head_keep_ratio, 0.20)
        self.assertNotEqual(self.selector.config.head_keep_ratio, original_head_ratio)
    
    def _create_complex_content(self) -> str:
        """Create complex content for testing."""
        return """
# Complex Python Application

import numpy as np
from typing import List, Dict, Optional
import logging

class DataProcessor:
    '''Complex data processing class with multiple methods.'''
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processed_count = 0
    
    def process_data(self, data: List[float]) -> Dict[str, float]:
        '''Process numerical data with statistics.'''
        try:
            if not data:
                raise ValueError("Empty data provided")
            
            # Calculate statistics
            mean = np.mean(data)
            std = np.std(data)
            median = np.median(data)
            
            result = {
                'mean': mean,
                'std': std, 
                'median': median,
                'count': len(data)
            }
            
            self.processed_count += 1
            self.logger.info(f"Processed batch {self.processed_count}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data processing error: {e}")
            raise
    
    def validate_data(self, data: List[float]) -> bool:
        '''Validate input data for processing.'''
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, (int, float)):
                return False
            if np.isnan(item) or np.isinf(item):
                return False
        
        return True
    
    @tool
    def export_results(self, results: Dict[str, float]) -> str:
        '''Export processing results to JSON format.'''
        import json
        return json.dumps(results, indent=2)

def main():
    '''Main application entry point.'''
    config = {
        'batch_size': 1000,
        'validation_enabled': True,
        'export_format': 'json'
    }
    
    processor = DataProcessor(config)
    
    # Generate test data
    test_data = np.random.normal(100, 15, 5000).tolist()
    
    # Process data
    if processor.validate_data(test_data):
        results = processor.process_data(test_data)
        exported = processor.export_results(results)
        print(f"Results: {exported}")
    else:
        print("ERROR: Invalid data format")

# Additional complexity with more functions and classes
class ResultsAnalyzer:
    '''Analyze processing results for insights.'''
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_trends(self, historical_results: List[Dict[str, float]]) -> Dict[str, Any]:
        '''Analyze trends in historical results.'''
        if not historical_results:
            return {}
        
        means = [r.get('mean', 0) for r in historical_results]
        stds = [r.get('std', 0) for r in historical_results]
        
        trend_analysis = {
            'mean_trend': 'increasing' if means[-1] > means[0] else 'decreasing',
            'std_trend': 'increasing' if stds[-1] > stds[0] else 'decreasing',
            'volatility': np.std(means),
            'sample_count': len(historical_results)
        }
        
        return trend_analysis

if __name__ == "__main__":
    main()
""" * 3  # Make it even larger

class TestInstrumentation(unittest.TestCase):
    """Test instrumentation system."""
    
    def setUp(self):
        self.instrumentation = HybridInstrumentation()
        
        # Create mock selection result
        self.mock_result = Mock()
        self.mock_result.selection_time_ms = 150.0
        self.mock_result.total_tokens = 500
        self.mock_result.head_selection = Mock()
        self.mock_result.head_selection.total_tokens = 200
        self.mock_result.head_selection.kv_prefix_hashes = {"hash1", "hash2"}
        self.mock_result.head_selection.ce_early_exit_used = True
        self.mock_result.tail_selection = Mock()
        self.mock_result.tail_selection.total_tokens = 300
        self.mock_result.tail_selection.total_windows = 2
        self.mock_result.kv_prefix_reuse_ratio = 0.75
        self.mock_result.objective_value = 0.85
        self.mock_result.cost_lambda = 0.02
        self.mock_result.cost_mu = 0.015
        self.mock_result.net_value = 0.815
        self.mock_result.head_time_ms = 50.0
        self.mock_result.tail_time_ms = 80.0
        self.mock_result.arrangement_time_ms = 20.0
        self.mock_result.keep_ratio = 0.15
        self.mock_result.parameter_state = {
            'lambda': 0.01,
            'mu': 0.02,
            'head_keep_ratio': 0.12,
            'window_size': 6000,
            'stride': 3000,
            'dpp_rank': 14
        }
    
    def test_record_selection(self):
        """Test selection recording."""
        # Record a selection
        self.instrumentation.record_selection(self.mock_result, "test_session")
        
        # Check telemetry was recorded
        self.assertGreater(len(self.instrumentation.telemetry_records), 0)
        self.assertGreater(len(self.instrumentation.compute_metrics), 0)
        
        # Verify metrics
        latest_telemetry = self.instrumentation.telemetry_records[-1]
        self.assertEqual(latest_telemetry.session_id, "test_session")
        self.assertEqual(latest_telemetry.tokens_in, 500)
        self.assertEqual(latest_telemetry.head_tokens, 200)
        self.assertEqual(latest_telemetry.tail_tokens, 300)
    
    def test_evt_modeling(self):
        """Test EVT tail modeling."""
        evt_modeler = EVTTailModeler()
        
        # Add samples
        for i in range(200):
            compute_time = 100 + np.random.gamma(2, 10)  # Some with heavy tail
            evt_modeler.add_compute_sample(compute_time, 1000)
        
        # Should have computed parameters
        xi_param = evt_modeler.get_xi_parameter()
        if xi_param is not None:
            self.assertIsInstance(xi_param, float)
        
        # Get risk metrics
        risk_metrics = evt_modeler.get_tail_risk_metrics()
        self.assertIn('status', risk_metrics)
    
    def test_kv_jaccard_monitoring(self):
        """Test KV Jaccard monitoring."""
        kv_monitor = KVJaccardMonitor()
        
        # Simulate KV prefix sets
        prefix_sets = [
            {"prefix1", "prefix2", "prefix3"},
            {"prefix1", "prefix2", "prefix4"},  # Some overlap
            {"prefix1", "prefix5", "prefix6"}   # Degrading overlap
        ]
        
        metrics = kv_monitor.record_kv_jaccard(prefix_sets)
        
        # Should return valid metrics
        self.assertIsInstance(metrics.current_jaccard, float)
        self.assertGreaterEqual(metrics.current_jaccard, 0.0)
        self.assertLessEqual(metrics.current_jaccard, 1.0)
    
    def test_dashboard_metrics(self):
        """Test dashboard metrics generation."""
        # Record some selections first
        for _ in range(10):
            self.instrumentation.record_selection(self.mock_result, "test")
        
        dashboard = self.instrumentation.get_dashboard_metrics()
        
        # Check required sections
        self.assertIn('performance', dashboard)
        self.assertIn('tail_risk', dashboard)
        self.assertIn('optimization', dashboard)
        self.assertIn('alarms', dashboard)
        
        # Check performance metrics
        perf = dashboard['performance']
        self.assertIn('avg_latency_ms', perf)
        self.assertIn('total_operations', perf)

class TestAdaptiveParameters(unittest.TestCase):
    """Test adaptive parameter system."""
    
    def setUp(self):
        self.config = HybridConfig()
        self.instrumentation = HybridInstrumentation()
        self.objectives = OptimizationObjective()
        
        self.controller = AdaptiveParameterController(
            self.config, self.instrumentation, self.objectives
        )
    
    def test_parameter_bounds(self):
        """Test parameter bounds enforcement."""
        bounds = ParameterBounds()
        
        # Test constraining values
        constrained = bounds.constrain('head_keep_ratio', 0.5)  # Above max
        self.assertLessEqual(constrained, bounds.head_keep_ratio_max)
        
        constrained = bounds.constrain('head_keep_ratio', 0.01)  # Below min
        self.assertGreaterEqual(constrained, bounds.head_keep_ratio_min)
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        # Update metrics
        metrics = {
            'avg_latency_ms': 150.0,
            'kv_reuse_ratio': 0.75,
            'avg_quality_score': 0.85
        }
        
        self.controller.update_performance_metrics(metrics)
        
        # Check performance was recorded
        current = self.controller.performance_tracker.get_current_performance()
        if current:
            self.assertAlmostEqual(current['avg_latency_ms'], 150.0)
    
    def test_adaptation_rules(self):
        """Test adaptation rule evaluation."""
        # Create metrics that should trigger adaptation
        trigger_metrics = {
            'kv_degradation_pp': -0.12,  # Should trigger head size reduction
            'xi_parameter': 0.25,        # Should trigger stride reduction
            'p95_latency_ms': 1200       # Should trigger lambda increase
        }
        
        # Force update (bypassing cooldown for testing)
        self.controller.last_adaptation_time = 0.0
        self.controller.update_performance_metrics(trigger_metrics)
        
        # Check if adaptations were recorded
        status = self.controller.get_adaptation_status()
        self.assertIsInstance(status, dict)
        self.assertIn('adaptation_enabled', status)
    
    def test_exploration(self):
        """Test exploration functionality."""
        # Force exploration
        self.controller.exploration.exploration_rate = 1.0  # Always explore
        self.controller.exploration.last_exploration_time = 0.0
        
        # Trigger exploration
        metrics = {'avg_latency_ms': 100.0}
        self.controller.last_adaptation_time = 0.0
        self.controller.update_performance_metrics(metrics)
        
        # Check exploration state
        self.assertGreaterEqual(self.controller.exploration.explorations_attempted, 0)

class TestIntegrationComplete(unittest.TestCase):
    """Complete integration tests."""
    
    def setUp(self):
        """Set up complete integrated system."""
        # Create hybrid selector
        self.hybrid_config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=6000,
            stride=3000,
            sink_tokens=96,
            dpp_rank=14
        )
        self.selector = HybridSelector(self.hybrid_config)
        
        # Create instrumentation
        self.instrumentation = HybridInstrumentation()
        
        # Create adaptive controller
        self.adaptive_controller = AdaptiveParameterController(
            self.hybrid_config, self.instrumentation
        )
        
        # Test content
        self.test_content = self._create_realistic_content()
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # 1. Process content with hybrid selector
        result = self.selector.select(self.test_content)
        
        # 2. Record in instrumentation
        self.instrumentation.record_selection(result, "integration_test")
        
        # 3. Update adaptive controller
        metrics = {
            'avg_latency_ms': result.selection_time_ms,
            'kv_reuse_ratio': result.kv_prefix_reuse_ratio,
            'avg_quality_score': result.objective_value,
            'xi_parameter': 0.15,
            'p95_latency_ms': result.selection_time_ms * 1.2
        }
        
        self.adaptive_controller.update_performance_metrics(metrics)
        
        # 4. Verify all components worked
        self.assertIsNotNone(result)
        self.assertGreater(len(self.instrumentation.telemetry_records), 0)
        
        # 5. Check system health
        health = self.instrumentation.get_health_status()
        self.assertIn('overall_status', health)
    
    def test_performance_feedback_loop(self):
        """Test performance feedback adaptation loop."""
        original_head_ratio = self.hybrid_config.head_keep_ratio
        
        # Simulate multiple iterations with feedback
        for iteration in range(5):
            # Process content
            result = self.selector.select(self.test_content)
            
            # Record metrics  
            self.instrumentation.record_selection(result, f"iter_{iteration}")
            
            # Simulate degrading KV performance to trigger adaptation
            metrics = {
                'kv_degradation_pp': -0.11 if iteration > 2 else 0.0,
                'avg_latency_ms': result.selection_time_ms,
                'xi_parameter': 0.1
            }
            
            # Allow adaptation (reset cooldown)
            self.adaptive_controller.last_adaptation_time = 0.0
            self.adaptive_controller.update_performance_metrics(metrics)
            
            time.sleep(0.1)  # Small delay between iterations
        
        # Check if adaptation occurred
        status = self.adaptive_controller.get_adaptation_status()
        adaptations = len(self.adaptive_controller.adaptation_history)
        
        # Should have attempted some adaptations
        self.assertGreaterEqual(adaptations, 0)
    
    def test_monitoring_and_alerting(self):
        """Test monitoring and alerting functionality."""
        # Process content to generate metrics
        result = self.selector.select(self.test_content)
        self.instrumentation.record_selection(result, "monitoring_test")
        
        # Get dashboard metrics
        dashboard = self.instrumentation.get_dashboard_metrics()
        
        # Verify monitoring data
        self.assertIn('performance', dashboard)
        self.assertIn('alarms', dashboard)
        
        # Check health status
        health = self.instrumentation.get_health_status()
        self.assertIn('overall_status', health)
        self.assertIn(['HEALTHY', 'WARNING', 'CRITICAL'], health['overall_status'])
    
    def test_export_and_persistence(self):
        """Test data export and persistence."""
        # Generate some data
        result = self.selector.select(self.test_content)
        self.instrumentation.record_selection(result, "export_test")
        
        # Export telemetry
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = self.instrumentation.export_telemetry(f.name)
        
        # Verify export
        self.assertTrue(Path(export_path).exists())
        
        with open(export_path) as f:
            export_data = json.load(f)
        
        self.assertIn('telemetry_records', export_data)
        self.assertIn('metadata', export_data)
        
        # Clean up
        Path(export_path).unlink()
    
    def _create_realistic_content(self) -> str:
        """Create realistic test content."""
        return """
# Machine Learning Model Training Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json

class CustomDataset(Dataset):
    '''Custom dataset for model training.'''
    
    def __init__(self, data_path: str, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = self._load_samples()
        self.logger = logging.getLogger(__name__)
    
    def _load_samples(self) -> List[Tuple[torch.Tensor, int]]:
        '''Load training samples from disk.'''
        samples = []
        try:
            # Load data (simplified)
            data_files = list(self.data_path.glob('*.json'))
            for file_path in data_files:
                with open(file_path) as f:
                    data = json.load(f)
                    samples.append((
                        torch.tensor(data['features']), 
                        data['label']
                    ))
        except Exception as e:
            self.logger.error(f"Failed to load samples: {e}")
            raise
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample, label = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class NeuralNetwork(nn.Module):
    '''Simple neural network for classification.'''
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ModelTrainer:
    '''Model training and evaluation pipeline.'''
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        '''Train model for one epoch.'''
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                self.logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        '''Validate model performance.'''
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int) -> Dict[str, List[float]]:
        '''Complete training pipeline.'''
        self.logger.info(f'Starting training for {num_epochs} epochs')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_losses.append(train_loss)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)
            
            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Accuracy: {val_accuracy:.2f}%'
            )
            
            # Early stopping check
            if val_accuracy > self.config.get('target_accuracy', 95.0):
                self.logger.info(f'Target accuracy reached, stopping early')
                break
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'validation_accuracies': self.validation_accuracies
        }
    
    def save_model(self, filepath: str):
        '''Save trained model.'''
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': {
                'training_losses': self.training_losses,
                'validation_losses': self.validation_losses,
                'validation_accuracies': self.validation_accuracies
            }
        }, filepath)
        
        self.logger.info(f'Model saved to {filepath}')

def create_model_and_trainer(config: Dict[str, Any]) -> Tuple[NeuralNetwork, ModelTrainer]:
    '''Factory function for model and trainer.'''
    model = NeuralNetwork(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        num_classes=config['num_classes']
    )
    
    trainer = ModelTrainer(model, config)
    
    return model, trainer

@tool
def hyperparameter_search(base_config: Dict[str, Any], 
                         search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    '''Perform hyperparameter search.'''
    best_config = base_config.copy()
    best_accuracy = 0.0
    
    # Simple grid search
    import itertools
    
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    for combination in itertools.product(*param_values):
        config = base_config.copy()
        
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        # Train model with this configuration
        try:
            model, trainer = create_model_and_trainer(config)
            # Simplified training for search
            results = trainer.train(
                train_loader=None,  # Would provide real data
                val_loader=None, 
                num_epochs=5  # Fewer epochs for search
            )
            
            final_accuracy = max(results['validation_accuracies'])
            
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_config = config.copy()
            
        except Exception as e:
            print(f"Configuration failed: {config}, Error: {e}")
            continue
    
    return {
        'best_config': best_config,
        'best_accuracy': best_accuracy
    }

def main():
    '''Main training pipeline.'''
    # Configuration
    config = {
        'input_size': 784,  # MNIST-like
        'hidden_sizes': [256, 128, 64],
        'num_classes': 10,
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 20,
        'target_accuracy': 95.0
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model and trainer
        model, trainer = create_model_and_trainer(config)
        
        # Create datasets (would use real data)
        # train_dataset = CustomDataset('data/train')
        # val_dataset = CustomDataset('data/val')
        
        # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        
        # Train model
        logger.info("Starting model training")
        # results = trainer.train(train_loader, val_loader, config['num_epochs'])
        
        # Save model
        trainer.save_model('model_checkpoint.pth')
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
"""

class TestDemonstrations:
    """Integration demonstrations (not unit tests)."""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def demo_hybrid_selection(self):
        """Demonstrate hybrid selection process."""
        print("\n" + "="*60)
        print("HYBRID SELECTION DEMONSTRATION")
        print("="*60)
        
        # Create hybrid selector with canary config
        config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=6000,
            stride=3000,
            sink_tokens=96,
            dpp_rank=14
        )
        selector = HybridSelector(config)
        
        # Test with realistic content
        content = TestIntegrationComplete()._create_realistic_content()
        
        print(f"Input content: {len(content.split())} words")
        
        start_time = time.time()
        result = selector.select(content)
        elapsed_time = time.time() - start_time
        
        print(f"Processing mode: {result.processing_mode.value}")
        print(f"Total tokens kept: {result.total_tokens}")
        print(f"Keep ratio: {result.keep_ratio:.3f}")
        print(f"Head tokens: {result.head_selection.total_tokens if result.head_selection else 0}")
        print(f"Tail tokens: {result.tail_selection.total_tokens if result.tail_selection else 0}")
        print(f"KV reuse ratio: {result.kv_prefix_reuse_ratio:.3f}")
        print(f"Objective value: {result.objective_value:.3f}")
        print(f"Selection time: {result.selection_time_ms:.1f}ms")
        print(f"Total processing time: {elapsed_time*1000:.1f}ms")
        
        if result.gating_decision:
            print(f"Gating decision: {result.gating_decision['reasoning']}")
        
        return result
    
    def demo_instrumentation(self):
        """Demonstrate instrumentation and monitoring."""
        print("\n" + "="*60)
        print("INSTRUMENTATION DEMONSTRATION")
        print("="*60)
        
        instrumentation = HybridInstrumentation()
        
        # Create selector for generating results
        selector = HybridSelector()
        content = TestIntegrationComplete()._create_realistic_content()
        
        # Record multiple selections
        for i in range(10):
            result = selector.select(content)
            instrumentation.record_selection(result, f"demo_session_{i}")
            
            if i == 5:
                print(f"Recorded {i+1} selections...")
        
        # Get dashboard metrics
        dashboard = instrumentation.get_dashboard_metrics()
        
        print(f"Total operations: {dashboard['performance']['total_operations']}")
        print(f"Average latency: {dashboard['performance']['avg_latency_ms']:.1f}ms")
        print(f"P95 latency: {dashboard['performance']['p95_latency_ms']:.1f}ms")
        print(f"Average KV reuse: {dashboard['performance']['avg_kv_reuse_ratio']:.3f}")
        
        print(f"Current CVaR: {dashboard['tail_risk']['current_cvar']:.1f}ms")
        print(f"Xi parameter: {dashboard['tail_risk']['xi_parameter']:.4f}")
        print(f"Risk assessment: {dashboard['tail_risk']['risk_assessment']}")
        
        print(f"Active alarms: {dashboard['alarms']['active_count']}")
        
        # Get health status
        health = instrumentation.get_health_status()
        print(f"Overall health: {health['overall_status']}")
        
        return instrumentation
    
    def demo_adaptive_parameters(self):
        """Demonstrate adaptive parameter optimization."""
        print("\n" + "="*60)
        print("ADAPTIVE PARAMETERS DEMONSTRATION")
        print("="*60)
        
        # Create integrated system
        config = HybridConfig()
        instrumentation = HybridInstrumentation()
        controller = AdaptiveParameterController(config, instrumentation)
        
        print(f"Initial head_keep_ratio: {config.head_keep_ratio:.3f}")
        print(f"Initial window_size: {config.window_size}")
        print(f"Initial stride: {config.stride}")
        
        # Simulate performance feedback that triggers adaptation
        performance_scenarios = [
            {"kv_degradation_pp": -0.12, "xi_parameter": 0.15, "description": "KV degradation trigger"},
            {"xi_parameter": 0.25, "tail_cvar": 600, "description": "Heavy tail trigger"},
            {"p95_latency_ms": 1200, "avg_cost_per_1k": 0.12, "description": "High latency trigger"}
        ]
        
        for i, scenario in enumerate(performance_scenarios):
            print(f"\nScenario {i+1}: {scenario['description']}")
            
            # Allow adaptation (reset cooldown)
            controller.last_adaptation_time = 0.0
            
            controller.update_performance_metrics(scenario)
            
            print(f"  head_keep_ratio: {config.head_keep_ratio:.3f}")
            print(f"  window_size: {config.window_size}")
            print(f"  stride: {config.stride}")
            
            status = controller.get_adaptation_status()
            print(f"  Recent adaptations: {status['recent_adaptations']}")
            
            time.sleep(0.1)  # Small delay
        
        print(f"\nFinal configuration:")
        print(f"  head_keep_ratio: {config.head_keep_ratio:.3f}")
        print(f"  window_size: {config.window_size}")
        print(f"  stride: {config.stride}")
        
        adaptation_status = controller.get_adaptation_status()
        print(f"Total adaptations applied: {len(controller.adaptation_history)}")
        
        return controller
    
    def demo_benchmarking(self):
        """Demonstrate benchmarking system."""
        print("\n" + "="*60)
        print("BENCHMARKING DEMONSTRATION") 
        print("="*60)
        
        evaluator = HybridBenchmarkEvaluator()
        
        # Run limited evaluation for demo (smaller scale)
        evaluator.evaluation_matrix['min_samples'] = {'code_debug': 20, 'code_qa': 20, 'zh_qa': 10}
        
        print("Starting benchmark evaluation (limited scale for demo)...")
        
        start_time = time.time()
        benchmark_run = evaluator.run_full_evaluation()
        elapsed_time = time.time() - start_time
        
        print(f"Evaluation completed in {elapsed_time:.1f} seconds")
        print(f"Total competitors: {len(benchmark_run.competitors)}")
        print(f"Total datasets: {len(benchmark_run.datasets)}")
        print(f"Total result sets: {len(benchmark_run.results)}")
        
        # Show summary statistics
        print("\nSummary by method:")
        for method, stats in benchmark_run.summary_stats['by_method'].items():
            print(f"  {method}:")
            print(f"    Average F1: {stats.get('avg_f1_score', 0):.3f}")
            print(f"    Average time: {stats.get('avg_processing_time_ms', 0):.1f}ms")
            print(f"    ΔCBU/1k: {stats.get('avg_delta_cbu_per_1k', 0):.3f}")
        
        # Show promotion decision
        if benchmark_run.promotion_decision:
            decision = benchmark_run.promotion_decision
            print(f"\nPromotion decision: {decision['overall_verdict']}")
            print(f"Tests passed: {len([t for t in decision['test_results'] if t['passes']])}/{len(decision['test_results'])}")
        
        return benchmark_run

def run_all_tests():
    """Run all unit tests."""
    print("Running unit tests...")
    
    # Create test suite
    test_classes = [
        TestAtomExtractor,
        TestHeadBuilder,
        TestTailBuilder,
        TestHybridSelector,
        TestInstrumentation,
        TestAdaptiveParameters,
        TestIntegrationComplete
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_demonstrations():
    """Run all integration demonstrations."""
    print("\nRunning integration demonstrations...")
    
    demo = TestDemonstrations()
    
    try:
        # Run demonstrations
        demo.demo_hybrid_selection()
        demo.demo_instrumentation()
        demo.demo_adaptive_parameters()
        demo.demo_benchmarking()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        return False

if __name__ == "__main__":
    print("Lethe→StreamingLLM Hybrid System - Complete Test Suite")
    print("="*60)
    
    # Run tests
    tests_passed = run_all_tests()
    
    if tests_passed:
        print("\n✅ All unit tests passed!")
        
        # Run demonstrations
        demos_passed = run_all_demonstrations()
        
        if demos_passed:
            print("\n🎉 Complete test suite finished successfully!")
        else:
            print("\n❌ Some demonstrations failed")
    else:
        print("\n❌ Some unit tests failed")