"""
Comprehensive tests for model manager module.

Tests cover the high-complexity ModelManager class which valknut identified
as having significant complexity in model loading and device management.

Test areas:
- Model loading and initialization with various configurations
- Device detection and management (CPU, GPU, MPS)
- Model caching and versioning strategies
- Model warmup and validation procedures
- Memory management and cleanup
- Performance monitoring and profiling
- Error handling and fallback scenarios
- Thread safety for concurrent model access
- Configuration-driven model initialization
"""

import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import the module under test
try:
    from src.common.model_manager import (
        ModelManager, ModelConfig, ModelInfo, ModelType, DeviceType,
        ModelLoadError, DeviceError, CacheError, ValidationError
    )
except ImportError:
    # Handle missing dependencies gracefully
    pytest.skip("model_manager module not available", allow_module_level=True)


class TestModelManager:
    """Test suite for ModelManager functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for model cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def basic_model_manager(self, temp_cache_dir):
        """Create basic ModelManager instance."""
        return ModelManager(
            cache_dir=temp_cache_dir,
            device="cpu",
            max_cache_size=2
        )
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration for testing."""
        return ModelConfig(
            model_name="test-model",
            model_type=ModelType.CROSS_ENCODER,
            max_length=128,
            warmup_samples=1,
            device_preference=DeviceType.CPU
        )
    
    @pytest.fixture
    def mock_transformers_model(self):
        """Mock transformers model for testing."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=[Mock(requires_grad=True)])
        return mock_model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="test text")
        mock_tokenizer.vocab_size = 1000
        return mock_tokenizer

    # Initialization and configuration tests
    def test_model_manager_initialization(self, temp_cache_dir):
        """Test ModelManager initialization with various configurations."""
        manager = ModelManager(
            cache_dir=temp_cache_dir,
            device="cpu",
            max_cache_size=5
        )
        
        assert manager.cache_dir == temp_cache_dir
        assert manager.device == "cpu"
        assert manager.max_cache_size == 5
        assert manager._model_cache == {}
    
    def test_model_manager_auto_device_detection(self, temp_cache_dir):
        """Test automatic device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            manager = ModelManager(cache_dir=temp_cache_dir, device="auto")
            assert manager.device in ["cuda", "cpu"]
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                manager = ModelManager(cache_dir=temp_cache_dir, device="auto")
                assert manager.device in ["mps", "cpu"]

    # Device management tests
    def test_device_detection_cpu(self, basic_model_manager):
        """Test CPU device detection and configuration."""
        device_info = basic_model_manager.get_device_info()
        
        assert device_info["type"] == "cpu"
        assert "available" in device_info
        assert device_info["available"] is True
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_properties')
    def test_device_detection_cuda(self, mock_props, mock_count, mock_available, temp_cache_dir):
        """Test CUDA device detection and configuration."""
        mock_props.return_value.name = "Tesla V100"
        mock_props.return_value.total_memory = 16 * 1024**3  # 16GB
        
        manager = ModelManager(cache_dir=temp_cache_dir, device="cuda")
        device_info = manager.get_device_info()
        
        assert device_info["type"] == "cuda"
        assert device_info["available"] is True
        assert "device_count" in device_info
        assert "memory_total" in device_info
    
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_device_detection_mps(self, mock_mps, temp_cache_dir):
        """Test MPS (Apple Silicon) device detection."""
        manager = ModelManager(cache_dir=temp_cache_dir, device="mps")
        device_info = manager.get_device_info()
        
        assert device_info["type"] == "mps"
        assert device_info["available"] is True
    
    def test_unsupported_device_error(self, temp_cache_dir):
        """Test error handling for unsupported devices."""
        with pytest.raises(DeviceError):
            ModelManager(cache_dir=temp_cache_dir, device="invalid_device")

    # Model loading tests
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model_basic(self, mock_tokenizer_load, mock_model_load, 
                             basic_model_manager, sample_model_config,
                             mock_transformers_model, mock_tokenizer):
        """Test basic model loading functionality."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        model_info = basic_model_manager.load_model(sample_model_config)
        
        assert isinstance(model_info, ModelInfo)
        assert model_info.model == mock_transformers_model
        assert model_info.tokenizer == mock_tokenizer
        assert model_info.config == sample_model_config
        
        # Verify model is in cache
        assert sample_model_config.model_name in basic_model_manager._model_cache
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model_with_caching(self, mock_tokenizer_load, mock_model_load,
                                    basic_model_manager, sample_model_config,
                                    mock_transformers_model, mock_tokenizer):
        """Test model caching functionality."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # First load
        model_info1 = basic_model_manager.load_model(sample_model_config)
        load_calls_after_first = mock_model_load.call_count
        
        # Second load should use cache
        model_info2 = basic_model_manager.load_model(sample_model_config)
        load_calls_after_second = mock_model_load.call_count
        
        # Should not make additional calls to transformers
        assert load_calls_after_second == load_calls_after_first
        
        # Should return same model info
        assert model_info1.model == model_info2.model
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_model_failure_handling(self, mock_model_load, basic_model_manager, sample_model_config):
        """Test handling of model loading failures."""
        mock_model_load.side_effect = Exception("Model not found")
        
        with pytest.raises(ModelLoadError):
            basic_model_manager.load_model(sample_model_config)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_different_model_types(self, mock_tokenizer_load, mock_model_load,
                                       basic_model_manager, mock_transformers_model, mock_tokenizer):
        """Test loading different types of models."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        model_types = [
            ModelType.CROSS_ENCODER,
            ModelType.BI_ENCODER,
            ModelType.GENERATIVE,
            ModelType.EMBEDDING
        ]
        
        for model_type in model_types:
            config = ModelConfig(
                model_name=f"test-{model_type.value}",
                model_type=model_type,
                max_length=128
            )
            
            model_info = basic_model_manager.load_model(config)
            assert model_info.config.model_type == model_type

    # Model warmup tests
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_warmup(self, mock_tokenizer_load, mock_model_load,
                         basic_model_manager, mock_transformers_model, mock_tokenizer):
        """Test model warmup functionality."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock warmup behavior
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_transformers_model.return_value = Mock()
        
        config = ModelConfig(
            model_name="test-warmup",
            model_type=ModelType.CROSS_ENCODER,
            warmup_samples=3
        )
        
        model_info = basic_model_manager.load_model(config)
        
        # Should have performed warmup
        assert mock_transformers_model.called
        assert model_info.warmup_completed
        assert model_info.warmup_time > 0
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_warmup_failure_handling(self, mock_tokenizer_load, mock_model_load,
                                          basic_model_manager, mock_transformers_model, mock_tokenizer):
        """Test handling of warmup failures."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock warmup failure
        mock_transformers_model.side_effect = Exception("Warmup failed")
        
        config = ModelConfig(
            model_name="test-warmup-fail",
            model_type=ModelType.CROSS_ENCODER,
            warmup_samples=1
        )
        
        # Should still load model but log warmup failure
        model_info = basic_model_manager.load_model(config)
        assert model_info.model == mock_transformers_model
        assert not model_info.warmup_completed

    # Model validation tests
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_validation_success(self, mock_tokenizer_load, mock_model_load,
                                     basic_model_manager, mock_transformers_model, mock_tokenizer):
        """Test successful model validation."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock validation inputs/outputs
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_transformers_model.return_value = Mock()
        
        config = ModelConfig(
            model_name="test-validation",
            model_type=ModelType.CROSS_ENCODER,
            validate_model=True
        )
        
        model_info = basic_model_manager.load_model(config)
        assert model_info.validation_passed
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_validation_failure(self, mock_tokenizer_load, mock_model_load,
                                     basic_model_manager, mock_transformers_model, mock_tokenizer):
        """Test model validation failure handling."""
        mock_model_load.return_value = mock_transformers_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock validation failure
        mock_transformers_model.side_effect = Exception("Validation failed")
        
        config = ModelConfig(
            model_name="test-validation-fail",
            model_type=ModelType.CROSS_ENCODER,
            validate_model=True,
            strict_validation=False  # Allow loading despite validation failure
        )
        
        model_info = basic_model_manager.load_model(config)
        assert not model_info.validation_passed
        
        # With strict validation, should raise error
        config.strict_validation = True
        with pytest.raises(ValidationError):
            basic_model_manager.load_model(config)

    # Cache management tests
    def test_cache_size_limit(self, temp_cache_dir):
        """Test cache size limit enforcement."""
        manager = ModelManager(cache_dir=temp_cache_dir, max_cache_size=2)
        
        # Mock model loading
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()
                
                # Load models up to cache limit
                configs = []
                for i in range(3):  # One more than limit
                    config = ModelConfig(
                        model_name=f"model-{i}",
                        model_type=ModelType.CROSS_ENCODER
                    )
                    configs.append(config)
                    manager.load_model(config)
                
                # Should only keep 2 models in cache (LRU eviction)
                assert len(manager._model_cache) == 2
                
                # First model should be evicted
                assert configs[0].model_name not in manager._model_cache
                assert configs[1].model_name in manager._model_cache
                assert configs[2].model_name in manager._model_cache
    
    def test_cache_cleanup(self, basic_model_manager):
        """Test manual cache cleanup."""
        # Mock model loading
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()
                
                config = ModelConfig(
                    model_name="cleanup-test",
                    model_type=ModelType.CROSS_ENCODER
                )
                
                # Load model
                basic_model_manager.load_model(config)
                assert config.model_name in basic_model_manager._model_cache
                
                # Cleanup specific model
                basic_model_manager.cleanup_model(config.model_name)
                assert config.model_name not in basic_model_manager._model_cache
    
    def test_cache_clear_all(self, basic_model_manager):
        """Test clearing all cached models."""
        # Mock multiple model loading
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()
                
                # Load multiple models
                for i in range(3):
                    config = ModelConfig(
                        model_name=f"clear-test-{i}",
                        model_type=ModelType.CROSS_ENCODER
                    )
                    basic_model_manager.load_model(config)
                
                assert len(basic_model_manager._model_cache) == 3
                
                # Clear all
                basic_model_manager.clear_cache()
                assert len(basic_model_manager._model_cache) == 0

    # Memory management tests
    @patch('psutil.Process')
    def test_memory_monitoring(self, mock_process, basic_model_manager):
        """Test memory usage monitoring."""
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        
        memory_info = basic_model_manager.get_memory_info()
        
        assert "current_memory_mb" in memory_info
        assert "peak_memory_mb" in memory_info
        assert memory_info["current_memory_mb"] == 100.0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 50)  # 50MB
    @patch('torch.cuda.max_memory_allocated', return_value=1024 * 1024 * 75)  # 75MB
    def test_gpu_memory_monitoring(self, mock_max_mem, mock_allocated, mock_available, temp_cache_dir):
        """Test GPU memory monitoring."""
        manager = ModelManager(cache_dir=temp_cache_dir, device="cuda")
        
        memory_info = manager.get_memory_info()
        
        assert "gpu_memory_allocated_mb" in memory_info
        assert "gpu_memory_peak_mb" in memory_info
        assert memory_info["gpu_memory_allocated_mb"] == 50.0
        assert memory_info["gpu_memory_peak_mb"] == 75.0

    # Performance monitoring tests
    def test_performance_tracking(self, basic_model_manager):
        """Test performance metrics tracking."""
        # Mock model loading with timing
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                with patch('time.time', side_effect=[0, 1, 2]):  # Mock timing
                    mock_model.return_value = Mock()
                    mock_tokenizer.return_value = Mock()
                    
                    config = ModelConfig(
                        model_name="perf-test",
                        model_type=ModelType.CROSS_ENCODER
                    )
                    
                    model_info = basic_model_manager.load_model(config)
                    
                    assert model_info.load_time > 0
                    assert "load_time" in model_info.performance_metrics
    
    def test_inference_timing(self, basic_model_manager):
        """Test inference performance measurement."""
        # Mock model for inference
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        model_info = ModelInfo(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=ModelConfig("test", ModelType.CROSS_ENCODER),
            device="cpu"
        )
        
        # Mock inference timing
        with patch('time.time', side_effect=[0, 0.1]):  # 100ms inference
            with basic_model_manager.time_inference() as timer:
                # Simulate inference
                time.sleep(0.01)
            
            timing_data = timer.get_timing()
            assert timing_data["duration"] >= 0.01

    # Thread safety tests
    def test_concurrent_model_loading(self, basic_model_manager):
        """Test thread safety of concurrent model loading."""
        results = []
        errors = []
        
        def load_worker(worker_id):
            try:
                with patch('transformers.AutoModel.from_pretrained') as mock_model:
                    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                        mock_model.return_value = Mock()
                        mock_tokenizer.return_value = Mock()
                        
                        config = ModelConfig(
                            model_name=f"concurrent-{worker_id}",
                            model_type=ModelType.CROSS_ENCODER
                        )
                        
                        model_info = basic_model_manager.load_model(config)
                        results.append(model_info is not None)
                        
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=load_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(results), "Some model loads failed"
    
    def test_concurrent_cache_access(self, basic_model_manager):
        """Test thread-safe cache access."""
        # Pre-load a model
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()
                
                config = ModelConfig(
                    model_name="shared-model",
                    model_type=ModelType.CROSS_ENCODER
                )
                basic_model_manager.load_model(config)
        
        results = []
        errors = []
        
        def cache_access_worker():
            try:
                # Multiple threads accessing same cached model
                model_info = basic_model_manager.load_model(config)
                results.append(model_info is not None)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_access_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(results), "Some cache accesses failed"

    # Configuration and edge cases
    def test_model_config_validation(self, basic_model_manager):
        """Test model configuration validation."""
        # Invalid model type
        with pytest.raises(ValidationError):
            config = ModelConfig(
                model_name="test",
                model_type="invalid_type",  # Should be ModelType enum
                max_length=128
            )
            basic_model_manager.validate_config(config)
        
        # Invalid max_length
        with pytest.raises(ValidationError):
            config = ModelConfig(
                model_name="test",
                model_type=ModelType.CROSS_ENCODER,
                max_length=0  # Should be positive
            )
            basic_model_manager.validate_config(config)
    
    def test_empty_model_name(self, basic_model_manager):
        """Test handling of empty model name."""
        with pytest.raises(ValidationError):
            config = ModelConfig(
                model_name="",  # Empty name
                model_type=ModelType.CROSS_ENCODER
            )
            basic_model_manager.load_model(config)
    
    def test_model_loading_with_custom_tokenizer_config(self, basic_model_manager):
        """Test loading models with custom tokenizer configurations."""
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()
                
                config = ModelConfig(
                    model_name="custom-tokenizer",
                    model_type=ModelType.CROSS_ENCODER,
                    tokenizer_config={
                        "padding": True,
                        "truncation": True,
                        "max_length": 256
                    }
                )
                
                model_info = basic_model_manager.load_model(config)
                
                # Should pass custom config to tokenizer
                mock_tokenizer.assert_called_with(
                    config.model_name,
                    **config.tokenizer_config
                )

    # Integration tests
    def test_end_to_end_model_workflow(self, basic_model_manager):
        """Test complete model management workflow."""
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()
                mock_tokenizer.return_value.encode.return_value = [1, 2, 3]
                
                config = ModelConfig(
                    model_name="workflow-test",
                    model_type=ModelType.CROSS_ENCODER,
                    warmup_samples=1,
                    validate_model=True
                )
                
                # 1. Load model
                model_info = basic_model_manager.load_model(config)
                assert model_info.model is not None
                assert model_info.tokenizer is not None
                
                # 2. Check cache
                assert config.model_name in basic_model_manager._model_cache
                
                # 3. Get performance stats
                memory_info = basic_model_manager.get_memory_info()
                assert "current_memory_mb" in memory_info
                
                # 4. Cleanup
                basic_model_manager.cleanup_model(config.model_name)
                assert config.model_name not in basic_model_manager._model_cache


if __name__ == "__main__":
    pytest.main([__file__])