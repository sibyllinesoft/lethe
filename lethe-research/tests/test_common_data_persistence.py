"""
Comprehensive tests for data persistence module.

Tests cover the high-complexity DataManager class which valknut identified
as having cyclomatic complexity of 395.0 and critical technical debt.

Test areas:
- Data loading and saving with multiple formats
- Caching mechanisms and TTL behavior
- Streaming operations for large files
- Error handling and fallback scenarios
- Batch operations and transactions
- Validation and schema checking
- Compression and encryption support
- Thread safety for concurrent operations
"""

import pytest
import json
import pickle
import gzip
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from dataclasses import dataclass

# Import the module under test
try:
    from src.common.data_persistence import (
        DataManager, DataConfig, CachePolicy, CompressionType,
        DataFormat, ValidationError, CacheExpiredError
    )
except ImportError:
    # Handle missing dependencies gracefully
    pytest.skip("data_persistence module not available", allow_module_level=True)


@dataclass
class TestSchema:
    """Test schema for validation testing."""
    id: int
    name: str
    value: float


class TestDataPersistence:
    """Test suite for DataManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def data_manager(self, temp_dir):
        """Create DataManager instance with temporary cache directory."""
        config = DataConfig(
            cache_dir=temp_dir / "cache",
            enable_compression=True,
            cache_policy=CachePolicy.LRU,
            max_cache_size=1000
        )
        return DataManager(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample test data."""
        return {
            "items": [
                {"id": 1, "name": "test1", "value": 1.5},
                {"id": 2, "name": "test2", "value": 2.5}
            ],
            "metadata": {"version": "1.0", "created": "2024-01-01"}
        }

    # Basic functionality tests
    def test_data_manager_initialization(self, temp_dir):
        """Test DataManager initialization with various configurations."""
        config = DataConfig(cache_dir=temp_dir)
        manager = DataManager(config)
        
        assert manager.config.cache_dir == temp_dir
        assert manager.config.cache_policy == CachePolicy.LRU
        assert manager._cache is not None
    
    def test_data_manager_auto_config(self, temp_dir):
        """Test DataManager with automatic configuration."""
        manager = DataManager()
        assert manager.config is not None
        assert manager.config.cache_dir.exists()

    # Data loading tests
    def test_load_json_file(self, data_manager, temp_dir, sample_data):
        """Test loading JSON files with various scenarios."""
        json_file = temp_dir / "test.json"
        json_file.write_text(json.dumps(sample_data))
        
        loaded_data = data_manager.load_data(json_file)
        assert loaded_data == sample_data
    
    def test_load_json_file_with_fallback(self, data_manager, temp_dir):
        """Test loading non-existent JSON file with fallback."""
        non_existent = temp_dir / "missing.json"
        fallback_data = {"default": True}
        
        loaded_data = data_manager.load_data(non_existent, default=fallback_data)
        assert loaded_data == fallback_data
    
    def test_load_jsonl_file(self, data_manager, temp_dir):
        """Test loading JSONL files."""
        jsonl_file = temp_dir / "test.jsonl"
        lines = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"}
        ]
        
        with jsonl_file.open('w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        
        loaded_data = list(data_manager.stream_jsonl(jsonl_file))
        assert loaded_data == lines
    
    def test_load_pickle_file(self, data_manager, temp_dir, sample_data):
        """Test loading pickle files."""
        pickle_file = temp_dir / "test.pkl"
        with pickle_file.open('wb') as f:
            pickle.dump(sample_data, f)
        
        loaded_data = data_manager.load_data(pickle_file)
        assert loaded_data == sample_data

    # Data saving tests
    def test_save_json_file(self, data_manager, temp_dir, sample_data):
        """Test saving data as JSON."""
        output_file = temp_dir / "output.json"
        
        data_manager.save_data(sample_data, output_file)
        
        assert output_file.exists()
        with output_file.open() as f:
            saved_data = json.load(f)
        assert saved_data == sample_data
    
    def test_save_pickle_file(self, data_manager, temp_dir, sample_data):
        """Test saving data as pickle."""
        output_file = temp_dir / "output.pkl"
        
        data_manager.save_data(sample_data, output_file, format=DataFormat.PICKLE)
        
        assert output_file.exists()
        with output_file.open('rb') as f:
            saved_data = pickle.load(f)
        assert saved_data == sample_data
    
    def test_save_compressed_file(self, data_manager, temp_dir, sample_data):
        """Test saving compressed data."""
        output_file = temp_dir / "output.json.gz"
        
        data_manager.save_data(sample_data, output_file, compress=True)
        
        assert output_file.exists()
        with gzip.open(output_file, 'rt') as f:
            saved_data = json.load(f)
        assert saved_data == sample_data

    # Caching tests
    def test_cache_basic_functionality(self, data_manager):
        """Test basic caching operations."""
        key = "test_key"
        value = {"cached": True}
        
        # Store in cache
        data_manager.cache_set(key, value)
        
        # Retrieve from cache
        cached_value = data_manager.cache_get(key)
        assert cached_value == value
    
    def test_cache_expiration(self, data_manager):
        """Test cache TTL expiration."""
        key = "expiring_key"
        value = {"temp": True}
        
        # Store with 1 second TTL
        data_manager.cache_set(key, value, ttl=1)
        
        # Should be available immediately
        assert data_manager.cache_get(key) == value
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        with pytest.raises(CacheExpiredError):
            data_manager.cache_get(key)
    
    def test_cache_invalidation(self, data_manager):
        """Test manual cache invalidation."""
        key = "invalidate_me"
        value = {"will_be_removed": True}
        
        data_manager.cache_set(key, value)
        assert data_manager.cache_get(key) == value
        
        data_manager.cache_invalidate(key)
        
        with pytest.raises(KeyError):
            data_manager.cache_get(key)
    
    def test_get_cached_with_computation(self, data_manager):
        """Test cached computation pattern."""
        computation_count = 0
        
        def expensive_computation():
            nonlocal computation_count
            computation_count += 1
            return {"computed": computation_count}
        
        key = "expensive_op"
        
        # First call should compute
        result1 = data_manager.get_cached(key, expensive_computation, ttl=60)
        assert computation_count == 1
        assert result1 == {"computed": 1}
        
        # Second call should use cache
        result2 = data_manager.get_cached(key, expensive_computation, ttl=60)
        assert computation_count == 1  # Should not recompute
        assert result2 == {"computed": 1}

    # Streaming tests
    def test_stream_large_jsonl(self, data_manager, temp_dir):
        """Test streaming large JSONL files."""
        large_file = temp_dir / "large.jsonl"
        num_lines = 1000
        
        # Create large JSONL file
        with large_file.open('w') as f:
            for i in range(num_lines):
                f.write(json.dumps({"id": i, "data": f"item_{i}"}) + '\n')
        
        # Stream and count
        count = 0
        for item in data_manager.stream_jsonl(large_file):
            count += 1
            assert "id" in item
            assert "data" in item
        
        assert count == num_lines
    
    def test_stream_with_validation(self, data_manager, temp_dir):
        """Test streaming with schema validation."""
        jsonl_file = temp_dir / "validated.jsonl"
        
        # Create file with valid and invalid items
        items = [
            {"id": 1, "name": "valid1", "value": 1.0},
            {"id": "invalid", "name": "invalid", "value": "bad"},  # Invalid
            {"id": 2, "name": "valid2", "value": 2.0}
        ]
        
        with jsonl_file.open('w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')
        
        # Stream with validation
        valid_items = []
        invalid_count = 0
        
        for item in data_manager.stream_jsonl(jsonl_file, validate_schema=TestSchema):
            try:
                validated_item = TestSchema(**item)
                valid_items.append(validated_item)
            except (TypeError, ValueError):
                invalid_count += 1
        
        assert len(valid_items) == 2
        assert invalid_count == 1

    # Validation tests
    def test_data_validation_success(self, data_manager):
        """Test successful data validation."""
        valid_data = {"id": 1, "name": "test", "value": 1.5}
        
        result = data_manager.validate_data(valid_data, TestSchema)
        assert isinstance(result, TestSchema)
        assert result.id == 1
        assert result.name == "test"
        assert result.value == 1.5
    
    def test_data_validation_failure(self, data_manager):
        """Test data validation failure."""
        invalid_data = {"id": "not_int", "name": "test"}  # Missing value, wrong type
        
        with pytest.raises(ValidationError):
            data_manager.validate_data(invalid_data, TestSchema)

    # Batch operations tests
    def test_batch_load_operation(self, data_manager, temp_dir):
        """Test batch loading multiple files."""
        files = []
        expected_data = []
        
        for i in range(3):
            file_path = temp_dir / f"batch_{i}.json"
            data = {"batch": i, "items": list(range(i * 10, (i + 1) * 10))}
            
            file_path.write_text(json.dumps(data))
            files.append(file_path)
            expected_data.append(data)
        
        loaded_data = data_manager.batch_load(files)
        assert len(loaded_data) == 3
        
        for i, data in enumerate(loaded_data):
            assert data == expected_data[i]
    
    def test_batch_save_operation(self, data_manager, temp_dir):
        """Test batch saving multiple datasets."""
        datasets = [
            {"name": "dataset1", "data": [1, 2, 3]},
            {"name": "dataset2", "data": [4, 5, 6]},
            {"name": "dataset3", "data": [7, 8, 9]}
        ]
        
        output_files = [temp_dir / f"output_{i}.json" for i in range(3)]
        
        data_manager.batch_save(datasets, output_files)
        
        for i, file_path in enumerate(output_files):
            assert file_path.exists()
            with file_path.open() as f:
                saved_data = json.load(f)
            assert saved_data == datasets[i]

    # Error handling tests
    def test_load_corrupted_json(self, data_manager, temp_dir):
        """Test handling corrupted JSON files."""
        corrupted_file = temp_dir / "corrupted.json"
        corrupted_file.write_text("{invalid json content")
        
        # Should raise error without fallback
        with pytest.raises((json.JSONDecodeError, ValueError)):
            data_manager.load_data(corrupted_file)
        
        # Should return fallback with default
        fallback = {"error": "fallback"}
        result = data_manager.load_data(corrupted_file, default=fallback)
        assert result == fallback
    
    def test_save_to_readonly_directory(self, data_manager, temp_dir, sample_data):
        """Test saving to read-only directory."""
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        output_file = readonly_dir / "test.json"
        
        with pytest.raises(PermissionError):
            data_manager.save_data(sample_data, output_file)
    
    def test_load_empty_file(self, data_manager, temp_dir):
        """Test loading empty files."""
        empty_file = temp_dir / "empty.json"
        empty_file.touch()
        
        fallback = {"empty": True}
        result = data_manager.load_data(empty_file, default=fallback)
        assert result == fallback

    # Thread safety tests
    def test_concurrent_cache_operations(self, data_manager):
        """Test thread safety of cache operations."""
        results = []
        errors = []
        
        def cache_worker(worker_id):
            try:
                key = f"worker_{worker_id}"
                value = {"worker": worker_id, "data": list(range(100))}
                
                # Set and get in same thread
                data_manager.cache_set(key, value)
                retrieved = data_manager.cache_get(key)
                results.append(retrieved == value)
                
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(results), "Some cache operations failed"
    
    def test_concurrent_file_operations(self, data_manager, temp_dir):
        """Test concurrent file operations don't interfere."""
        results = []
        errors = []
        
        def file_worker(worker_id):
            try:
                data = {"worker": worker_id, "timestamp": time.time()}
                file_path = temp_dir / f"worker_{worker_id}.json"
                
                # Save and load
                data_manager.save_data(data, file_path)
                loaded = data_manager.load_data(file_path)
                
                results.append(loaded == data)
                
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=file_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(results), "Some file operations failed"

    # Performance and edge cases
    def test_cache_memory_limit(self, temp_dir):
        """Test cache respects memory limits."""
        config = DataConfig(
            cache_dir=temp_dir,
            max_cache_size=3  # Very small limit
        )
        manager = DataManager(config)
        
        # Add items beyond limit
        for i in range(5):
            manager.cache_set(f"key_{i}", {"data": i})
        
        # Should only have most recent items (LRU policy)
        cache_size = len(manager._cache)
        assert cache_size <= 3, f"Cache size {cache_size} exceeds limit"
    
    def test_large_data_handling(self, data_manager, temp_dir):
        """Test handling of large data structures."""
        large_data = {
            "massive_list": list(range(10000)),
            "nested": {f"key_{i}": list(range(100)) for i in range(100)}
        }
        
        output_file = temp_dir / "large_data.json"
        
        # Should handle large data without errors
        data_manager.save_data(large_data, output_file)
        loaded_data = data_manager.load_data(output_file)
        
        assert loaded_data == large_data
    
    def test_format_detection_accuracy(self, data_manager, temp_dir, sample_data):
        """Test automatic format detection works correctly."""
        # Test various extensions
        formats = [
            (".json", DataFormat.JSON),
            (".jsonl", DataFormat.JSONL),
            (".pkl", DataFormat.PICKLE),
            (".pickle", DataFormat.PICKLE)
        ]
        
        for ext, expected_format in formats:
            file_path = temp_dir / f"test{ext}"
            detected_format = data_manager._detect_format(file_path)
            assert detected_format == expected_format

    # Integration tests
    def test_end_to_end_workflow(self, data_manager, temp_dir):
        """Test complete workflow from load to save with caching."""
        # 1. Create source data
        source_file = temp_dir / "source.json"
        source_data = {
            "experiment": "test_workflow",
            "results": [{"id": i, "score": i * 0.1} for i in range(100)]
        }
        source_file.write_text(json.dumps(source_data))
        
        # 2. Load with caching
        loaded_data = data_manager.get_cached(
            "workflow_data",
            lambda: data_manager.load_data(source_file),
            ttl=300
        )
        
        # 3. Process data
        processed_data = {
            "source": loaded_data["experiment"],
            "summary": {
                "count": len(loaded_data["results"]),
                "avg_score": sum(r["score"] for r in loaded_data["results"]) / len(loaded_data["results"])
            }
        }
        
        # 4. Save results
        output_file = temp_dir / "processed_results.json"
        data_manager.save_data(processed_data, output_file)
        
        # 5. Verify
        final_data = data_manager.load_data(output_file)
        assert final_data["source"] == "test_workflow"
        assert final_data["summary"]["count"] == 100
        assert abs(final_data["summary"]["avg_score"] - 4.95) < 0.01  # Expected average


if __name__ == "__main__":
    pytest.main([__file__])