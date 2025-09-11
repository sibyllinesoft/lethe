"""
Comprehensive tests for retriever.embeddings module.

Tests the DenseEmbeddingManager with mocking for external dependencies
to ensure the functionality works correctly without requiring actual model downloads.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass

from src.retriever.config import EmbeddingConfig
from src.retriever.embeddings import (
    EmbeddingMetadata,
    DenseEmbeddingManager,
    create_embedding_manager,
    TRANSFORMERS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE
)


class TestEmbeddingMetadata:
    """Tests for EmbeddingMetadata dataclass."""
    
    def test_embedding_metadata_creation(self):
        """Test creating EmbeddingMetadata with all fields."""
        metadata = EmbeddingMetadata(
            model_name="test-model",
            model_hash="abc123",
            collection_name="test-collection",
            num_embeddings=100,
            embedding_dim=384,
            max_length=512,
            normalize=True,
            batch_size=32,
            content_hash="def456",
            storage_format="numpy",
            file_path="/path/to/file.npy",
            file_size_mb=10.5,
            encoding_time_sec=5.2,
            throughput_docs_per_sec=19.2
        )
        
        assert metadata.model_name == "test-model"
        assert metadata.model_hash == "abc123"
        assert metadata.num_embeddings == 100
        assert metadata.embedding_dim == 384
        
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = EmbeddingMetadata(
            model_name="test-model",
            model_hash="abc123",
            collection_name="test-collection",
            num_embeddings=100,
            embedding_dim=384,
            max_length=512,
            normalize=True,
            batch_size=32,
            content_hash="def456",
            storage_format="numpy",
            file_path="/path/to/file.npy",
            file_size_mb=10.5,
            encoding_time_sec=5.2,
            throughput_docs_per_sec=19.2
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result["model_name"] == "test-model"
        assert result["num_embeddings"] == 100
        assert result["normalize"] is True
        assert result["file_size_mb"] == 10.5
        
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "model_name": "test-model",
            "model_hash": "abc123",
            "collection_name": "test-collection",
            "num_embeddings": 100,
            "embedding_dim": 384,
            "max_length": 512,
            "normalize": True,
            "batch_size": 32,
            "content_hash": "def456",
            "storage_format": "numpy",
            "file_path": "/path/to/file.npy",
            "file_size_mb": 10.5,
            "encoding_time_sec": 5.2,
            "throughput_docs_per_sec": 19.2,
            "created_at": "2024-01-01 12:00:00"
        }
        
        metadata = EmbeddingMetadata.from_dict(data)
        
        assert metadata.model_name == "test-model"
        assert metadata.num_embeddings == 100
        assert metadata.normalize is True
        assert metadata.file_size_mb == 10.5


class TestDenseEmbeddingManager:
    """Tests for DenseEmbeddingManager."""
    
    @pytest.fixture
    def config(self):
        """Create test embedding configuration."""
        return EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=4,
            max_length=128,
            device="cpu",
            normalize_embeddings=True,
            cache_embeddings=True
        )
        
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def mock_timing_harness(self):
        """Create mock timing harness."""
        mock_harness = Mock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock(return_value=None)
        mock_harness.measure = Mock(return_value=mock_context)
        return mock_harness
        
    def test_initialization_valid_config(self, config, temp_cache_dir):
        """Test successful initialization with valid config."""
        manager = DenseEmbeddingManager(
            config=config,
            cache_dir=temp_cache_dir
        )
        
        assert manager.config == config
        assert manager.cache_dir == temp_cache_dir
        assert manager.cache_dir.exists()
        
    def test_initialization_invalid_config(self, temp_cache_dir):
        """Test initialization fails with invalid config."""
        invalid_config = EmbeddingConfig(batch_size=-1)  # Invalid
        
        with pytest.raises(ValueError, match="Invalid embedding configuration"):
            DenseEmbeddingManager(
                config=invalid_config,
                cache_dir=temp_cache_dir
            )
            
    def test_cache_dir_creation(self, config):
        """Test cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "new_cache"
            assert not cache_path.exists()
            
            manager = DenseEmbeddingManager(
                config=config,
                cache_dir=cache_path
            )
            
            assert cache_path.exists()
            assert cache_path.is_dir()
            
    @patch('src.retriever.embeddings.torch')
    @patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('src.retriever.embeddings.SentenceTransformer')
    def test_model_initialization_sentence_transformers(self, mock_st, mock_torch, config, temp_cache_dir):
        """Test model initialization with sentence-transformers."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        manager._initialize_model()
        
        mock_st.assert_called_once_with(
            config.model_name,
            cache_folder=config.model_cache_dir,
            device="cpu"
        )
        assert manager._model == mock_model
        assert manager._device == "cpu"
        
    @patch('src.retriever.embeddings.torch')
    @patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    @patch('src.retriever.embeddings.TRANSFORMERS_AVAILABLE', True)
    @patch('src.retriever.embeddings.AutoTokenizer')
    @patch('src.retriever.embeddings.AutoModel')
    def test_model_initialization_transformers_fallback(self, mock_auto_model, mock_auto_tokenizer, 
                                                       mock_torch, config, temp_cache_dir):
        """Test model initialization with transformers fallback."""
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        manager._initialize_model()
        
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            config.model_name,
            cache_dir=config.model_cache_dir
        )
        mock_auto_model.from_pretrained.assert_called_once_with(
            config.model_name,
            cache_dir=config.model_cache_dir
        )
        assert manager._model == mock_model
        assert manager._tokenizer == mock_tokenizer
        
    @patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    @patch('src.retriever.embeddings.TRANSFORMERS_AVAILABLE', False)
    def test_model_initialization_no_libraries(self, config, temp_cache_dir):
        """Test model initialization fails when no libraries available."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        with pytest.raises(ImportError, match="Neither sentence-transformers nor transformers available"):
            manager._initialize_model()
            
    def test_compute_model_hash(self, config, temp_cache_dir):
        """Test model hash computation."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        hash_val = manager._compute_model_hash()
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # Short hash
        
        # Hash should be consistent
        hash_val2 = manager._compute_model_hash()
        assert hash_val == hash_val2
        
    def test_encode_texts_empty_input(self, config, temp_cache_dir):
        """Test encoding empty text list."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        result = manager.encode_texts([])
        
        assert result.shape == (0, 0)
        
    @patch('src.retriever.embeddings.torch')
    @patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('src.retriever.embeddings.SentenceTransformer')
    def test_encode_texts_sentence_transformers(self, mock_st, mock_torch, config, temp_cache_dir):
        """Test text encoding with sentence-transformers."""
        mock_torch.cuda.is_available.return_value = False
        
        # Mock model
        mock_model = Mock()
        mock_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        texts = ["test text 1", "test text 2"]
        
        result = manager.encode_texts(texts, show_progress=False)
        
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=config.normalize_embeddings,
            convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, mock_embeddings)
        
    @patch('src.retriever.embeddings.torch')
    @patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', False)  
    @patch('src.retriever.embeddings.TRANSFORMERS_AVAILABLE', True)
    @patch('src.retriever.embeddings.AutoTokenizer')
    @patch('src.retriever.embeddings.AutoModel')
    def test_encode_texts_transformers_fallback(self, mock_auto_model, mock_auto_tokenizer,
                                               mock_torch, config, temp_cache_dir):
        """Test text encoding with transformers fallback."""
        mock_torch.cuda.is_available.return_value = False
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_inputs = {
            'input_ids': mock_torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': mock_torch.tensor([[1, 1, 1], [1, 1, 0]])
        }
        mock_tokenizer.return_value = mock_inputs
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = mock_torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]]
        ])
        mock_model.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Mock tensor operations
        mock_torch.no_grad = MagicMock()
        mock_torch.nn.functional.normalize = Mock(side_effect=lambda x, p, dim: x)
        
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        texts = ["test text 1", "test text 2"]
        
        result = manager.encode_texts(texts, show_progress=False)
        
        # Verify tokenizer was called correctly
        mock_tokenizer.assert_called()
        
        # Verify model was called
        mock_model.assert_called()
        
    def test_encode_texts_with_timing_harness(self, config, temp_cache_dir, mock_timing_harness):
        """Test encoding with timing harness."""
        with patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st, \
             patch('src.retriever.embeddings.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1.0, 2.0]])
            mock_st.return_value = mock_model
            
            manager = DenseEmbeddingManager(config, temp_cache_dir, mock_timing_harness)
            manager.encode_texts(["test"], show_progress=False)
            
            mock_timing_harness.measure.assert_called_once_with(
                "dense_encoding",
                {"num_texts": 1, "batch_size": config.batch_size}
            )
            
    @patch('src.retriever.embeddings.time.time')
    def test_encode_collection_no_cache(self, mock_time, config, temp_cache_dir):
        """Test encoding collection without cache."""
        mock_time.side_effect = [100.0, 105.0]  # Start and end times
        
        with patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st, \
             patch('src.retriever.embeddings.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            config.cache_embeddings = False
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            
            documents = [
                {"text": "test doc 1", "id": "1"},
                {"text": "test doc 2", "id": "2"}
            ]
            
            embeddings, metadata = manager.encode_collection(
                documents, "test-collection"
            )
            
            np.testing.assert_array_equal(embeddings, mock_embeddings)
            assert metadata.collection_name == "test-collection"
            assert metadata.num_embeddings == 2
            assert metadata.embedding_dim == 3
            assert metadata.encoding_time_sec == 5.0
            assert metadata.throughput_docs_per_sec == 2.0 / 5.0
            
    def test_encode_collection_empty_documents(self, config, temp_cache_dir):
        """Test encoding collection with empty documents."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        with pytest.raises(ValueError, match="No texts found in documents"):
            manager.encode_collection([], "test-collection")
            
    def test_encode_collection_missing_text_field(self, config, temp_cache_dir):
        """Test encoding collection with missing text field."""
        with patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st, \
             patch('src.retriever.embeddings.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1.0, 2.0]])
            mock_st.return_value = mock_model
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            
            # Documents without text field get empty string
            documents = [{"id": "1"}]
            
            embeddings, metadata = manager.encode_collection(
                documents, "test-collection"
            )
            
            # Should encode empty strings
            mock_model.encode.assert_called()
            call_args = mock_model.encode.call_args[0][0]
            assert call_args == [""]  # Empty string for missing text field
            
    def test_compute_content_hash(self, config, temp_cache_dir):
        """Test content hash computation."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        texts = ["text 1", "text 2", "text 3"]
        hash_val = manager._compute_content_hash(texts)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex digest
        
        # Hash should be consistent
        hash_val2 = manager._compute_content_hash(texts)
        assert hash_val == hash_val2
        
        # Different content should produce different hash
        hash_val3 = manager._compute_content_hash(["different text"])
        assert hash_val != hash_val3
        
    def test_get_cache_path(self, config, temp_cache_dir):
        """Test cache path generation."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        embeddings_path, metadata_path = manager._get_cache_path("test-collection")
        
        assert embeddings_path == temp_cache_dir / "test-collection_embeddings.npy"
        assert metadata_path == temp_cache_dir / "test-collection_metadata.json"
        
    def test_cache_embeddings(self, config, temp_cache_dir):
        """Test caching embeddings to disk."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        metadata = EmbeddingMetadata(
            model_name="test-model",
            model_hash="abc123",
            collection_name="test-collection",
            num_embeddings=2,
            embedding_dim=2,
            max_length=512,
            normalize=True,
            batch_size=32,
            content_hash="def456",
            storage_format="numpy",
            file_path="",
            file_size_mb=0.0,
            encoding_time_sec=1.0,
            throughput_docs_per_sec=2.0
        )
        
        manager._cache_embeddings(embeddings, metadata)
        
        # Check files were created
        embeddings_path = temp_cache_dir / "test-collection_embeddings.npy"
        metadata_path = temp_cache_dir / "test-collection_metadata.json"
        
        assert embeddings_path.exists()
        assert metadata_path.exists()
        
        # Check embeddings file
        loaded_embeddings = np.load(embeddings_path)
        np.testing.assert_array_equal(loaded_embeddings, embeddings)
        
        # Check metadata file
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
            
        assert loaded_metadata["collection_name"] == "test-collection"
        assert loaded_metadata["num_embeddings"] == 2
        assert loaded_metadata["file_path"] == str(embeddings_path)
        assert loaded_metadata["file_size_mb"] > 0
        
    def test_load_cached_embeddings_not_exists(self, config, temp_cache_dir):
        """Test loading cached embeddings when files don't exist."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        embeddings, metadata = manager._load_cached_embeddings("nonexistent")
        
        assert embeddings is None
        assert metadata is None
        
    def test_load_cached_embeddings_model_mismatch(self, config, temp_cache_dir):
        """Test loading cached embeddings with model mismatch."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        # Create cached files with different model
        embeddings_path = temp_cache_dir / "test-collection_embeddings.npy"
        metadata_path = temp_cache_dir / "test-collection_metadata.json"
        
        embeddings = np.array([[1.0, 2.0]])
        np.save(embeddings_path, embeddings)
        
        metadata_dict = {
            "model_name": "different-model",  # Different from config
            "collection_name": "test-collection",
            "num_embeddings": 1,
            "embedding_dim": 2
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
            
        result_embeddings, result_metadata = manager._load_cached_embeddings("test-collection")
        
        assert result_embeddings is None
        assert result_metadata is None
        
    def test_load_cached_embeddings_success(self, config, temp_cache_dir):
        """Test successful loading of cached embeddings."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        # Create cached files
        embeddings_path = temp_cache_dir / "test-collection_embeddings.npy"
        metadata_path = temp_cache_dir / "test-collection_metadata.json"
        
        original_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.save(embeddings_path, original_embeddings)
        
        metadata_dict = {
            "model_name": config.model_name,  # Same as config
            "model_hash": "abc123",
            "collection_name": "test-collection",
            "num_embeddings": 2,
            "embedding_dim": 2,
            "max_length": 512,
            "normalize": True,
            "batch_size": 32,
            "content_hash": "def456",
            "storage_format": "numpy",
            "file_path": str(embeddings_path),
            "file_size_mb": 1.0,
            "encoding_time_sec": 1.0,
            "throughput_docs_per_sec": 2.0,
            "created_at": "2024-01-01 12:00:00"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
            
        loaded_embeddings, loaded_metadata = manager._load_cached_embeddings("test-collection")
        
        np.testing.assert_array_equal(loaded_embeddings, original_embeddings)
        assert loaded_metadata.collection_name == "test-collection"
        assert loaded_metadata.num_embeddings == 2
        
    def test_list_cached_collections(self, config, temp_cache_dir):
        """Test listing cached collections."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        # Create some metadata files
        (temp_cache_dir / "collection1_metadata.json").touch()
        (temp_cache_dir / "collection2_metadata.json").touch()
        (temp_cache_dir / "other_file.txt").touch()  # Should be ignored
        
        collections = manager.list_cached_collections()
        
        assert set(collections) == {"collection1", "collection2"}
        assert collections == sorted(collections)  # Should be sorted
        
    def test_get_cached_metadata_success(self, config, temp_cache_dir):
        """Test getting cached metadata."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        metadata_path = temp_cache_dir / "test-collection_metadata.json"
        metadata_dict = {
            "model_name": "test-model",
            "model_hash": "abc123",
            "collection_name": "test-collection",
            "num_embeddings": 100,
            "embedding_dim": 384,
            "max_length": 512,
            "normalize": True,
            "batch_size": 32,
            "content_hash": "def456",
            "storage_format": "numpy",
            "file_path": "/path/to/file.npy",
            "file_size_mb": 10.5,
            "encoding_time_sec": 5.2,
            "throughput_docs_per_sec": 19.2,
            "created_at": "2024-01-01 12:00:00"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
            
        metadata = manager.get_cached_metadata("test-collection")
        
        assert metadata.model_name == "test-model"
        assert metadata.num_embeddings == 100
        
    def test_get_cached_metadata_not_exists(self, config, temp_cache_dir):
        """Test getting cached metadata when file doesn't exist."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        metadata = manager.get_cached_metadata("nonexistent")
        
        assert metadata is None
        
    def test_clear_cache_specific_collection(self, config, temp_cache_dir):
        """Test clearing cache for specific collection."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        # Create test files
        embeddings_path = temp_cache_dir / "test-collection_embeddings.npy"
        metadata_path = temp_cache_dir / "test-collection_metadata.json"
        other_path = temp_cache_dir / "other_file.txt"
        
        embeddings_path.touch()
        metadata_path.touch()
        other_path.touch()
        
        manager.clear_cache("test-collection")
        
        assert not embeddings_path.exists()
        assert not metadata_path.exists()
        assert other_path.exists()  # Other files should remain
        
    def test_clear_cache_all(self, config, temp_cache_dir):
        """Test clearing entire cache."""
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        
        # Create test files
        file1 = temp_cache_dir / "file1.npy"
        file2 = temp_cache_dir / "file2.json"
        subdir = temp_cache_dir / "subdir"
        subdir.mkdir()
        
        file1.touch()
        file2.touch()
        
        manager.clear_cache()
        
        assert not file1.exists()
        assert not file2.exists()
        assert subdir.exists()  # Directories should remain
        
    @patch('src.retriever.embeddings.torch')
    @patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('src.retriever.embeddings.SentenceTransformer')
    def test_get_model_info(self, mock_st, mock_torch, config, temp_cache_dir):
        """Test getting model information."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]])  # 3-dimensional
        mock_st.return_value = mock_model
        
        manager = DenseEmbeddingManager(config, temp_cache_dir)
        info = manager.get_model_info()
        
        assert info["model_name"] == config.model_name
        assert info["device"] == "cpu"
        assert info["max_length"] == config.max_length
        assert info["normalize_embeddings"] == config.normalize_embeddings
        assert info["embedding_dim"] == 3
        assert info["sentence_transformers_available"] == SENTENCE_TRANSFORMERS_AVAILABLE
        assert info["transformers_available"] == TRANSFORMERS_AVAILABLE


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""
    
    def test_factory_function(self):
        """Test factory function for creating embedding manager."""
        config = EmbeddingConfig()
        
        manager = create_embedding_manager(config)
        
        assert isinstance(manager, DenseEmbeddingManager)
        assert manager.config == config
        
    def test_device_selection_cuda_available(self, temp_cache_dir):
        """Test device selection when CUDA is available."""
        config = EmbeddingConfig(device="auto")
        
        with patch('src.retriever.embeddings.torch') as mock_torch, \
             patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer'):
            
            mock_torch.cuda.is_available.return_value = True
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            manager._initialize_model()
            
            assert manager._device == "cuda"
            
    def test_device_selection_cuda_not_available(self, temp_cache_dir):
        """Test device selection when CUDA is not available."""
        config = EmbeddingConfig(device="auto")
        
        with patch('src.retriever.embeddings.torch') as mock_torch, \
             patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer'):
            
            mock_torch.cuda.is_available.return_value = False
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            manager._initialize_model()
            
            assert manager._device == "cpu"
            
    def test_fp16_mode_cuda(self, temp_cache_dir):
        """Test FP16 mode when using CUDA."""
        config = EmbeddingConfig(device="cuda", fp16=True)
        
        with patch('src.retriever.embeddings.torch') as mock_torch, \
             patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st:
            
            mock_torch.cuda.is_available.return_value = True
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            manager._initialize_model()
            
            mock_model.half.assert_called_once()
            
    def test_fp16_mode_cpu(self, temp_cache_dir):
        """Test FP16 mode is not applied on CPU."""
        config = EmbeddingConfig(device="cpu", fp16=True)
        
        with patch('src.retriever.embeddings.torch') as mock_torch, \
             patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            manager._initialize_model()
            
            mock_model.half.assert_not_called()
            
    def test_unicode_text_handling(self, temp_cache_dir):
        """Test handling of Unicode text."""
        config = EmbeddingConfig()
        
        with patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st, \
             patch('src.retriever.embeddings.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1.0, 2.0]])
            mock_st.return_value = mock_model
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            
            # Test with various Unicode characters
            texts = ["Hello 世界", "Café résumé", "🤖 AI text", ""]
            result = manager.encode_texts(texts, show_progress=False)
            
            # Should successfully encode without errors
            assert result.shape[0] == len(texts)
            
    def test_large_batch_handling(self, temp_cache_dir):
        """Test handling of large text batches."""
        config = EmbeddingConfig(batch_size=2)
        
        with patch('src.retriever.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('src.retriever.embeddings.SentenceTransformer') as mock_st, \
             patch('src.retriever.embeddings.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            # Return different embeddings for each batch
            mock_model.encode.return_value = np.array([
                [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]
            ])
            mock_st.return_value = mock_model
            
            manager = DenseEmbeddingManager(config, temp_cache_dir)
            
            # Test with 5 texts, batch_size=2
            texts = [f"text {i}" for i in range(5)]
            result = manager.encode_texts(texts, show_progress=False)
            
            assert result.shape == (5, 2)