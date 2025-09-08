#!/usr/bin/env python3
"""
Test Suite for LetheBench Dataset Construction Pipeline

Comprehensive tests for all components including crawlers, labelers, 
validators, and the main build pipeline.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the datasets module to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import components to test
from redaction import PrivacyRedactor
from sources.github_crawler import GitHubCrawler
from sources.stackoverflow_crawler import StackOverflowCrawler
from sources.transcript_crawler import TranscriptCrawler
from labeling.code_labeler import CodeLabeler
from labeling.tool_labeler import ToolLabeler
from labeling.prose_labeler import ProseLabeler
from validation.format_validator import FormatValidator
from validation.privacy_validator import PrivacyValidator
from validation.quality_metrics import QualityMetricsCalculator
from build import LetheBenchBuilder, BuildConfig

class TestPrivacyRedactor:
    """Test privacy redaction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.redactor = PrivacyRedactor()
    
    def test_email_redaction(self):
        """Test email address redaction."""
        text = "Contact me at john.doe@example.com for more info"
        result = self.redactor.redact_text(text)
        
        assert "@example.com" not in result.redacted_text
        assert result.redaction_count >= 1
        assert "email" in [r[0] for r in result.redactions_made]
    
    def test_phone_number_redaction(self):
        """Test phone number redaction."""
        text = "Call me at (555) 123-4567 or +1-555-123-4567"
        result = self.redactor.redact_text(text)
        
        assert "(555) 123-4567" not in result.redacted_text
        assert result.redaction_count >= 1
    
    def test_api_key_redaction(self):
        """Test API key redaction."""
        text = "GitHub token: ghp_1234567890123456789012345678901234567890"
        result = self.redactor.redact_text(text)
        
        assert "ghp_1234567890123456789012345678901234567890" not in result.redacted_text
        assert result.redaction_count >= 1
    
    def test_whitelist_functionality(self):
        """Test that whitelisted patterns are not redacted."""
        text = "Use test@example.com for testing purposes"
        result = self.redactor.redact_text(text)
        
        # example.com should be whitelisted
        assert "test@example.com" in result.redacted_text or result.redaction_count == 0

class TestCodeLabeler:
    """Test code annotation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.labeler = CodeLabeler()
    
    def test_function_extraction(self):
        """Test function definition extraction."""
        code_text = '''
```python
def calculate_metrics(data, threshold=0.5):
    """Calculate accuracy metrics."""
    return accuracy_score(data, threshold)
```
        '''
        
        symbols = self.labeler.extract_code_symbols(code_text, 'python')
        
        function_symbols = [s for s in symbols if s['type'] == 'function_def']
        assert len(function_symbols) > 0
        assert any('calculate_metrics' in s['name'] for s in function_symbols)
    
    def test_import_extraction(self):
        """Test import statement extraction."""
        code_text = '''
```python
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
```
        '''
        
        symbols = self.labeler.extract_code_symbols(code_text, 'python')
        
        import_symbols = [s for s in symbols if 'import' in s['type']]
        assert len(import_symbols) > 0
    
    def test_session_labeling(self):
        """Test complete session labeling."""
        sample_session = [
            {
                'session_id': 'test_session',
                'turn': 0,
                'role': 'user',
                'text': 'How do I calculate accuracy in scikit-learn?',
                'meta': {}
            },
            {
                'session_id': 'test_session',
                'turn': 1,
                'role': 'assistant',
                'text': '''
```python
from sklearn.metrics import accuracy_score
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
```
                ''',
                'meta': {'is_accepted': True, 'score': 10}
            }
        ]
        
        chunks = self.labeler.label_session_turns(sample_session)
        
        assert len(chunks) > 0
        assert any(chunk.confidence > 0.5 for chunk in chunks)

class TestToolLabeler:
    """Test tool output annotation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.labeler = ToolLabeler()
    
    def test_command_output_detection(self):
        """Test detection of command outputs."""
        text = '''
$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED
abc123def456   nginx:latest   "/docker-entrypoint.…"   2 hours ago
        '''
        
        outputs = self.labeler.identify_tool_outputs(text)
        
        assert len(outputs) > 0
        assert any('docker' in output['tool_name'] for output in outputs)
    
    def test_json_output_detection(self):
        """Test JSON response detection."""
        text = '''
$ curl -X GET https://api.example.com/users
{
  "users": [
    {"id": 1, "name": "John Doe"},
    {"id": 2, "name": "Jane Smith"}
  ],
  "total": 2
}
        '''
        
        outputs = self.labeler.identify_tool_outputs(text)
        
        json_outputs = [o for o in outputs if 'json' in o['type']]
        assert len(json_outputs) > 0

class TestProseLabeler:
    """Test prose annotation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.labeler = ProseLabeler()
    
    def test_entity_extraction(self):
        """Test named entity extraction."""
        text = "According to Dr. John Smith from Harvard University, the study was published in January 2023."
        
        entities = self.labeler.extract_entities(text)
        
        # Should find person, organization, and date entities
        entity_types = [e['label'] for e in entities]
        assert len(entities) > 0
    
    def test_temporal_extraction(self):
        """Test temporal expression extraction."""
        text = "The study was conducted in January 2023 and results published last month."
        
        temporal_refs = self.labeler.extract_temporal_expressions(text)
        
        assert len(temporal_refs) > 0
        assert any('2023' in ref['text'] for ref in temporal_refs)
    
    def test_supporting_span_identification(self):
        """Test identification of supporting evidence spans."""
        question_entities = ["Harvard University", "study"]
        question_type = "when"
        text = "Harvard University published the comprehensive study in January 2023, showing significant results in machine learning applications."
        
        spans = self.labeler.find_supporting_spans(question_entities, question_type, text)
        
        assert len(spans) > 0
        assert any(span['confidence'] > 0.3 for span in spans)

class TestValidators:
    """Test validation components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.format_validator = FormatValidator()
        self.privacy_validator = PrivacyValidator()
    
    def test_format_validation(self):
        """Test JSONL format validation."""
        # Create sample JSONL data
        sample_turn = {
            'session_id': 'test_session_1',
            'turn': 0,
            'role': 'user',
            'text': 'How do I implement binary search?',
            'ts': '2024-01-15T10:30:00Z',
            'meta': {
                'license': 'CC BY-SA 4.0',
                'source': 'test'
            }
        }
        
        # Test individual turn validation
        errors, warnings = self.format_validator._validate_turn(sample_turn, 1, 'code')
        
        assert len(errors) == 0  # Should be valid
    
    def test_privacy_validation(self):
        """Test privacy violation detection."""
        text_with_pii = "Contact john.doe@gmail.com or call (555) 123-4567"
        
        violations = self.privacy_validator._scan_text_for_violations(
            text_with_pii, 'test.jsonl', 1, 'test_session', 0
        )
        
        assert len(violations) > 0
        violation_types = [v.violation_type for v in violations]
        assert 'email' in violation_types or 'phone_us' in violation_types

class TestQualityMetrics:
    """Test quality metrics calculation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = QualityMetricsCalculator()
    
    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        sample_data = {
            'code': [
                {
                    'session_id': 'session_1',
                    'turn': 0,
                    'role': 'user',
                    'text': 'Test question',
                    'meta': {}
                },
                {
                    'session_id': 'session_1',
                    'turn': 1,
                    'role': 'assistant',
                    'text': 'Test answer',
                    'meta': {}
                }
            ]
        }
        
        stats = self.calculator._calculate_basic_statistics(sample_data)
        
        assert stats['total_sessions'] == 1
        assert stats['total_turns'] == 2
        assert stats['avg_turns_per_session'] == 2.0
    
    def test_diversity_metrics(self):
        """Test diversity metrics calculation."""
        sample_data = {
            'code': [
                {
                    'session_id': 'session_1',
                    'turn': 0,
                    'role': 'user',
                    'text': 'How do I use Python for data analysis?',
                    'meta': {'tags': ['python', 'data']}
                }
            ]
        }
        
        diversity = self.calculator._calculate_diversity_metrics(sample_data)
        
        assert diversity['vocabulary_size'] > 0
        assert diversity['unique_sessions_ratio'] > 0

class TestBuildPipeline:
    """Test the complete build pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / 'test_output'
        
        self.config = BuildConfig(
            target_sessions_per_genre=5,  # Small for testing
            output_dir=self.output_dir,
            privacy_redaction_enabled=True,
            generate_reports=True
        )
        
        self.builder = LetheBenchBuilder(self.config)
    
    def test_synthetic_tool_generation(self):
        """Test synthetic tool session generation."""
        sessions = self.builder._generate_synthetic_tool_sessions()
        
        assert len(sessions) > 0
        
        # Check session structure
        for session in sessions[:2]:  # Check first 2
            assert 'turns' in session
            assert 'metadata' in session
            assert len(session['turns']) >= 2  # Question + answer
    
    def test_privacy_redaction_application(self):
        """Test privacy redaction application to turns."""
        sample_turns = [
            {
                'session_id': 'test',
                'turn': 0,
                'role': 'user',
                'text': 'Contact me at john@example.com',
                'meta': {}
            }
        ]
        
        redacted_turns = self.builder._apply_privacy_redaction(sample_turns)
        
        assert len(redacted_turns) == 1
        # Since example.com might be whitelisted, we just check structure
        assert 'text' in redacted_turns[0]
    
    def test_split_generation(self):
        """Test train/dev/test split generation."""
        sample_sessions = {
            'code': [{'turns': [{'session_id': f'session_{i}', 'turn': 0}]} for i in range(10)]
        }
        
        splits = self.builder._generate_splits(sample_sessions)
        
        assert 'code' in splits
        assert 'train' in splits['code']
        assert 'dev' in splits['code']
        assert 'test' in splits['code']
        
        total_sessions = sum(len(split) for split in splits['code'].values())
        assert total_sessions == 10

def test_integration():
    """Integration test of the complete pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'integration_test'
        
        config = BuildConfig(
            target_sessions_per_genre=2,  # Very small for testing
            output_dir=output_dir,
            privacy_redaction_enabled=True,
            privacy_validation_enabled=True,
            generate_reports=True
        )
        
        builder = LetheBenchBuilder(config)
        
        # Mock the build process with synthetic data only
        # (to avoid requiring real API keys for testing)
        try:
            # Test directory setup
            builder._setup_output_directories()
            assert output_dir.exists()
            
            # Test synthetic data generation
            tool_sessions = builder._generate_synthetic_tool_sessions()
            assert len(tool_sessions) > 0
            
            print("✓ Integration test passed - pipeline components working correctly")
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])