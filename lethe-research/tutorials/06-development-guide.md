# Development Guide: Extending the System

Learn how to extend, customize, and contribute to the Lethe Prompt Monitoring System with custom plugins, metrics, and integrations.

## 🎯 Overview

This guide covers:

- **Architecture Understanding**: Core components and extension points
- **Custom Metrics**: Creating domain-specific quality assessments
- **Plugin Development**: Building modular extensions
- **Integration Patterns**: Connecting with external systems
- **Performance Optimization**: Scaling for high-volume environments
- **Testing & Quality**: Ensuring robustness and reliability

## 🏗️ System Architecture

### Core Components

```python
# System Architecture Overview
src/monitoring/
├── prompt_tracker.py     # Core tracking engine
├── dashboard.py          # Streamlit web interface
├── integration_examples.py # Workflow integration helpers
└── __init__.py          # Public API exports

scripts/
└── prompt_monitor.py    # CLI interface

# Extension Points:
# 1. Custom Quality Assessors
# 2. Storage Backends  
# 3. Export Formats
# 4. Dashboard Components
# 5. CLI Commands
```

### Extension Architecture

```python
# File: examples/extension_architecture_example.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ExtensionMetadata:
    """Metadata for system extensions."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    enabled: bool = True

class BaseExtension(ABC):
    """Base class for all system extensions."""
    
    def __init__(self, metadata: ExtensionMetadata):
        self.metadata = metadata
        self.config = {}
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the extension with configuration."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources when extension is disabled."""
        pass

class QualityAssessorExtension(BaseExtension):
    """Base class for custom quality assessment extensions."""
    
    @abstractmethod
    def assess_quality(self, prompt: str, response: str, context: Dict) -> Dict[str, float]:
        """Assess response quality and return metric scores."""
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Return list of metrics this assessor provides."""
        pass

class StorageExtension(BaseExtension):
    """Base class for custom storage backend extensions."""
    
    @abstractmethod
    def store_execution(self, execution_data: Dict) -> str:
        """Store execution data and return execution ID."""
        pass
    
    @abstractmethod
    def retrieve_execution(self, execution_id: str) -> Dict:
        """Retrieve execution data by ID."""
        pass
    
    @abstractmethod
    def query_executions(self, filters: Dict, limit: int = 100) -> List[Dict]:
        """Query executions with filters."""
        pass

class DashboardExtension(BaseExtension):
    """Base class for custom dashboard components."""
    
    @abstractmethod
    def render_component(self, data: Dict) -> None:
        """Render custom dashboard component using Streamlit."""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Return list of data requirements for this component."""
        pass
```

## 🔧 Custom Quality Metrics

### Creating Domain-Specific Assessors

```python
# File: examples/custom_quality_assessor.py
import re
import json
import nltk
from typing import Dict, List, Any
from textstat import flesch_reading_ease, coleman_liau_index
from src.monitoring import QualityAssessorExtension, ExtensionMetadata

class AcademicQualityAssessor(QualityAssessorExtension):
    """Custom quality assessor for academic content."""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="Academic Quality Assessor",
            version="1.0.0",
            description="Specialized quality assessment for academic content",
            author="Lethe Research Team",
            dependencies=["nltk", "textstat"]
        )
        super().__init__(metadata)
        self.citation_patterns = [
            r'\[[0-9]+\]',                    # [1], [2], etc.
            r'\([^)]*\d{4}[^)]*\)',          # (Author, 2023)
            r'\b\w+\s+et\s+al\.?\s*\(',     # Author et al. (
            r'doi:\s*10\.\d+/\S+',          # DOI references
        ]
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with configuration."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.config = config
            return True
        except Exception as e:
            print(f"Failed to initialize Academic Quality Assessor: {e}")
            return False
    
    def assess_quality(self, prompt: str, response: str, context: Dict) -> Dict[str, float]:
        """Assess academic content quality."""
        scores = {}
        
        # 1. Citation Density and Quality
        scores['citation_quality'] = self._assess_citations(response)
        
        # 2. Technical Vocabulary Usage
        scores['technical_vocabulary'] = self._assess_vocabulary(response, context.get('domain', 'general'))
        
        # 3. Readability and Clarity
        scores['readability'] = self._assess_readability(response)
        
        # 4. Evidence-Based Language
        scores['evidence_language'] = self._assess_evidence_language(response)
        
        # 5. Methodological Soundness
        scores['methodology_discussion'] = self._assess_methodology(response)
        
        # 6. Bias and Objectivity
        scores['objectivity'] = self._assess_objectivity(response)
        
        return scores
    
    def _assess_citations(self, text: str) -> float:
        """Assess citation quality and density."""
        word_count = len(text.split())
        if word_count < 50:
            return 0.7  # Short responses don't need many citations
        
        total_citations = 0
        for pattern in self.citation_patterns:
            total_citations += len(re.findall(pattern, text))
        
        # Expected: ~1 citation per 100 words for academic content
        expected_citations = word_count / 100
        citation_ratio = total_citations / max(expected_citations, 1)
        
        # Score based on ratio with diminishing returns
        score = min(1.0, 0.4 + 0.6 * (1 - pow(0.5, citation_ratio)))
        
        # Bonus for proper formatting
        if re.search(r'doi:\s*10\.\d+/\S+', text):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_vocabulary(self, text: str, domain: str) -> float:
        """Assess technical vocabulary appropriate for domain."""
        
        # Domain-specific technical terms
        domain_vocabularies = {
            'machine_learning': [
                'algorithm', 'neural network', 'supervised', 'unsupervised',
                'training', 'validation', 'overfitting', 'regularization',
                'gradient', 'optimization', 'feature', 'embedding'
            ],
            'retrieval': [
                'vector', 'embedding', 'similarity', 'ranking', 'relevance',
                'indexing', 'query', 'corpus', 'precision', 'recall',
                'tf-idf', 'bm25', 'semantic', 'lexical'
            ],
            'nlp': [
                'tokenization', 'parsing', 'semantics', 'syntax', 'morphology',
                'named entity', 'part-of-speech', 'dependency', 'transformer',
                'attention', 'context', 'disambiguation'
            ],
            'general': [
                'analysis', 'methodology', 'systematic', 'empirical',
                'hypothesis', 'variable', 'correlation', 'significance'
            ]
        }
        
        relevant_terms = domain_vocabularies.get(domain, domain_vocabularies['general'])
        
        text_lower = text.lower()
        technical_term_count = sum(1 for term in relevant_terms if term in text_lower)
        
        # Score based on technical term density
        word_count = len(text.split())
        tech_density = technical_term_count / word_count if word_count > 0 else 0
        
        # Optimal density is 2-5% for academic content
        if 0.02 <= tech_density <= 0.05:
            return 1.0
        elif 0.01 <= tech_density <= 0.07:
            return 0.8
        elif tech_density > 0:
            return 0.6
        else:
            return 0.3
    
    def _assess_readability(self, text: str) -> float:
        """Assess readability appropriate for academic content."""
        try:
            flesch_score = flesch_reading_ease(text)
            coleman_score = coleman_liau_index(text)
            
            # Academic content should be readable but sophisticated
            # Flesch: 30-50 (college level)
            # Coleman-Liau: 12-16 (college/graduate level)
            
            flesch_normalized = 1.0 - abs(flesch_score - 40) / 40
            coleman_normalized = 1.0 - abs(coleman_score - 14) / 6
            
            return (flesch_normalized + coleman_normalized) / 2
            
        except Exception:
            # Fallback: basic sentence length analysis
            sentences = text.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # Optimal: 15-25 words per sentence for academic writing
            if 15 <= avg_sentence_length <= 25:
                return 1.0
            elif 10 <= avg_sentence_length <= 30:
                return 0.8
            else:
                return 0.6
    
    def _assess_evidence_language(self, text: str) -> float:
        """Assess use of evidence-based language."""
        evidence_phrases = [
            'according to', 'research shows', 'studies indicate', 'evidence suggests',
            'findings reveal', 'data demonstrates', 'analysis confirms',
            'empirical evidence', 'experimental results', 'statistical analysis'
        ]
        
        hedge_phrases = [
            'may', 'might', 'could', 'possibly', 'potentially', 'likely',
            'suggests that', 'indicates that', 'appears to', 'seems to'
        ]
        
        text_lower = text.lower()
        
        evidence_count = sum(1 for phrase in evidence_phrases if phrase in text_lower)
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in text_lower)
        
        word_count = len(text.split())
        evidence_density = evidence_count / word_count * 100
        hedge_density = hedge_count / word_count * 100
        
        # Balance evidence language with appropriate hedging
        score = min(1.0, evidence_density * 0.5 + hedge_density * 0.3)
        
        return score
    
    def _assess_methodology(self, text: str) -> float:
        """Assess discussion of methodology and approach."""
        methodology_terms = [
            'methodology', 'method', 'approach', 'procedure', 'protocol',
            'systematic', 'controlled', 'randomized', 'sample size',
            'statistical', 'hypothesis', 'variable', 'measurement',
            'validation', 'reliability', 'reproducible'
        ]
        
        text_lower = text.lower()
        method_mentions = sum(1 for term in methodology_terms if term in text_lower)
        
        # Score based on methodology discussion depth
        if method_mentions >= 5:
            return 1.0
        elif method_mentions >= 3:
            return 0.8
        elif method_mentions >= 1:
            return 0.6
        else:
            return 0.3
    
    def _assess_objectivity(self, text: str) -> float:
        """Assess objectivity and lack of bias."""
        subjective_indicators = [
            'obviously', 'clearly', 'undoubtedly', 'certainly', 'definitely',
            'best', 'worst', 'amazing', 'terrible', 'perfect', 'awful'
        ]
        
        first_person = ['i think', 'i believe', 'in my opinion', 'personally']
        
        text_lower = text.lower()
        
        subjective_count = sum(1 for indicator in subjective_indicators if indicator in text_lower)
        first_person_count = sum(1 for phrase in first_person if phrase in text_lower)
        
        # Penalty for subjective language
        total_bias_indicators = subjective_count + first_person_count * 2
        word_count = len(text.split())
        bias_penalty = min(0.5, total_bias_indicators / word_count * 10)
        
        return max(0.0, 1.0 - bias_penalty)
    
    def get_supported_metrics(self) -> List[str]:
        """Return supported quality metrics."""
        return [
            'citation_quality',
            'technical_vocabulary', 
            'readability',
            'evidence_language',
            'methodology_discussion',
            'objectivity'
        ]
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

# Usage example
def main():
    assessor = AcademicQualityAssessor()
    assessor.initialize({})
    
    prompt = "Analyze the effectiveness of transformer architectures in NLP tasks."
    response = """
    Transformer architectures have demonstrated significant effectiveness in natural language processing tasks according to extensive empirical research [1]. Studies by Vaswani et al. (2017) and subsequent analyses indicate that the self-attention mechanism enables superior performance compared to traditional recurrent neural networks [2,3].
    
    The methodology employed in comparative studies typically involves systematic evaluation on standard benchmarks such as GLUE and SuperGLUE. Statistical analysis reveals that transformer-based models achieve state-of-the-art results across multiple tasks, with effect sizes suggesting practical significance beyond statistical significance.
    
    However, the computational requirements and potential for overfitting in smaller datasets suggest that careful consideration of trade-offs is necessary. The evidence indicates that while transformers show promise, methodological rigor in evaluation remains crucial for reliable conclusions.
    """
    
    context = {'domain': 'nlp'}
    scores = assessor.assess_quality(prompt, response, context)
    
    print("🎯 Academic Quality Assessment:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
    
    overall_score = sum(scores.values()) / len(scores)
    print(f"\n📊 Overall Academic Quality: {overall_score:.3f}")

if __name__ == "__main__":
    main()
```

## 🔌 Plugin System Development

### Plugin Architecture

```python
# File: examples/plugin_system.py
import os
import json
import importlib
from typing import Dict, List, Type, Any
from pathlib import Path
from src.monitoring import BaseExtension, ExtensionMetadata

class PluginManager:
    """Manages loading and lifecycle of system plugins."""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.loaded_plugins = {}
        self.plugin_configs = {}
        
        # Ensure plugin directory exists
        self.plugin_directory.mkdir(exist_ok=True)
    
    def discover_plugins(self) -> List[Path]:
        """Discover available plugin files."""
        plugin_files = []
        
        for file_path in self.plugin_directory.rglob("*.py"):
            if not file_path.name.startswith("_"):
                plugin_files.append(file_path)
        
        return plugin_files
    
    def load_plugin(self, plugin_path: Path) -> bool:
        """Load a single plugin from file."""
        try:
            # Import plugin module
            spec = importlib.util.spec_from_file_location(
                plugin_path.stem, plugin_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseExtension) and 
                    attr != BaseExtension):
                    plugin_classes.append(attr)
            
            # Load plugin configuration if exists
            config_path = plugin_path.with_suffix('.json')
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            
            # Initialize plugins
            for plugin_class in plugin_classes:
                plugin_instance = plugin_class()
                
                if plugin_instance.initialize(config):
                    plugin_name = plugin_instance.metadata.name
                    self.loaded_plugins[plugin_name] = plugin_instance
                    self.plugin_configs[plugin_name] = config
                    print(f"✅ Loaded plugin: {plugin_name}")
                else:
                    print(f"❌ Failed to initialize plugin: {plugin_class.__name__}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading plugin {plugin_path}: {e}")
            return False
    
    def load_all_plugins(self):
        """Load all discovered plugins."""
        plugin_files = self.discover_plugins()
        
        print(f"🔍 Discovered {len(plugin_files)} plugin files")
        
        loaded_count = 0
        for plugin_file in plugin_files:
            if self.load_plugin(plugin_file):
                loaded_count += 1
        
        print(f"🔌 Loaded {loaded_count}/{len(plugin_files)} plugins successfully")
        print(f"📊 Total active plugins: {len(self.loaded_plugins)}")
    
    def get_plugins_by_type(self, plugin_type: Type[BaseExtension]) -> List[BaseExtension]:
        """Get all loaded plugins of a specific type."""
        matching_plugins = []
        
        for plugin in self.loaded_plugins.values():
            if isinstance(plugin, plugin_type):
                matching_plugins.append(plugin)
        
        return matching_plugins
    
    def get_plugin(self, plugin_name: str) -> BaseExtension:
        """Get specific plugin by name."""
        return self.loaded_plugins.get(plugin_name)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        if plugin_name in self.loaded_plugins:
            try:
                self.loaded_plugins[plugin_name].cleanup()
                del self.loaded_plugins[plugin_name]
                del self.plugin_configs[plugin_name]
                print(f"🗑️ Unloaded plugin: {plugin_name}")
                return True
            except Exception as e:
                print(f"❌ Error unloading plugin {plugin_name}: {e}")
                return False
        
        return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin."""
        # Find the plugin file
        for plugin_file in self.discover_plugins():
            if self.unload_plugin(plugin_name):
                return self.load_plugin(plugin_file)
        
        return False
    
    def list_plugins(self) -> Dict[str, Dict]:
        """List all loaded plugins with metadata."""
        plugin_info = {}
        
        for name, plugin in self.loaded_plugins.items():
            plugin_info[name] = {
                'name': plugin.metadata.name,
                'version': plugin.metadata.version,
                'description': plugin.metadata.description,
                'author': plugin.metadata.author,
                'enabled': plugin.metadata.enabled,
                'type': type(plugin).__name__
            }
        
        return plugin_info
    
    def cleanup_all(self):
        """Clean up all loaded plugins."""
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)

# Example usage
def main():
    plugin_manager = PluginManager("examples/plugins")
    
    # Load all plugins
    plugin_manager.load_all_plugins()
    
    # List loaded plugins
    plugins = plugin_manager.list_plugins()
    print("\n🔌 Loaded Plugins:")
    for name, info in plugins.items():
        print(f"  {name} v{info['version']} ({info['type']})")
        print(f"    {info['description']}")
    
    # Get quality assessor plugins
    from examples.extension_architecture_example import QualityAssessorExtension
    quality_plugins = plugin_manager.get_plugins_by_type(QualityAssessorExtension)
    
    print(f"\n🎯 Quality Assessor Plugins: {len(quality_plugins)}")
    for plugin in quality_plugins:
        metrics = plugin.get_supported_metrics()
        print(f"  {plugin.metadata.name}: {', '.join(metrics)}")

if __name__ == "__main__":
    main()
```

### Example Plugin: Time-Series Analyzer

```python
# File: examples/plugins/time_series_analyzer.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from scipy import stats
from src.monitoring import BaseExtension, ExtensionMetadata

class TimeSeriesAnalyzer(BaseExtension):
    """Plugin for advanced time-series analysis of prompt performance."""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="Time Series Analyzer",
            version="1.2.0",
            description="Advanced time-series analysis for prompt performance patterns",
            author="Lethe Analytics Team",
            dependencies=["numpy", "pandas", "scipy"]
        )
        super().__init__(metadata)
        self.analysis_cache = {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the time series analyzer."""
        self.config = {
            'cache_duration': config.get('cache_duration', 3600),  # 1 hour
            'min_data_points': config.get('min_data_points', 10),
            'seasonality_detection': config.get('seasonality_detection', True),
            'anomaly_detection': config.get('anomaly_detection', True)
        }
        return True
    
    def analyze_performance_trends(self, execution_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends in execution data."""
        
        if len(execution_data) < self.config['min_data_points']:
            return {'error': 'Insufficient data points for analysis'}
        
        # Convert to DataFrame
        df = pd.DataFrame(execution_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Resample to regular intervals
        df_resampled = self._resample_data(df)
        
        analysis_results = {
            'trend_analysis': self._analyze_trends(df_resampled),
            'seasonality_analysis': self._analyze_seasonality(df_resampled),
            'anomaly_detection': self._detect_anomalies(df_resampled),
            'forecasting': self._generate_forecast(df_resampled),
            'correlation_analysis': self._analyze_correlations(df_resampled)
        }
        
        return analysis_results
    
    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to regular time intervals."""
        
        # Determine appropriate resampling frequency
        time_span = df['timestamp'].max() - df['timestamp'].min()
        
        if time_span > timedelta(days=30):
            freq = 'D'  # Daily
        elif time_span > timedelta(days=7):
            freq = 'H'  # Hourly
        else:
            freq = '15T'  # 15 minutes
        
        # Set timestamp as index and resample
        df_indexed = df.set_index('timestamp')
        
        # Aggregate numeric columns
        numeric_columns = df_indexed.select_dtypes(include=[np.number]).columns
        
        resampled = df_indexed[numeric_columns].resample(freq).agg({
            'execution_time_ms': 'mean',
            'response_quality_score': 'mean',
            'tokens_used': 'mean',
            'success': 'mean'  # Success rate
        }).dropna()
        
        return resampled
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term trends in the data."""
        
        trends = {}
        
        for column in df.columns:
            if len(df[column]) < 3:
                continue
            
            # Linear trend analysis
            x = np.arange(len(df))
            y = df[column].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Categorize trend
            if p_value < 0.05:
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            trends[column] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': trend_direction,
                'strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak'
            }
        
        return trends
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        
        if not self.config['seasonality_detection'] or len(df) < 24:
            return {'seasonal_patterns': 'insufficient_data'}
        
        seasonality = {}
        
        for column in df.columns:
            # Hour of day pattern
            df_with_hour = df.copy()
            df_with_hour['hour'] = df_with_hour.index.hour
            
            hourly_means = df_with_hour.groupby('hour')[column].mean()
            hourly_std = hourly_means.std()
            hourly_variation = hourly_std / hourly_means.mean() if hourly_means.mean() > 0 else 0
            
            # Day of week pattern (if enough data)
            daily_variation = 0
            if len(df) >= 7:
                df_with_dow = df.copy()
                df_with_dow['day_of_week'] = df_with_dow.index.dayofweek
                
                daily_means = df_with_dow.groupby('day_of_week')[column].mean()
                daily_std = daily_means.std()
                daily_variation = daily_std / daily_means.mean() if daily_means.mean() > 0 else 0
            
            seasonality[column] = {
                'hourly_variation': hourly_variation,
                'daily_variation': daily_variation,
                'peak_hour': hourly_means.idxmax(),
                'low_hour': hourly_means.idxmin(),
                'has_strong_hourly_pattern': hourly_variation > 0.2,
                'has_strong_daily_pattern': daily_variation > 0.1
            }
        
        return seasonality
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        
        if not self.config['anomaly_detection']:
            return {'anomalies': 'disabled'}
        
        anomalies = {}
        
        for column in df.columns:
            data = df[column].values
            
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(data))
            z_anomalies = df.index[z_scores > 3].tolist()
            
            # Interquartile range (IQR) based detection
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_anomalies = df.index[(data < lower_bound) | (data > upper_bound)].tolist()
            
            # Modified Z-score using median absolute deviation
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
            mad_anomalies = df.index[np.abs(modified_z_scores) > 3.5].tolist()
            
            anomalies[column] = {
                'z_score_anomalies': [str(ts) for ts in z_anomalies],
                'iqr_anomalies': [str(ts) for ts in iqr_anomalies],
                'mad_anomalies': [str(ts) for ts in mad_anomalies],
                'anomaly_rate': len(set(z_anomalies + iqr_anomalies + mad_anomalies)) / len(data)
            }
        
        return anomalies
    
    def _generate_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate simple forecasts using linear extrapolation."""
        
        forecasts = {}
        forecast_periods = 24  # Forecast next 24 periods
        
        for column in df.columns:
            if len(df[column]) < 5:
                continue
            
            # Simple linear extrapolation
            x = np.arange(len(df))
            y = df[column].values
            
            # Fit linear model
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Generate forecast
            future_x = np.arange(len(df), len(df) + forecast_periods)
            forecast_values = slope * future_x + intercept
            
            # Calculate confidence intervals (rough approximation)
            forecast_std = std_err * np.sqrt(1 + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            
            forecasts[column] = {
                'forecast_values': forecast_values.tolist(),
                'confidence_intervals': {
                    'lower': (forecast_values - 1.96 * forecast_std).tolist(),
                    'upper': (forecast_values + 1.96 * forecast_std).tolist()
                },
                'model_quality': r_value**2,
                'forecast_trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            }
        
        return forecasts
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        
        correlation_matrix = df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong positive' if corr_value > 0 else 'strong negative'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'insights': self._generate_correlation_insights(strong_correlations)
        }
    
    def _generate_correlation_insights(self, correlations: List[Dict]) -> List[str]:
        """Generate human-readable insights from correlations."""
        
        insights = []
        
        for corr in correlations:
            metric1, metric2 = corr['metric1'], corr['metric2']
            strength = corr['strength']
            value = corr['correlation']
            
            if 'quality' in metric1.lower() and 'time' in metric2.lower():
                if value < 0:
                    insights.append(f"Faster responses tend to have higher quality scores (r={value:.3f})")
                else:
                    insights.append(f"Slower responses are associated with higher quality (r={value:.3f})")
            
            elif 'success' in metric1.lower():
                if value > 0:
                    insights.append(f"Higher {metric2} is associated with better success rates (r={value:.3f})")
                else:
                    insights.append(f"Higher {metric2} is associated with more failures (r={value:.3f})")
        
        return insights
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.analysis_cache.clear()

# Plugin configuration file (time_series_analyzer.json)
PLUGIN_CONFIG = {
    "cache_duration": 3600,
    "min_data_points": 20,
    "seasonality_detection": True,
    "anomaly_detection": True
}
```

## 🔗 Custom Storage Backends

### Database Integration Example

```python
# File: examples/custom_storage.py
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.monitoring import StorageExtension, ExtensionMetadata

class PostgreSQLStorage(StorageExtension):
    """PostgreSQL storage backend for high-volume production use."""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="PostgreSQL Storage Backend",
            version="1.0.0", 
            description="Production-ready PostgreSQL storage for prompt executions",
            author="Lethe DevOps Team",
            dependencies=["asyncpg", "psycopg2-binary"]
        )
        super().__init__(metadata)
        self.connection_pool = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize PostgreSQL connection."""
        try:
            import asyncpg
            
            self.db_config = {
                'host': config.get('host', 'localhost'),
                'port': config.get('port', 5432),
                'database': config.get('database', 'prompt_monitoring'),
                'user': config.get('user', 'postgres'),
                'password': config.get('password', ''),
                'min_connections': config.get('min_connections', 5),
                'max_connections': config.get('max_connections', 20)
            }
            
            # Initialize connection pool
            asyncio.run(self._create_connection_pool())
            
            # Create tables if they don't exist
            asyncio.run(self._create_tables())
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize PostgreSQL storage: {e}")
            return False
    
    async def _create_connection_pool(self):
        """Create async connection pool."""
        import asyncpg
        
        self.connection_pool = await asyncpg.create_pool(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            min_size=self.db_config['min_connections'],
            max_size=self.db_config['max_connections']
        )
    
    async def _create_tables(self):
        """Create database tables."""
        
        create_executions_table = """
        CREATE TABLE IF NOT EXISTS prompt_executions (
            execution_id VARCHAR(255) PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            prompt_id VARCHAR(255) NOT NULL,
            prompt_text TEXT,
            response_text TEXT,
            model_config JSONB,
            execution_time_ms INTEGER,
            response_quality_score REAL,
            tokens_used INTEGER,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            tags TEXT[],
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_prompt_executions_timestamp 
            ON prompt_executions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_prompt_executions_prompt_id 
            ON prompt_executions(prompt_id);
        CREATE INDEX IF NOT EXISTS idx_prompt_executions_success 
            ON prompt_executions(success);
        CREATE INDEX IF NOT EXISTS idx_prompt_executions_tags 
            ON prompt_executions USING GIN(tags);
        CREATE INDEX IF NOT EXISTS idx_prompt_executions_metadata 
            ON prompt_executions USING GIN(metadata);
        """
        
        async with self.connection_pool.acquire() as connection:
            await connection.execute(create_executions_table)
    
    def store_execution(self, execution_data: Dict) -> str:
        """Store execution data in PostgreSQL."""
        return asyncio.run(self._async_store_execution(execution_data))
    
    async def _async_store_execution(self, execution_data: Dict) -> str:
        """Async store execution data."""
        
        query = """
        INSERT INTO prompt_executions (
            execution_id, timestamp, prompt_id, prompt_text, response_text,
            model_config, execution_time_ms, response_quality_score, tokens_used,
            success, error_message, tags, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (execution_id) DO UPDATE SET
            response_text = EXCLUDED.response_text,
            execution_time_ms = EXCLUDED.execution_time_ms,
            response_quality_score = EXCLUDED.response_quality_score,
            tokens_used = EXCLUDED.tokens_used,
            success = EXCLUDED.success,
            error_message = EXCLUDED.error_message,
            updated_at = NOW()
        RETURNING execution_id;
        """
        
        async with self.connection_pool.acquire() as connection:
            execution_id = await connection.fetchval(
                query,
                execution_data.get('execution_id'),
                execution_data.get('timestamp'),
                execution_data.get('prompt_id'),
                execution_data.get('prompt_text'),
                execution_data.get('response_text'),
                json.dumps(execution_data.get('model_config', {})),
                execution_data.get('execution_time_ms'),
                execution_data.get('response_quality_score'),
                execution_data.get('tokens_used'),
                execution_data.get('success', True),
                execution_data.get('error_message'),
                execution_data.get('tags', []),
                json.dumps(execution_data.get('metadata', {}))
            )
        
        return execution_id
    
    def retrieve_execution(self, execution_id: str) -> Dict:
        """Retrieve execution by ID."""
        return asyncio.run(self._async_retrieve_execution(execution_id))
    
    async def _async_retrieve_execution(self, execution_id: str) -> Dict:
        """Async retrieve execution."""
        
        query = """
        SELECT * FROM prompt_executions WHERE execution_id = $1;
        """
        
        async with self.connection_pool.acquire() as connection:
            row = await connection.fetchrow(query, execution_id)
            
            if row:
                return dict(row)
            else:
                return {}
    
    def query_executions(self, filters: Dict, limit: int = 100) -> List[Dict]:
        """Query executions with filters."""
        return asyncio.run(self._async_query_executions(filters, limit))
    
    async def _async_query_executions(self, filters: Dict, limit: int) -> List[Dict]:
        """Async query executions."""
        
        where_clauses = []
        params = []
        param_counter = 1
        
        # Build WHERE clause from filters
        if 'prompt_id' in filters:
            where_clauses.append(f"prompt_id = ${param_counter}")
            params.append(filters['prompt_id'])
            param_counter += 1
        
        if 'success' in filters:
            where_clauses.append(f"success = ${param_counter}")
            params.append(filters['success'])
            param_counter += 1
        
        if 'start_date' in filters:
            where_clauses.append(f"timestamp >= ${param_counter}")
            params.append(filters['start_date'])
            param_counter += 1
        
        if 'end_date' in filters:
            where_clauses.append(f"timestamp <= ${param_counter}")
            params.append(filters['end_date'])
            param_counter += 1
        
        if 'tags' in filters and filters['tags']:
            where_clauses.append(f"tags && ${param_counter}")
            params.append(filters['tags'])
            param_counter += 1
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        SELECT * FROM prompt_executions 
        {where_clause}
        ORDER BY timestamp DESC 
        LIMIT ${param_counter};
        """
        params.append(limit)
        
        async with self.connection_pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]
    
    def cleanup(self) -> None:
        """Clean up connection pool."""
        if self.connection_pool:
            asyncio.run(self.connection_pool.close())

class ClickHouseStorage(StorageExtension):
    """ClickHouse storage for high-volume analytics."""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="ClickHouse Analytics Storage",
            version="1.0.0",
            description="High-performance ClickHouse storage for analytics workloads",
            author="Lethe Analytics Team", 
            dependencies=["clickhouse-driver"]
        )
        super().__init__(metadata)
        self.client = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize ClickHouse connection."""
        try:
            from clickhouse_driver import Client
            
            self.client = Client(
                host=config.get('host', 'localhost'),
                port=config.get('port', 9000),
                database=config.get('database', 'prompt_analytics'),
                user=config.get('user', 'default'),
                password=config.get('password', '')
            )
            
            # Create table
            self._create_table()
            return True
            
        except Exception as e:
            print(f"Failed to initialize ClickHouse storage: {e}")
            return False
    
    def _create_table(self):
        """Create ClickHouse table optimized for analytics."""
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS prompt_executions (
            execution_id String,
            timestamp DateTime64(3),
            prompt_id String,
            model_name String,
            execution_time_ms UInt32,
            response_quality_score Float32,
            tokens_used UInt32,
            success UInt8,
            tags Array(String),
            metadata String,
            date Date MATERIALIZED toDate(timestamp)
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (prompt_id, timestamp)
        SETTINGS index_granularity = 8192;
        """
        
        self.client.execute(create_table_query)
    
    def store_execution(self, execution_data: Dict) -> str:
        """Store execution optimized for analytics."""
        
        # Extract model name from config for fast queries
        model_config = execution_data.get('model_config', {})
        model_name = model_config.get('model', 'unknown')
        
        insert_data = [(
            execution_data.get('execution_id', ''),
            execution_data.get('timestamp', datetime.now()),
            execution_data.get('prompt_id', ''),
            model_name,
            execution_data.get('execution_time_ms', 0),
            execution_data.get('response_quality_score', 0.0),
            execution_data.get('tokens_used', 0),
            1 if execution_data.get('success', True) else 0,
            execution_data.get('tags', []),
            json.dumps(execution_data.get('metadata', {}))
        )]
        
        self.client.execute(
            'INSERT INTO prompt_executions VALUES',
            insert_data
        )
        
        return execution_data.get('execution_id', '')
    
    def retrieve_execution(self, execution_id: str) -> Dict:
        """Retrieve execution (not optimized for ClickHouse)."""
        
        result = self.client.execute(
            'SELECT * FROM prompt_executions WHERE execution_id = %s LIMIT 1',
            [execution_id]
        )
        
        if result:
            row = result[0]
            return {
                'execution_id': row[0],
                'timestamp': row[1],
                'prompt_id': row[2],
                'model_name': row[3],
                'execution_time_ms': row[4],
                'response_quality_score': row[5],
                'tokens_used': row[6],
                'success': bool(row[7]),
                'tags': row[8],
                'metadata': json.loads(row[9]) if row[9] else {}
            }
        
        return {}
    
    def query_executions(self, filters: Dict, limit: int = 100) -> List[Dict]:
        """Query executions optimized for analytics."""
        
        where_conditions = ['1=1']  # Base condition
        
        if 'prompt_id' in filters:
            where_conditions.append(f"prompt_id = '{filters['prompt_id']}'")
        
        if 'model_name' in filters:
            where_conditions.append(f"model_name = '{filters['model_name']}'")
        
        if 'start_date' in filters:
            where_conditions.append(f"timestamp >= '{filters['start_date']}'")
        
        if 'end_date' in filters:
            where_conditions.append(f"timestamp <= '{filters['end_date']}'")
        
        query = f"""
        SELECT * FROM prompt_executions 
        WHERE {' AND '.join(where_conditions)}
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """
        
        results = self.client.execute(query)
        
        executions = []
        for row in results:
            executions.append({
                'execution_id': row[0],
                'timestamp': row[1],
                'prompt_id': row[2],
                'model_name': row[3],
                'execution_time_ms': row[4],
                'response_quality_score': row[5],
                'tokens_used': row[6],
                'success': bool(row[7]),
                'tags': row[8],
                'metadata': json.loads(row[9]) if row[9] else {}
            })
        
        return executions
    
    def get_analytics_data(self, filters: Dict = None) -> Dict:
        """Get aggregated analytics data (ClickHouse optimized)."""
        
        base_where = "1=1"
        if filters:
            if 'start_date' in filters:
                base_where += f" AND timestamp >= '{filters['start_date']}'"
            if 'end_date' in filters:
                base_where += f" AND timestamp <= '{filters['end_date']}'"
        
        analytics_query = f"""
        SELECT 
            count() as total_executions,
            avg(success) as success_rate,
            avg(execution_time_ms) as avg_execution_time,
            quantile(0.5)(execution_time_ms) as median_execution_time,
            quantile(0.95)(execution_time_ms) as p95_execution_time,
            avg(response_quality_score) as avg_quality_score,
            sum(tokens_used) as total_tokens
        FROM prompt_executions 
        WHERE {base_where}
        """
        
        result = self.client.execute(analytics_query)[0]
        
        return {
            'total_executions': result[0],
            'success_rate': result[1],
            'avg_execution_time': result[2],
            'median_execution_time': result[3],
            'p95_execution_time': result[4],
            'avg_quality_score': result[5],
            'total_tokens': result[6]
        }
    
    def cleanup(self) -> None:
        """Clean up ClickHouse connection."""
        if self.client:
            self.client.disconnect()
```

## 🧪 Testing Framework

### Plugin Testing Infrastructure

```python
# File: examples/testing_framework.py
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from src.monitoring import BaseExtension, ExtensionMetadata

class MockExtension(BaseExtension):
    """Mock extension for testing."""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="Mock Extension",
            version="1.0.0", 
            description="Mock extension for testing",
            author="Test Suite"
        )
        super().__init__(metadata)
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        self.initialized = True
        return True
    
    def cleanup(self) -> None:
        self.initialized = False

class TestExtensionFramework(unittest.TestCase):
    """Test suite for extension framework."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.extension = MockExtension()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        self.extension.cleanup()
    
    def test_extension_initialization(self):
        """Test extension initialization."""
        config = {'test_param': 'test_value'}
        
        result = self.extension.initialize(config)
        
        self.assertTrue(result)
        self.assertTrue(self.extension.initialized)
        self.assertEqual(self.extension.config, config)
    
    def test_extension_metadata(self):
        """Test extension metadata."""
        metadata = self.extension.metadata
        
        self.assertEqual(metadata.name, "Mock Extension")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertTrue(metadata.enabled)
    
    def test_extension_cleanup(self):
        """Test extension cleanup."""
        self.extension.initialize({})
        self.assertTrue(self.extension.initialized)
        
        self.extension.cleanup()
        self.assertFalse(self.extension.initialized)

class TestQualityAssessor(unittest.TestCase):
    """Test suite for quality assessor extensions."""
    
    def setUp(self):
        """Set up test environment."""
        from examples.custom_quality_assessor import AcademicQualityAssessor
        self.assessor = AcademicQualityAssessor()
        self.assessor.initialize({})
    
    def test_citation_assessment(self):
        """Test citation quality assessment."""
        text_with_citations = """
        According to recent research [1], machine learning has shown significant promise.
        Studies by Smith et al. (2023) demonstrate the effectiveness of this approach [2,3].
        The methodology follows established protocols (Johnson, 2022).
        """
        
        text_without_citations = """
        Machine learning is very promising. Many people think it works well.
        The approach seems effective based on general knowledge.
        """
        
        score_with = self.assessor._assess_citations(text_with_citations)
        score_without = self.assessor._assess_citations(text_without_citations)
        
        self.assertGreater(score_with, score_without)
        self.assertGreater(score_with, 0.7)  # Should be high with citations
        self.assertLess(score_without, 0.7)  # Should be lower without citations
    
    def test_technical_vocabulary(self):
        """Test technical vocabulary assessment."""
        technical_text = """
        The neural network architecture employs transformer layers with self-attention mechanisms.
        The training process uses gradient descent optimization with regularization techniques
        to prevent overfitting on the validation dataset.
        """
        
        general_text = """
        The system works very well and produces good results.
        It learns from data and makes predictions accurately.
        """
        
        tech_score = self.assessor._assess_vocabulary(technical_text, 'machine_learning')
        general_score = self.assessor._assess_vocabulary(general_text, 'machine_learning')
        
        self.assertGreater(tech_score, general_score)
    
    def test_full_assessment(self):
        """Test complete quality assessment."""
        prompt = "Analyze the effectiveness of deep learning in natural language processing."
        
        high_quality_response = """
        Deep learning has demonstrated remarkable effectiveness in natural language processing tasks,
        as evidenced by extensive empirical research [1,2]. According to Vaswani et al. (2017),
        the transformer architecture represents a significant advancement in sequence modeling [3].
        
        The methodology employed in comparative studies typically involves systematic evaluation
        on standardized benchmarks such as GLUE and SuperGLUE. Statistical analysis reveals
        that transformer-based models achieve state-of-the-art performance across multiple tasks,
        with effect sizes indicating both statistical and practical significance.
        
        However, the computational requirements and potential for overfitting in limited data
        scenarios suggest that careful consideration of trade-offs remains necessary.
        """
        
        context = {'domain': 'nlp'}
        scores = self.assessor.assess_quality(prompt, high_quality_response, context)
        
        # Check that all expected metrics are present
        expected_metrics = self.assessor.get_supported_metrics()
        for metric in expected_metrics:
            self.assertIn(metric, scores)
            self.assertIsInstance(scores[metric], float)
            self.assertGreaterEqual(scores[metric], 0.0)
            self.assertLessEqual(scores[metric], 1.0)
        
        # High-quality response should score well overall
        avg_score = sum(scores.values()) / len(scores)
        self.assertGreater(avg_score, 0.6)

class TestPluginManager(unittest.TestCase):
    """Test suite for plugin management."""
    
    def setUp(self):
        """Set up test environment."""
        from examples.plugin_system import PluginManager
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plugin_manager = PluginManager(str(self.temp_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        self.plugin_manager.cleanup_all()
        shutil.rmtree(self.temp_dir)
    
    def create_mock_plugin(self, plugin_name: str):
        """Create a mock plugin file for testing."""
        
        plugin_content = f'''
from examples.extension_architecture_example import BaseExtension, ExtensionMetadata
from typing import Dict, Any

class {plugin_name}(BaseExtension):
    def __init__(self):
        metadata = ExtensionMetadata(
            name="{plugin_name}",
            version="1.0.0",
            description="Test plugin {plugin_name}",
            author="Test Suite"
        )
        super().__init__(metadata)
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
'''
        
        plugin_file = self.temp_dir / f"{plugin_name.lower()}.py"
        plugin_file.write_text(plugin_content)
        return plugin_file
    
    def test_plugin_discovery(self):
        """Test plugin discovery."""
        # Create mock plugins
        self.create_mock_plugin("TestPlugin1")
        self.create_mock_plugin("TestPlugin2")
        
        discovered = self.plugin_manager.discover_plugins()
        self.assertEqual(len(discovered), 2)
    
    def test_plugin_loading(self):
        """Test plugin loading."""
        plugin_file = self.create_mock_plugin("TestPlugin")
        
        result = self.plugin_manager.load_plugin(plugin_file)
        self.assertTrue(result)
        
        plugins = self.plugin_manager.list_plugins()
        self.assertIn("TestPlugin", plugins)
    
    def test_plugin_unloading(self):
        """Test plugin unloading."""
        plugin_file = self.create_mock_plugin("TestPlugin")
        self.plugin_manager.load_plugin(plugin_file)
        
        result = self.plugin_manager.unload_plugin("TestPlugin")
        self.assertTrue(result)
        
        plugins = self.plugin_manager.list_plugins()
        self.assertNotIn("TestPlugin", plugins)

def run_tests():
    """Run all tests."""
    
    print("🧪 Running Extension Framework Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestExtensionFramework))
    suite.addTest(unittest.makeSuite(TestQualityAssessor))
    suite.addTest(unittest.makeSuite(TestPluginManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun:.1%}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
```

## 🚀 Deployment & Production

### Docker Integration

```dockerfile
# File: examples/Dockerfile.monitoring
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p experiments logs backups plugins

# Set environment variables
ENV PYTHONPATH=/app
ENV PROMPT_MONITOR_DB_PATH=/app/experiments/prompt_tracking.db

# Expose dashboard port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/prompt_monitor.py health || exit 1

# Default command
CMD ["python", "scripts/prompt_monitor.py", "dashboard", "--host", "0.0.0.0"]
```

### Kubernetes Deployment

```yaml
# File: examples/k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prompt-monitoring
  labels:
    app: prompt-monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: prompt-monitoring
  template:
    metadata:
      labels:
        app: prompt-monitoring
    spec:
      containers:
      - name: prompt-monitoring
        image: prompt-monitoring:latest
        ports:
        - containerPort: 8501
        env:
        - name: PROMPT_MONITOR_DB_PATH
          value: "/data/prompt_tracking.db"
        - name: POSTGRES_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        volumeMounts:
        - name: data-volume
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: monitoring-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: prompt-monitoring-service
spec:
  selector:
    app: prompt-monitoring
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

## 💡 Best Practices & Tips

### Performance Optimization

```python
# File: examples/performance_optimization.py

# 1. Use connection pooling for database operations
import asyncio
from contextlib import asynccontextmanager

class PerformantTracker:
    def __init__(self):
        self.connection_pool = None
        self.cache = {}
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Managed database connection from pool."""
        if not self.connection_pool:
            await self._initialize_pool()
        
        conn = await self.connection_pool.acquire()
        try:
            yield conn
        finally:
            await self.connection_pool.release(conn)

# 2. Implement caching for expensive operations
from functools import lru_cache
from typing import Tuple

class CachedAnalytics:
    @lru_cache(maxsize=100)
    def get_analytics(self, filters_hash: str) -> Dict:
        """Cached analytics with filter hash."""
        # Expensive analytics computation
        pass
    
    def cache_key(self, filters: Dict) -> str:
        """Generate cache key from filters."""
        import hashlib
        return hashlib.md5(str(sorted(filters.items())).encode()).hexdigest()

# 3. Use batch processing for high-volume scenarios
class BatchProcessor:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.pending_executions = []
    
    def add_execution(self, execution_data: Dict):
        """Add execution to batch."""
        self.pending_executions.append(execution_data)
        
        if len(self.pending_executions) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Process accumulated batch."""
        if self.pending_executions:
            self._process_batch(self.pending_executions)
            self.pending_executions.clear()
```

### Security Considerations

```python
# File: examples/security_practices.py

# 1. Data sanitization
import re
from typing import Dict, Any

def sanitize_prompt_data(execution_data: Dict) -> Dict:
    """Sanitize sensitive data from prompt executions."""
    
    sanitized = execution_data.copy()
    
    # Remove potential PII patterns
    if 'prompt_text' in sanitized:
        sanitized['prompt_text'] = sanitize_text(sanitized['prompt_text'])
    
    if 'response_text' in sanitized:
        sanitized['response_text'] = sanitize_text(sanitized['response_text'])
    
    # Remove sensitive config keys
    if 'model_config' in sanitized:
        config = sanitized['model_config'].copy()
        sensitive_keys = ['api_key', 'token', 'secret', 'password']
        for key in sensitive_keys:
            if key in config:
                config[key] = '[REDACTED]'
        sanitized['model_config'] = config
    
    return sanitized

def sanitize_text(text: str) -> str:
    """Remove PII patterns from text."""
    
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    
    # SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Credit card patterns
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
    
    return text

# 2. Access control
class SecureMonitoringSystem:
    def __init__(self):
        self.permissions = {}
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.has_permission(permission):
                    raise PermissionError(f"Permission required: {permission}")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def has_permission(self, permission: str) -> bool:
        """Check if current context has permission."""
        # Implement your permission checking logic
        return True  # Placeholder
```

---

**🎉 Development mastery achieved!** You now have the knowledge and tools to extend the Lethe Prompt Monitoring System with custom functionality, optimize it for your specific needs, and deploy it in production environments.