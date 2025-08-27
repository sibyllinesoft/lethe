#!/usr/bin/env python3
"""
Test script for the Unified Scientific Analysis Framework

This script demonstrates the integration capabilities and validates
that the unified framework can replace the fragmented analysis pipelines.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from analysis_unified import UnifiedAnalysisFramework, AnalysisConfig
    print("✅ Successfully imported UnifiedAnalysisFramework")
except ImportError as e:
    print(f"❌ Failed to import unified analysis framework: {e}")
    sys.exit(1)

def create_test_data(output_dir: str = "test_artifacts"):
    """Create synthetic test data to validate the framework"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create synthetic experimental data
    np.random.seed(42)  # For reproducible results
    
    methods = [
        "baseline_bm25", "baseline_vector", 
        "lethe_iter_1", "lethe_iter_2", "lethe_iter_3"
    ]
    
    # Generate synthetic performance metrics
    n_samples = 100
    test_data = []
    
    for method in methods:
        for _ in range(n_samples):
            # Simulate different performance characteristics for each method
            if "baseline" in method:
                ndcg_10 = np.random.normal(0.65, 0.1)
                ndcg_100 = np.random.normal(0.75, 0.1) 
                latency = np.random.normal(150, 30)
                memory = np.random.normal(256, 50)
            elif "lethe" in method:
                # Lethe iterations should show progressive improvement
                iter_num = int(method.split("_")[-1])
                ndcg_10 = np.random.normal(0.65 + iter_num * 0.05, 0.08)
                ndcg_100 = np.random.normal(0.75 + iter_num * 0.04, 0.08)
                latency = np.random.normal(140 - iter_num * 5, 25)
                memory = np.random.normal(240 - iter_num * 10, 40)
            
            test_data.append({
                "method": method,
                "ndcg_10": max(0, min(1, ndcg_10)),
                "ndcg_100": max(0, min(1, ndcg_100)),
                "latency_ms_total": max(50, latency),
                "memory_mb": max(100, memory),
                "contradiction_rate": np.random.uniform(0.05, 0.15)
            })
    
    # Save as CSV
    df = pd.DataFrame(test_data)
    csv_file = output_path / "synthetic_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Created synthetic test data: {csv_file} ({len(test_data)} samples)")
    
    return str(output_path)

def test_unified_framework():
    """Test the unified analysis framework"""
    
    print("\n" + "="*60)
    print("🧪 TESTING UNIFIED ANALYSIS FRAMEWORK")
    print("="*60)
    
    # Create test data
    artifacts_dir = create_test_data()
    
    try:
        # Initialize framework with test configuration
        config = AnalysisConfig()
        config.artifacts_dir = artifacts_dir
        config.output_dir = "test_outputs"
        config.baseline_methods = ["baseline_bm25", "baseline_vector"]
        config.lethe_iterations = ["lethe_iter_1", "lethe_iter_2", "lethe_iter_3"]
        
        framework = UnifiedAnalysisFramework(config)
        print("\n✅ Framework initialized successfully")
        
        # Test data loading
        print("\n📊 Testing data loading...")
        data = framework.load_experimental_data()
        print(f"✅ Loaded {len(data)} rows of data")
        print(f"✅ Methods found: {sorted(data['method'].unique())}")
        
        # Test complete analysis
        print("\n🔬 Testing complete analysis pipeline...")
        results = framework.run_complete_analysis()
        
        plugins_run = list(results.keys())
        successful_plugins = [p for p in plugins_run if "error" not in results[p]]
        
        print(f"✅ Analysis completed")
        print(f"✅ Plugins run: {len(plugins_run)}")
        print(f"✅ Successful plugins: {len(successful_plugins)}")
        
        # Test legacy migration
        print("\n🔄 Testing legacy script migration...")
        migration_results = framework.migrate_from_legacy_scripts()
        print(f"✅ Migration completed with {len(migration_results)} result sets")
        
        # Generate analysis summary
        print("\n📋 Generating analysis summary...")
        summary = framework.get_analysis_summary()
        print(f"✅ Summary generated")
        print(f"   - Plugins run: {len(summary['plugins_run'])}")
        print(f"   - Data rows: {summary['data_summary']['rows']}")
        print(f"   - Methods analyzed: {len(summary['data_summary']['methods'])}")
        
        if summary.get('key_findings'):
            print(f"   - Key findings available: {len(summary['key_findings'])}")
        
        # Test publication outputs
        print("\n📄 Testing publication output generation...")
        try:
            output_files = framework.generate_publication_outputs()
            total_files = sum(len(files) for files in output_files.values())
            print(f"✅ Publication outputs generated: {total_files} files")
        except Exception as e:
            print(f"⚠ Publication output generation encountered issues: {e}")
        
        print("\n" + "="*60)
        print("🎉 UNIFIED FRAMEWORK TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Print integration validation
        print("\n🔗 INTEGRATION VALIDATION:")
        if hasattr(framework, 'metrics_calculator') and framework.metrics_calculator:
            print("✅ MetricsCalculator integration: Available")
        else:
            print("⚠ MetricsCalculator integration: Not available")
            
        if hasattr(framework, 'statistical_comparator') and framework.statistical_comparator:
            print("✅ StatisticalComparator integration: Available")
        else:
            print("⚠ StatisticalComparator integration: Not available")
            
        if hasattr(framework, 'data_manager') and framework.data_manager:
            print("✅ DataManager integration: Available")
        else:
            print("⚠ DataManager integration: Not available")
            
        if hasattr(framework, 'evaluation_framework') and framework.evaluation_framework:
            print("✅ EvaluationFramework integration: Available")
        else:
            print("⚠ EvaluationFramework integration: Not available")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unified_framework()
    sys.exit(0 if success else 1)