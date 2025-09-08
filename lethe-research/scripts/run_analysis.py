#!/usr/bin/env python3
"""
Comprehensive Analysis Execution Script
=======================================

Orchestrates complete statistical analysis including:
- Hypothesis testing framework (H1-H4)
- Statistical significance testing with bootstrap confidence intervals
- Effect size calculations (Cliff's delta)
- Multiple comparison corrections (Holm-Bonferroni)
- Fraud-proofing validation

Usage:
    python run_analysis.py --baseline-results baselines/ --lethe-results lethe_runs/ --output analysis/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml
from datetime import datetime

# Add research directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiments.score import StatisticalAnalyzer
from experiments.fraud_proof import FraudProofingFramework

def setup_logging(level: str) -> logging.Logger:
    """Configure logging with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class ComprehensiveAnalyzer:
    """Orchestrates complete research analysis pipeline."""
    
    def __init__(self, baseline_results_dir: Path, lethe_results_dir: Path,
                 output_dir: Path, config_path: Path, hypothesis_framework_path: Path,
                 logger: logging.Logger):
        self.baseline_results_dir = baseline_results_dir
        self.lethe_results_dir = lethe_results_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.hypothesis_framework_path = hypothesis_framework_path
        self.logger = logger
        
        # Load configurations
        self.config = self._load_config()
        self.hypothesis_framework = self._load_hypothesis_framework()
        
        # Initialize analyzers
        self.statistical_analyzer = StatisticalAnalyzer()
        self.fraud_framework = FraudProofingFramework()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load analysis configuration."""
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return {}
            
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _load_hypothesis_framework(self) -> Dict[str, Any]:
        """Load hypothesis testing framework."""
        if not self.hypothesis_framework_path.exists():
            # Create default hypothesis framework
            default_framework = {
                "hypotheses": {
                    "H1_Quality": {
                        "description": "Lethe achieves superior retrieval quality",
                        "metrics": ["ndcg_at_10", "recall_at_10", "mrr_at_10"],
                        "direction": "greater",
                        "effect_size_threshold": 0.3,
                        "statistical_power": 0.8
                    },
                    "H2_Efficiency": {
                        "description": "Lethe maintains acceptable efficiency",
                        "metrics": ["latency_p95", "memory_peak"],
                        "direction": "less",
                        "thresholds": {"latency_p95": 3000, "memory_peak": 1500},
                        "statistical_power": 0.8
                    },
                    "H3_Robustness": {
                        "description": "Lethe demonstrates robustness across domains",
                        "metrics": ["coverage_at_10", "coverage_at_20"],
                        "direction": "greater",
                        "min_domains": 3,
                        "consistency_threshold": 0.7
                    },
                    "H4_Adaptivity": {
                        "description": "Lethe adapts effectively to different contexts",
                        "metrics": ["consistency_score", "contradiction_rate"],
                        "adaptivity_measure": "parameter_sensitivity",
                        "threshold": 0.15
                    }
                }
            }
            
            # Save default framework
            self.hypothesis_framework_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.hypothesis_framework_path, 'w') as f:
                json.dump(default_framework, f, indent=2)
            
            return default_framework
        
        with open(self.hypothesis_framework_path) as f:
            return json.load(f)
    
    def load_experimental_data(self) -> Dict[str, Any]:
        """Load all experimental results."""
        self.logger.info("Loading experimental data...")
        
        # Load baseline results
        baseline_data = self._load_results_directory(self.baseline_results_dir, "baseline")
        
        # Load Lethe results  
        lethe_data = self._load_results_directory(self.lethe_results_dir, "lethe")
        
        # Combine data
        experimental_data = {
            'baseline_results': baseline_data,
            'lethe_results': lethe_data,
            'metadata': {
                'baseline_configs': len(baseline_data),
                'lethe_configs': len(lethe_data),
                'total_configs': len(baseline_data) + len(lethe_data),
                'loaded_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        self.logger.info(f"Loaded {len(baseline_data)} baseline and {len(lethe_data)} lethe configurations")
        
        return experimental_data
    
    def _load_results_directory(self, results_dir: Path, config_type: str) -> Dict[str, Any]:
        """Load results from directory structure."""
        results = {}
        
        if not results_dir.exists():
            self.logger.warning(f"Results directory not found: {results_dir}")
            return results
        
        # Look for result files (JSON format expected)
        for result_file in results_dir.rglob('*.json'):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                # Extract configuration ID from file path or data
                config_id = data.get('config_id') or result_file.stem
                
                # Ensure proper structure
                if 'results' not in data:
                    self.logger.warning(f"No results found in {result_file}")
                    continue
                
                results[config_id] = {
                    'config_id': config_id,
                    'config_type': config_type,
                    'config': data.get('config', {}),
                    'results': data['results'],
                    'metadata': data.get('metadata', {}),
                    'file_path': str(result_file)
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to load {result_file}: {e}")
        
        return results
    
    def run_statistical_analysis(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive statistical analysis."""
        self.logger.info("Running statistical analysis...")
        
        # Prepare data for analysis
        analysis_data = self._prepare_analysis_data(experimental_data)
        
        # Run hypothesis testing
        hypothesis_results = self.statistical_analyzer.test_hypotheses(
            analysis_data, self.hypothesis_framework
        )
        
        # Calculate effect sizes
        effect_sizes = self.statistical_analyzer.calculate_effect_sizes(analysis_data)
        
        # Perform multiple comparison correction
        corrected_results = self.statistical_analyzer.correct_multiple_comparisons(
            hypothesis_results, method='holm_bonferroni'
        )
        
        # Generate confidence intervals
        confidence_intervals = self.statistical_analyzer.bootstrap_confidence_intervals(
            analysis_data, n_bootstrap=10000, confidence_level=0.95
        )
        
        statistical_results = {
            'hypothesis_testing': corrected_results,
            'effect_sizes': effect_sizes,
            'confidence_intervals': confidence_intervals,
            'statistical_summary': self._generate_statistical_summary(
                corrected_results, effect_sizes
            ),
            'metadata': {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'framework_version': self.hypothesis_framework.get('version', '1.0'),
                'total_comparisons': len(corrected_results)
            }
        }
        
        return statistical_results
    
    def _prepare_analysis_data(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for statistical analysis."""
        # Convert experimental data to format expected by StatisticalAnalyzer
        prepared_data = {
            'experiments': [],
            'metadata': experimental_data['metadata']
        }
        
        # Add baseline results
        for config_id, data in experimental_data['baseline_results'].items():
            experiment = {
                'config_id': config_id,
                'config_type': 'baseline',
                'config': data['config'],
                'results': data['results'],
                'status': 'completed'  # Assume loaded results are completed
            }
            prepared_data['experiments'].append(experiment)
        
        # Add lethe results
        for config_id, data in experimental_data['lethe_results'].items():
            experiment = {
                'config_id': config_id,
                'config_type': 'lethe',
                'config': data['config'],
                'results': data['results'],
                'status': 'completed'
            }
            prepared_data['experiments'].append(experiment)
        
        return prepared_data
    
    def _generate_statistical_summary(self, hypothesis_results: Dict[str, Any],
                                    effect_sizes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of statistical findings."""
        summary = {
            'supported_hypotheses': [],
            'rejected_hypotheses': [],
            'significant_effects': [],
            'overall_conclusions': []
        }
        
        for hypothesis, result in hypothesis_results.items():
            if result.get('significant', False):
                summary['supported_hypotheses'].append({
                    'hypothesis': hypothesis,
                    'p_value': result.get('p_value'),
                    'corrected_p_value': result.get('corrected_p_value'),
                    'effect_size': effect_sizes.get(hypothesis, {}).get('cliff_delta')
                })
            else:
                summary['rejected_hypotheses'].append({
                    'hypothesis': hypothesis,
                    'p_value': result.get('p_value'),
                    'reason': 'Not statistically significant'
                })
        
        # Identify significant effects
        for metric, effect_data in effect_sizes.items():
            cliff_delta = effect_data.get('cliff_delta', 0)
            if abs(cliff_delta) >= 0.3:  # Medium effect size threshold
                summary['significant_effects'].append({
                    'metric': metric,
                    'effect_size': cliff_delta,
                    'magnitude': 'large' if abs(cliff_delta) >= 0.5 else 'medium'
                })
        
        # Generate overall conclusions
        supported_count = len(summary['supported_hypotheses'])
        total_count = len(hypothesis_results)
        
        if supported_count == total_count:
            summary['overall_conclusions'].append("All hypotheses supported with statistical significance")
        elif supported_count > total_count * 0.5:
            summary['overall_conclusions'].append(f"Majority of hypotheses supported ({supported_count}/{total_count})")
        else:
            summary['overall_conclusions'].append(f"Limited hypothesis support ({supported_count}/{total_count})")
        
        return summary
    
    def run_fraud_proofing(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive fraud-proofing validation."""
        self.logger.info("Running fraud-proofing validation...")
        
        # Convert data format for fraud framework
        fraud_data = self._prepare_fraud_data(experimental_data)
        
        # Run fraud-proofing checks
        fraud_results = self.fraud_framework.validate_results(fraud_data)
        
        return fraud_results
    
    def _prepare_fraud_data(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for fraud-proofing framework."""
        # Similar to statistical analysis preparation
        return self._prepare_analysis_data(experimental_data)
    
    def generate_comprehensive_report(self, experimental_data: Dict[str, Any],
                                    statistical_results: Dict[str, Any],
                                    fraud_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        self.logger.info("Generating comprehensive report...")
        
        report = {
            'metadata': {
                'report_generated': datetime.utcnow().isoformat(),
                'analysis_version': '1.0.0',
                'total_configurations': experimental_data['metadata']['total_configs'],
                'hypothesis_framework_version': self.hypothesis_framework.get('version', '1.0')
            },
            'experimental_overview': {
                'baseline_systems': len(experimental_data['baseline_results']),
                'lethe_configurations': len(experimental_data['lethe_results']),
                'evaluation_metrics': self._extract_available_metrics(experimental_data)
            },
            'hypothesis_results': statistical_results['hypothesis_testing'],
            'statistical_analysis': {
                'effect_sizes': statistical_results['effect_sizes'],
                'confidence_intervals': statistical_results['confidence_intervals'],
                'summary': statistical_results['statistical_summary']
            },
            'fraud_proofing': {
                'validation_results': fraud_results.get('validation_results', {}),
                'overall_status': fraud_results.get('overall_status', 'unknown'),
                'warnings': fraud_results.get('warnings', [])
            },
            'recommendations': self._generate_recommendations(
                statistical_results, fraud_results
            ),
            'publication_readiness': self._assess_publication_readiness(
                statistical_results, fraud_results
            )
        }
        
        return report
    
    def _extract_available_metrics(self, experimental_data: Dict[str, Any]) -> List[str]:
        """Extract list of available metrics from experimental data."""
        metrics = set()
        
        for config_data in experimental_data['baseline_results'].values():
            metrics.update(config_data.get('results', {}).keys())
        
        for config_data in experimental_data['lethe_results'].values():
            metrics.update(config_data.get('results', {}).keys())
        
        return sorted(list(metrics))
    
    def _generate_recommendations(self, statistical_results: Dict[str, Any],
                                fraud_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Statistical recommendations
        supported = len(statistical_results['statistical_summary']['supported_hypotheses'])
        total = len(statistical_results['hypothesis_testing'])
        
        if supported == total:
            recommendations.append("Strong evidence supports all hypotheses - ready for publication")
        elif supported > total * 0.7:
            recommendations.append("Good evidence for most hypotheses - consider additional analysis for unsupported claims")
        else:
            recommendations.append("Limited hypothesis support - conduct additional experiments")
        
        # Effect size recommendations
        significant_effects = statistical_results['statistical_summary']['significant_effects']
        if len(significant_effects) < 2:
            recommendations.append("Consider additional metrics to demonstrate practical significance")
        
        # Fraud-proofing recommendations
        fraud_status = fraud_results.get('overall_status', 'unknown')
        if fraud_status != 'passed':
            recommendations.append("Address fraud-proofing validation issues before publication")
        
        if fraud_results.get('warnings'):
            recommendations.append("Review and address fraud-proofing warnings")
        
        return recommendations
    
    def _assess_publication_readiness(self, statistical_results: Dict[str, Any],
                                    fraud_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for academic publication."""
        readiness = {
            'overall_score': 0.0,
            'criteria': {},
            'blocking_issues': [],
            'recommendations': []
        }
        
        # Statistical significance (25%)
        supported = len(statistical_results['statistical_summary']['supported_hypotheses'])
        total = len(statistical_results['hypothesis_testing'])
        stat_score = (supported / total) if total > 0 else 0
        readiness['criteria']['statistical_significance'] = stat_score
        
        # Effect sizes (25%)
        significant_effects = len(statistical_results['statistical_summary']['significant_effects'])
        effect_score = min(1.0, significant_effects / 3)  # Expect at least 3 significant effects
        readiness['criteria']['effect_sizes'] = effect_score
        
        # Fraud-proofing (30%)
        fraud_score = 1.0 if fraud_results.get('overall_status') == 'passed' else 0.0
        readiness['criteria']['fraud_proofing'] = fraud_score
        
        # Data quality (20%)
        total_configs = statistical_results['metadata']['total_comparisons']
        data_score = min(1.0, total_configs / 50)  # Expect at least 50 configurations
        readiness['criteria']['data_completeness'] = data_score
        
        # Calculate overall score
        weights = {'statistical_significance': 0.25, 'effect_sizes': 0.25, 
                  'fraud_proofing': 0.30, 'data_completeness': 0.20}
        
        readiness['overall_score'] = sum(
            score * weights[criteria] 
            for criteria, score in readiness['criteria'].items()
        )
        
        # Identify blocking issues
        if fraud_score < 1.0:
            readiness['blocking_issues'].append("Fraud-proofing validation failures")
        
        if stat_score < 0.5:
            readiness['blocking_issues'].append("Insufficient statistical evidence")
        
        # Publication readiness assessment
        if readiness['overall_score'] >= 0.8:
            readiness['status'] = 'ready'
            readiness['recommendations'].append("Results are publication-ready for NeurIPS submission")
        elif readiness['overall_score'] >= 0.6:
            readiness['status'] = 'nearly_ready'
            readiness['recommendations'].append("Address minor issues and rerun analysis")
        else:
            readiness['status'] = 'not_ready'
            readiness['recommendations'].append("Significant additional work needed before publication")
        
        return readiness
    
    def execute_analysis(self) -> Dict[str, Any]:
        """Execute complete analysis pipeline."""
        self.logger.info("🔬 Starting comprehensive analysis...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experimental data
        experimental_data = self.load_experimental_data()
        
        # Save experimental data for reference
        data_file = self.output_dir / 'experimental_data.json'
        with open(data_file, 'w') as f:
            json.dump(experimental_data, f, indent=2)
        
        # Run statistical analysis
        statistical_results = self.run_statistical_analysis(experimental_data)
        
        # Save statistical results
        stats_file = self.output_dir / 'statistical_results.json'
        with open(stats_file, 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        # Run fraud-proofing
        fraud_results = self.run_fraud_proofing(experimental_data)
        
        # Save fraud-proofing results
        fraud_file = self.output_dir / 'fraud_proofing_results.json'
        with open(fraud_file, 'w') as f:
            json.dump(fraud_results, f, indent=2)
        
        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(
            experimental_data, statistical_results, fraud_results
        )
        
        # Save comprehensive report
        report_file = self.output_dir / 'summary_report.json'
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        self.logger.info(f"Analysis complete - results saved to: {self.output_dir}")
        
        return comprehensive_report

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive analysis')
    parser.add_argument('--baseline-results', type=Path, required=True,
                       help='Directory containing baseline evaluation results')
    parser.add_argument('--lethe-results', type=Path, required=True,
                       help='Directory containing Lethe evaluation results')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--config', type=Path,
                       help='Analysis configuration file')
    parser.add_argument('--hypothesis-framework', type=Path,
                       help='Hypothesis testing framework JSON file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Create analyzer
        analyzer = ComprehensiveAnalyzer(
            baseline_results_dir=args.baseline_results,
            lethe_results_dir=args.lethe_results,
            output_dir=args.output,
            config_path=args.config or Path('experiments/grid_config.yaml'),
            hypothesis_framework_path=args.hypothesis_framework or Path('experiments/hypothesis_framework.json'),
            logger=logger
        )
        
        # Execute analysis
        results = analyzer.execute_analysis()
        
        # Print summary
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"Publication readiness: {results['publication_readiness']['status']}")
        logger.info(f"Overall score: {results['publication_readiness']['overall_score']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()