#!/usr/bin/env python3
"""
Risk Assessment System for Lethe Hybrid IR System
=================================================

Implements comprehensive risk assessment with composite scoring for:
- Statistical risk (CI bounds, significance)
- Mutation testing risk (fault detection)
- Static analysis risk (SAST findings)
- Performance risk (latency, memory)
- Robustness risk (cross-validation stability)

Key Features:
- T2 risk score computation
- Multi-dimensional risk profiling
- Decision feature normalization
- Failure mode identification
- Risk-based configuration filtering
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from scipy import stats


class RiskAssessmentSystem:
    """Comprehensive risk assessment system for hybrid IR configurations"""
    
    def __init__(self, output_dir: Path = Path("analysis")):
        """
        Initialize risk assessment system
        
        Args:
            output_dir: Directory for output files and reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Risk thresholds from Task 5 specifications
        self.thresholds = {
            'mutation_score_min': 0.80,    # T_mut threshold
            'property_score_min': 0.70,    # T_prop threshold  
            'sast_high_max': 0,            # SAST high severity max
            'statistical_significance': 0.05,  # Statistical significance
            'ci_lower_bound_min': 0.0,     # CI policy: wins only when CI > 0
            'performance_p95_max': 3000,   # P95 latency budget (ms)
            'memory_max': 1500             # Memory budget (MB)
        }
        
        # Risk weight configuration
        self.risk_weights = {
            'statistical_risk': 0.25,      # Statistical significance/CI risks
            'mutation_risk': 0.20,         # Mutation testing fault detection  
            'static_analysis_risk': 0.15,  # SAST findings
            'performance_risk': 0.20,      # Latency/memory risks
            'robustness_risk': 0.20        # Cross-validation stability
        }
        
        print(f"Initialized Risk Assessment System")
        print(f"Risk thresholds: {self.thresholds}")
        print(f"Risk weights: {self.risk_weights}")
        print(f"Output directory: {self.output_dir}")
    
    def compute_statistical_risk(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute statistical risk score based on CI bounds and significance
        
        Args:
            config_data: Configuration performance and statistical data
            
        Returns:
            Statistical risk assessment
        """
        risk_factors = []
        risk_details = {}
        
        # CI lower bound risk (Task 5: wins only when CI lower bound > 0)
        ci_data = config_data.get('confidence_intervals', {})
        for metric, ci_info in ci_data.items():
            if isinstance(ci_info, dict) and 'ci_lower' in ci_info:
                ci_lower = ci_info['ci_lower']
                if ci_lower <= self.thresholds['ci_lower_bound_min']:
                    # High risk if CI includes zero or negative values
                    risk_factor = max(0, -ci_lower + 0.1)  # More negative = higher risk
                    risk_factors.append(risk_factor)
                    risk_details[f'ci_risk_{metric}'] = float(risk_factor)
                else:
                    risk_details[f'ci_risk_{metric}'] = 0.0
        
        # P-value significance risk
        p_values = config_data.get('p_values', {})
        for comparison, p_value in p_values.items():
            if p_value > self.thresholds['statistical_significance']:
                # Risk increases as p-value approaches or exceeds significance threshold
                risk_factor = min(1.0, p_value / self.thresholds['statistical_significance'])
                risk_factors.append(risk_factor)
                risk_details[f'significance_risk_{comparison}'] = float(risk_factor)
        
        # Effect size confidence risk
        effect_sizes = config_data.get('effect_sizes', {})
        for comparison, effect_data in effect_sizes.items():
            if isinstance(effect_data, dict):
                ci_lower = effect_data.get('ci_lower', 0)
                ci_upper = effect_data.get('ci_upper', 0)
                
                # Risk if effect size CI is wide or includes zero
                ci_width = abs(ci_upper - ci_lower)
                includes_zero = ci_lower <= 0 <= ci_upper
                
                width_risk = min(1.0, ci_width / 2.0)  # Normalize wide CIs
                zero_risk = 1.0 if includes_zero else 0.0
                
                combined_risk = max(width_risk, zero_risk)
                risk_factors.append(combined_risk)
                risk_details[f'effect_size_risk_{comparison}'] = float(combined_risk)
        
        # Aggregate statistical risk
        statistical_risk = np.mean(risk_factors) if risk_factors else 0.0
        
        return {
            'statistical_risk_score': float(statistical_risk),
            'risk_factors_count': len(risk_factors),
            'risk_details': risk_details,
            'high_risk': statistical_risk > 0.5
        }
    
    def compute_mutation_risk(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute mutation testing risk based on fault detection capability
        
        Args:
            config_data: Configuration data including mutation testing results
            
        Returns:
            Mutation testing risk assessment
        """
        mutation_score = config_data.get('mutation_score', 0.8)  # Default to threshold
        
        # Risk increases as mutation score falls below threshold
        if mutation_score < self.thresholds['mutation_score_min']:
            score_deficit = self.thresholds['mutation_score_min'] - mutation_score
            mutation_risk = min(1.0, score_deficit / 0.3)  # Scale deficit to [0,1]
        else:
            mutation_risk = 0.0
        
        # Additional risk factors
        mutation_details = config_data.get('mutation_details', {})
        failed_mutations = mutation_details.get('failed_mutations', 0)
        total_mutations = mutation_details.get('total_mutations', 100)
        
        # Risk from specific mutation failures
        equivalence_risk = mutation_details.get('equivalent_mutations', 0) / total_mutations
        timeout_risk = mutation_details.get('timeout_mutations', 0) / total_mutations
        
        # Combine risks
        combined_risk = max(mutation_risk, equivalence_risk + timeout_risk)
        
        return {
            'mutation_risk_score': float(combined_risk),
            'mutation_score': float(mutation_score),
            'score_deficit': float(max(0, self.thresholds['mutation_score_min'] - mutation_score)),
            'failed_mutations': failed_mutations,
            'total_mutations': total_mutations,
            'equivalence_risk': float(equivalence_risk),
            'timeout_risk': float(timeout_risk),
            'high_risk': combined_risk > 0.3
        }
    
    def compute_static_analysis_risk(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute static analysis risk based on SAST findings
        
        Args:
            config_data: Configuration data including SAST results
            
        Returns:
            Static analysis risk assessment
        """
        sast_results = config_data.get('sast_results', {})
        
        # Count issues by severity
        critical_issues = sast_results.get('critical_issues', 0)
        high_issues = sast_results.get('high_issues', 0) 
        medium_issues = sast_results.get('medium_issues', 0)
        low_issues = sast_results.get('low_issues', 0)
        
        # Risk from high/critical issues (Task 5: SAST_high = 0 requirement)
        high_severity_count = critical_issues + high_issues
        if high_severity_count > self.thresholds['sast_high_max']:
            severity_risk = min(1.0, high_severity_count / 5.0)  # Scale to [0,1]
        else:
            severity_risk = 0.0
        
        # Risk from issue density
        total_issues = critical_issues + high_issues + medium_issues + low_issues
        lines_of_code = config_data.get('lines_of_code', 10000)  # Estimate
        issue_density = total_issues / lines_of_code * 1000  # Issues per 1k lines
        
        density_risk = min(1.0, issue_density / 10.0)  # Normalize density
        
        # Security-specific risks
        security_issues = sast_results.get('security_issues', 0)
        vulnerability_score = sast_results.get('vulnerability_score', 0)
        
        security_risk = min(1.0, (security_issues + vulnerability_score) / 10.0)
        
        # Combine risks
        combined_risk = max(severity_risk, density_risk, security_risk)
        
        return {
            'static_analysis_risk_score': float(combined_risk),
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'medium_issues': medium_issues,
            'low_issues': low_issues,
            'total_issues': total_issues,
            'issue_density_per_1k_loc': float(issue_density),
            'security_issues': security_issues,
            'severity_risk': float(severity_risk),
            'density_risk': float(density_risk),
            'security_risk': float(security_risk),
            'high_risk': combined_risk > 0.2
        }
    
    def compute_performance_risk(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute performance risk based on latency and memory constraints
        
        Args:
            config_data: Configuration performance data
            
        Returns:
            Performance risk assessment
        """
        # Latency risk
        p95_latency = config_data.get('latency_p95', 
                                    config_data.get('latency_ms_total', 1000) * 1.5)
        
        if p95_latency > self.thresholds['performance_p95_max']:
            latency_excess = p95_latency - self.thresholds['performance_p95_max']
            latency_risk = min(1.0, latency_excess / self.thresholds['performance_p95_max'])
        else:
            latency_risk = 0.0
        
        # Memory risk  
        memory_usage = config_data.get('memory_peak_mb', 
                                     config_data.get('memory_mb', 800))
        
        if memory_usage > self.thresholds['memory_max']:
            memory_excess = memory_usage - self.thresholds['memory_max']
            memory_risk = min(1.0, memory_excess / self.thresholds['memory_max'])
        else:
            memory_risk = 0.0
        
        # Throughput risk (if available)
        throughput = config_data.get('throughput_qps', 100)
        min_throughput = config_data.get('min_throughput_required', 50)
        
        if throughput < min_throughput:
            throughput_deficit = min_throughput - throughput
            throughput_risk = min(1.0, throughput_deficit / min_throughput)
        else:
            throughput_risk = 0.0
        
        # Stability risk from performance variance
        latency_variance = config_data.get('latency_variance', 0.1)
        variance_risk = min(1.0, latency_variance)
        
        # Combine risks
        combined_risk = max(latency_risk, memory_risk, throughput_risk, variance_risk)
        
        return {
            'performance_risk_score': float(combined_risk),
            'p95_latency_ms': float(p95_latency),
            'memory_peak_mb': float(memory_usage),
            'throughput_qps': float(throughput),
            'latency_risk': float(latency_risk),
            'memory_risk': float(memory_risk),
            'throughput_risk': float(throughput_risk),
            'variance_risk': float(variance_risk),
            'latency_budget_exceeded': p95_latency > self.thresholds['performance_p95_max'],
            'memory_budget_exceeded': memory_usage > self.thresholds['memory_max'],
            'high_risk': combined_risk > 0.4
        }
    
    def compute_robustness_risk(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute robustness risk based on cross-validation stability
        
        Args:
            config_data: Configuration robustness data
            
        Returns:
            Robustness risk assessment
        """
        # Cross-validation variance risk
        cv_results = config_data.get('cross_validation', {})
        metric_variances = []
        
        for metric, cv_scores in cv_results.items():
            if isinstance(cv_scores, list) and len(cv_scores) > 1:
                variance = np.var(cv_scores)
                mean_score = np.mean(cv_scores)
                
                # Coefficient of variation as risk measure
                if mean_score > 0:
                    cv_risk = min(1.0, variance / (mean_score ** 2))
                    metric_variances.append(cv_risk)
        
        cv_variance_risk = np.mean(metric_variances) if metric_variances else 0.0
        
        # Data drift risk
        train_performance = config_data.get('train_performance', {})
        test_performance = config_data.get('test_performance', {})
        
        drift_risks = []
        for metric in ['ndcg_at_10', 'recall_at_20']:
            if metric in train_performance and metric in test_performance:
                train_score = train_performance[metric]
                test_score = test_performance[metric]
                
                if train_score > 0:
                    drift = abs(train_score - test_score) / train_score
                    drift_risks.append(min(1.0, drift * 2))  # Scale drift
        
        drift_risk = np.mean(drift_risks) if drift_risks else 0.0
        
        # Adversarial robustness risk (if available)
        adversarial_drop = config_data.get('adversarial_performance_drop', 0.0)
        adversarial_risk = min(1.0, adversarial_drop)
        
        # Input sensitivity risk
        input_sensitivity = config_data.get('input_sensitivity_score', 0.1)
        sensitivity_risk = min(1.0, input_sensitivity)
        
        # Combine risks
        combined_risk = max(cv_variance_risk, drift_risk, adversarial_risk, sensitivity_risk)
        
        return {
            'robustness_risk_score': float(combined_risk),
            'cv_variance_risk': float(cv_variance_risk),
            'drift_risk': float(drift_risk),
            'adversarial_risk': float(adversarial_risk),
            'sensitivity_risk': float(sensitivity_risk),
            'cv_metrics_analyzed': len(metric_variances),
            'train_test_metrics_compared': len(drift_risks),
            'high_risk': combined_risk > 0.3
        }
    
    def compute_composite_risk_score(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute T2 composite risk score from all risk dimensions
        
        Args:
            config_data: Complete configuration data
            
        Returns:
            Composite risk assessment with T2 score
        """
        # Compute individual risk scores
        statistical = self.compute_statistical_risk(config_data)
        mutation = self.compute_mutation_risk(config_data)
        static_analysis = self.compute_static_analysis_risk(config_data)
        performance = self.compute_performance_risk(config_data)
        robustness = self.compute_robustness_risk(config_data)
        
        # Weighted composite score (T2 risk score)
        t2_risk_score = (
            self.risk_weights['statistical_risk'] * statistical['statistical_risk_score'] +
            self.risk_weights['mutation_risk'] * mutation['mutation_risk_score'] +
            self.risk_weights['static_analysis_risk'] * static_analysis['static_analysis_risk_score'] +
            self.risk_weights['performance_risk'] * performance['performance_risk_score'] +
            self.risk_weights['robustness_risk'] * robustness['robustness_risk_score']
        )
        
        # Normalize features to [0,1] for gatekeeper
        normalized_features = {
            'statistical_risk_norm': float(statistical['statistical_risk_score']),
            'mutation_risk_norm': float(mutation['mutation_risk_score']),
            'static_analysis_risk_norm': float(static_analysis['static_analysis_risk_score']),
            'performance_risk_norm': float(performance['performance_risk_score']),
            'robustness_risk_norm': float(robustness['robustness_risk_score'])
        }
        
        # Risk profile categorization
        high_risk_dimensions = sum([
            statistical['high_risk'],
            mutation['high_risk'],
            static_analysis['high_risk'],
            performance['high_risk'],
            robustness['high_risk']
        ])
        
        # Overall risk category
        if t2_risk_score < 0.2:
            risk_category = 'low'
        elif t2_risk_score < 0.5:
            risk_category = 'medium'
        elif t2_risk_score < 0.8:
            risk_category = 'high'
        else:
            risk_category = 'critical'
        
        # Failure mode identification
        failure_modes = []
        if statistical['high_risk']:
            failure_modes.append('statistical_significance')
        if mutation['high_risk']:
            failure_modes.append('fault_detection')
        if static_analysis['high_risk']:
            failure_modes.append('security_vulnerability')
        if performance['high_risk']:
            failure_modes.append('performance_degradation')
        if robustness['high_risk']:
            failure_modes.append('robustness_instability')
        
        return {
            't2_risk_score': float(t2_risk_score),
            'risk_category': risk_category,
            'high_risk_dimensions': high_risk_dimensions,
            'failure_modes': failure_modes,
            'normalized_features': normalized_features,
            'detailed_assessments': {
                'statistical': statistical,
                'mutation': mutation,
                'static_analysis': static_analysis,
                'performance': performance,
                'robustness': robustness
            },
            'risk_weights': self.risk_weights,
            'thresholds_applied': self.thresholds
        }
    
    def assess_configuration_risks(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess risks for multiple configurations
        
        Args:
            configurations: List of configuration data dictionaries
            
        Returns:
            Comprehensive risk assessment for all configurations
        """
        print(f"Assessing risks for {len(configurations)} configurations...")
        
        risk_assessments = []
        
        for i, config in enumerate(configurations):
            config_id = config.get('method', f'config_{i}')
            
            try:
                risk_assessment = self.compute_composite_risk_score(config)
                risk_assessment['configuration_id'] = config_id
                risk_assessment['assessment_index'] = i
                risk_assessments.append(risk_assessment)
                
            except Exception as e:
                print(f"Warning: Risk assessment failed for {config_id}: {e}")
                # Create minimal risk assessment
                risk_assessments.append({
                    'configuration_id': config_id,
                    'assessment_index': i,
                    't2_risk_score': 0.5,  # Default medium risk
                    'risk_category': 'medium',
                    'high_risk_dimensions': 0,
                    'failure_modes': ['assessment_failure'],
                    'error': str(e)
                })
        
        # Aggregate statistics
        risk_scores = [r['t2_risk_score'] for r in risk_assessments if 't2_risk_score' in r]
        
        risk_summary = {
            'total_configurations': len(configurations),
            'successful_assessments': len([r for r in risk_assessments if 'error' not in r]),
            'failed_assessments': len([r for r in risk_assessments if 'error' in r]),
            'risk_score_distribution': {
                'mean': float(np.mean(risk_scores)) if risk_scores else 0.0,
                'median': float(np.median(risk_scores)) if risk_scores else 0.0,
                'std': float(np.std(risk_scores)) if risk_scores else 0.0,
                'min': float(np.min(risk_scores)) if risk_scores else 0.0,
                'max': float(np.max(risk_scores)) if risk_scores else 0.0
            },
            'risk_categories': {
                'low': len([r for r in risk_assessments if r.get('risk_category') == 'low']),
                'medium': len([r for r in risk_assessments if r.get('risk_category') == 'medium']),
                'high': len([r for r in risk_assessments if r.get('risk_category') == 'high']),
                'critical': len([r for r in risk_assessments if r.get('risk_category') == 'critical'])
            }
        }
        
        return {
            'assessment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'assessor': 'RiskAssessmentSystem',
                'version': '1.0.0',
                'risk_weights': self.risk_weights,
                'thresholds': self.thresholds
            },
            'risk_summary': risk_summary,
            'individual_assessments': risk_assessments,
            'recommendations': self._generate_risk_recommendations(risk_assessments)
        }
    
    def _generate_risk_recommendations(self, risk_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate risk-based recommendations"""
        
        # Find lowest risk configurations
        valid_assessments = [r for r in risk_assessments if 't2_risk_score' in r]
        
        if not valid_assessments:
            return {'error': 'No valid risk assessments available'}
        
        # Sort by risk score
        sorted_assessments = sorted(valid_assessments, key=lambda x: x['t2_risk_score'])
        
        recommendations = {
            'lowest_risk_configurations': [
                {
                    'configuration': a['configuration_id'],
                    'risk_score': a['t2_risk_score'],
                    'risk_category': a['risk_category'],
                    'failure_modes': a['failure_modes']
                }
                for a in sorted_assessments[:5]
            ],
            'high_risk_configurations': [
                {
                    'configuration': a['configuration_id'],
                    'risk_score': a['t2_risk_score'],
                    'risk_category': a['risk_category'],
                    'failure_modes': a['failure_modes']
                }
                for a in sorted_assessments if a['risk_category'] in ['high', 'critical']
            ],
            'risk_mitigation_strategies': {
                'statistical_significance': 'Increase sample size, use more conservative significance thresholds',
                'fault_detection': 'Improve test coverage, add property-based testing',
                'security_vulnerability': 'Run additional SAST tools, manual security review',
                'performance_degradation': 'Optimize algorithms, add caching, scale infrastructure',
                'robustness_instability': 'Add regularization, improve data quality, ensemble methods'
            }
        }
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_file: Path) -> None:
        """Save risk assessment results to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Risk assessment results saved to: {output_file}")


def load_configuration_data(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Load configuration data from artifacts directory"""
    
    data_sources = [
        artifacts_dir / "bca_bootstrap_results.json",
        artifacts_dir / "pareto_analysis_results.json",
        artifacts_dir / "statistical_analysis_results.json"
    ]
    
    configurations = []
    
    for source in data_sources:
        if source.exists():
            print(f"Loading configuration data from {source}")
            with open(source, 'r') as f:
                data = json.load(f)
            
            # Extract configuration data based on file type
            if 'method_comparisons' in data:  # Bootstrap results
                for family, family_data in data['method_comparisons'].items():
                    for method, method_data in family_data.items():
                        config = {'method': method, 'data_source': str(source)}
                        
                        # Add statistical risk data
                        for metric, comparison in method_data.items():
                            config[f'{metric}_p_value'] = comparison.get('p_value', 1.0)
                            config[f'{metric}_significant'] = comparison.get('significant', False)
                            if 'difference_ci' in comparison:
                                ci_data = comparison['difference_ci']
                                config[f'{metric}_ci_lower'] = ci_data.get('ci_lower', 0)
                                config[f'{metric}_ci_upper'] = ci_data.get('ci_upper', 0)
                        
                        configurations.append(config)
            
            elif 'pareto_configurations' in data:  # Pareto results
                for config in data['pareto_configurations']:
                    config['data_source'] = str(source)
                    configurations.append(config)
            
            elif 'summary_statistics' in data:  # Statistical results
                for method, method_stats in data['summary_statistics'].items():
                    config = {'method': method, 'data_source': str(source)}
                    
                    # Add performance data
                    for metric, stats in method_stats.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            config[metric] = stats['mean']
                    
                    configurations.append(config)
    
    if not configurations:
        # Create mock data for demonstration
        print("No configuration data found, creating mock data...")
        configurations = create_mock_configuration_data()
    
    print(f"Loaded {len(configurations)} configurations for risk assessment")
    return configurations


def create_mock_configuration_data() -> List[Dict[str, Any]]:
    """Create mock configuration data for testing"""
    
    methods = ['iter1', 'iter2', 'iter3', 'iter4', 'baseline_bm25_vector_simple']
    configurations = []
    
    np.random.seed(42)  # Reproducible
    
    for method in methods:
        config = {
            'method': method,
            'data_source': 'mock',
            # Performance metrics
            'ndcg_at_10': np.random.uniform(0.4, 0.9),
            'latency_ms_total': np.random.uniform(500, 2500),
            'memory_peak_mb': np.random.uniform(600, 1400),
            # Statistical risk factors
            'ndcg_at_10_p_value': np.random.uniform(0.001, 0.1),
            'ndcg_at_10_ci_lower': np.random.uniform(-0.1, 0.3),
            'ndcg_at_10_ci_upper': np.random.uniform(0.3, 0.8),
            # Mutation testing
            'mutation_score': np.random.uniform(0.65, 0.95),
            # SAST results
            'sast_results': {
                'critical_issues': np.random.poisson(0.5),
                'high_issues': np.random.poisson(1),
                'medium_issues': np.random.poisson(3),
                'low_issues': np.random.poisson(8)
            },
            # Cross-validation stability
            'cross_validation': {
                'ndcg_at_10': np.random.normal(0.7, 0.05, 5).tolist()
            }
        }
        
        configurations.append(config)
    
    return configurations


def main():
    """Main execution function"""
    artifacts_dir = Path("artifacts")
    output_dir = Path("analysis")
    
    # Initialize risk assessment system
    risk_assessor = RiskAssessmentSystem(output_dir)
    
    # Load configuration data
    configurations = load_configuration_data(artifacts_dir)
    
    # Run risk assessment
    risk_results = risk_assessor.assess_configuration_risks(configurations)
    
    # Save results
    output_file = output_dir / "risk_assessment_results.json"
    risk_assessor.save_results(risk_results, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("RISK ASSESSMENT SUMMARY")
    print("="*60)
    
    summary = risk_results['risk_summary']
    print(f"Configurations assessed: {summary['total_configurations']}")
    print(f"Successful assessments: {summary['successful_assessments']}")
    print(f"Failed assessments: {summary['failed_assessments']}")
    
    dist = summary['risk_score_distribution']
    print(f"\nRisk Score Distribution:")
    print(f"  Mean: {dist['mean']:.3f}")
    print(f"  Median: {dist['median']:.3f}")
    print(f"  Range: [{dist['min']:.3f}, {dist['max']:.3f}]")
    
    categories = summary['risk_categories']
    print(f"\nRisk Categories:")
    for category, count in categories.items():
        print(f"  {category.title()}: {count}")
    
    recommendations = risk_results.get('recommendations', {})
    if 'lowest_risk_configurations' in recommendations:
        lowest_risk = recommendations['lowest_risk_configurations'][0]
        print(f"\nLowest risk configuration: {lowest_risk['configuration']}")
        print(f"  Risk score: {lowest_risk['risk_score']:.3f}")
        print(f"  Risk category: {lowest_risk['risk_category']}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()