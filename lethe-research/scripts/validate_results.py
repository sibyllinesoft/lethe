#!/usr/bin/env python3
"""
Results Validation Script
=========================

Validates the integrity and quality of experimental results for publication readiness.
Performs comprehensive checks on:
- Data completeness and format validation
- Statistical result consistency
- Figure quality and completeness
- Publication readiness assessment

Usage:
    python validate_results.py --results-dir artifacts/full_evaluation_20241223_154529/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import yaml
from datetime import datetime

def setup_logging(level: str) -> logging.Logger:
    """Configure logging with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class ResultsValidator:
    """Comprehensive results validation framework."""
    
    def __init__(self, results_dir: Path, logger: logging.Logger):
        self.results_dir = results_dir
        self.logger = logger
        
        # Validation results
        self.validation_results = {
            'overall_status': 'unknown',
            'validation_timestamp': datetime.utcnow().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': [],
            'publication_readiness': {
                'score': 0.0,
                'status': 'not_ready',
                'blocking_issues': [],
                'recommendations': []
            }
        }
    
    def validate_directory_structure(self) -> bool:
        """Validate expected directory structure exists."""
        self.logger.info("Validating directory structure...")
        
        expected_dirs = [
            'datasets', 'baselines', 'lethe_runs', 'analysis', 'figures'
        ]
        
        expected_files = [
            'environment.json', 
            'analysis/summary_report.json',
            'figures/figure_manifest.json'
        ]
        
        structure_valid = True
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_name in expected_dirs:
            dir_path = self.results_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                structure_valid = False
                
        # Check files
        for file_path in expected_files:
            full_path = self.results_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                structure_valid = False
        
        # Record results
        self.validation_results['checks']['directory_structure'] = {
            'status': 'passed' if structure_valid else 'failed',
            'missing_directories': missing_dirs,
            'missing_files': missing_files
        }
        
        if missing_dirs:
            self.validation_results['errors'].append(f"Missing directories: {missing_dirs}")
        
        if missing_files:
            self.validation_results['errors'].append(f"Missing files: {missing_files}")
        
        self.logger.info(f"Directory structure validation: {'✅ PASSED' if structure_valid else '❌ FAILED'}")
        return structure_valid
    
    def validate_dataset_integrity(self) -> bool:
        """Validate dataset completeness and format."""
        self.logger.info("Validating dataset integrity...")
        
        dataset_path = self.results_dir / 'datasets' / 'lethebench.json'
        
        if not dataset_path.exists():
            self.validation_results['errors'].append("Dataset file not found")
            self.validation_results['checks']['dataset_integrity'] = {'status': 'failed', 'reason': 'file_not_found'}
            return False
        
        try:
            with open(dataset_path) as f:
                dataset = json.load(f)
            
            # Validate structure
            required_fields = ['metadata', 'examples', 'queries', 'evaluation_config']
            missing_fields = [field for field in required_fields if field not in dataset]
            
            if missing_fields:
                self.validation_results['errors'].append(f"Dataset missing fields: {missing_fields}")
                self.validation_results['checks']['dataset_integrity'] = {
                    'status': 'failed', 
                    'missing_fields': missing_fields
                }
                return False
            
            # Validate content
            num_examples = len(dataset.get('examples', []))
            num_queries = len(dataset.get('queries', []))
            
            if num_examples < 10:
                self.validation_results['warnings'].append(f"Low number of examples: {num_examples}")
            
            if num_queries < 50:
                self.validation_results['warnings'].append(f"Low number of queries: {num_queries}")
            
            # Record results
            self.validation_results['checks']['dataset_integrity'] = {
                'status': 'passed',
                'num_examples': num_examples,
                'num_queries': num_queries,
                'domains': len(set(ex.get('domain', 'unknown') for ex in dataset['examples']))
            }
            
            self.logger.info(f"Dataset validation: ✅ PASSED ({num_examples} examples, {num_queries} queries)")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Dataset validation failed: {e}")
            self.validation_results['checks']['dataset_integrity'] = {
                'status': 'failed', 
                'error': str(e)
            }
            self.logger.error(f"Dataset validation: ❌ FAILED ({e})")
            return False
    
    def validate_experimental_results(self) -> bool:
        """Validate experimental results completeness."""
        self.logger.info("Validating experimental results...")
        
        baseline_dir = self.results_dir / 'baselines'
        lethe_dir = self.results_dir / 'lethe_runs'
        
        baseline_count = 0
        lethe_count = 0
        result_validation = True
        
        # Count baseline results
        if baseline_dir.exists():
            baseline_files = list(baseline_dir.rglob('*.json'))
            baseline_count = len(baseline_files)
            
            # Validate a sample of baseline files
            sample_size = min(5, len(baseline_files))
            for result_file in baseline_files[:sample_size]:
                if not self._validate_result_file(result_file):
                    result_validation = False
        else:
            self.validation_results['warnings'].append("Baseline results directory not found")
        
        # Count Lethe results
        if lethe_dir.exists():
            lethe_files = list(lethe_dir.rglob('*.json'))
            lethe_count = len(lethe_files)
            
            # Validate a sample of Lethe files
            sample_size = min(5, len(lethe_files))
            for result_file in lethe_files[:sample_size]:
                if not self._validate_result_file(result_file):
                    result_validation = False
        else:
            self.validation_results['warnings'].append("Lethe results directory not found")
        
        # Check minimum requirements
        if baseline_count < 5:
            self.validation_results['warnings'].append(f"Low baseline count: {baseline_count}")
        
        if lethe_count < 10:
            self.validation_results['warnings'].append(f"Low Lethe configuration count: {lethe_count}")
        
        # Record results
        self.validation_results['checks']['experimental_results'] = {
            'status': 'passed' if result_validation else 'failed',
            'baseline_count': baseline_count,
            'lethe_count': lethe_count,
            'total_configurations': baseline_count + lethe_count
        }
        
        self.logger.info(f"Experimental results: {'✅ PASSED' if result_validation else '❌ FAILED'} " +
                        f"({baseline_count} baselines, {lethe_count} lethe configs)")
        
        return result_validation
    
    def _validate_result_file(self, result_file: Path) -> bool:
        """Validate individual result file format."""
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = ['config', 'results']
            if not all(field in data for field in required_fields):
                return False
            
            # Check results contain expected metrics
            results = data.get('results', {})
            expected_metrics = ['ndcg_at_10', 'recall_at_10', 'mrr_at_10']
            
            missing_metrics = [metric for metric in expected_metrics if metric not in results]
            if missing_metrics:
                self.validation_results['warnings'].append(
                    f"File {result_file.name} missing metrics: {missing_metrics}"
                )
            
            return True
            
        except Exception as e:
            self.validation_results['warnings'].append(f"Failed to validate {result_file.name}: {e}")
            return False
    
    def validate_statistical_analysis(self) -> bool:
        """Validate statistical analysis results."""
        self.logger.info("Validating statistical analysis...")
        
        summary_path = self.results_dir / 'analysis' / 'summary_report.json'
        
        if not summary_path.exists():
            self.validation_results['errors'].append("Statistical analysis summary not found")
            self.validation_results['checks']['statistical_analysis'] = {
                'status': 'failed', 
                'reason': 'summary_not_found'
            }
            return False
        
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            
            # Validate hypothesis testing results
            hypothesis_results = summary.get('hypothesis_results', {})
            if not hypothesis_results:
                self.validation_results['warnings'].append("No hypothesis testing results found")
            
            # Count supported hypotheses
            supported_hypotheses = sum(1 for result in hypothesis_results.values() 
                                     if result.get('significant', False))
            total_hypotheses = len(hypothesis_results)
            
            # Check for statistical rigor indicators
            has_effect_sizes = 'statistical_analysis' in summary and 'effect_sizes' in summary['statistical_analysis']
            has_confidence_intervals = 'statistical_analysis' in summary and 'confidence_intervals' in summary['statistical_analysis']
            
            # Validate publication readiness
            pub_readiness = summary.get('publication_readiness', {})
            readiness_score = pub_readiness.get('overall_score', 0.0)
            
            # Record results
            self.validation_results['checks']['statistical_analysis'] = {
                'status': 'passed',
                'supported_hypotheses': supported_hypotheses,
                'total_hypotheses': total_hypotheses,
                'hypothesis_support_rate': supported_hypotheses / max(total_hypotheses, 1),
                'has_effect_sizes': has_effect_sizes,
                'has_confidence_intervals': has_confidence_intervals,
                'publication_readiness_score': readiness_score
            }
            
            # Check publication readiness
            if readiness_score < 0.6:
                self.validation_results['warnings'].append(f"Low publication readiness score: {readiness_score:.2f}")
            
            self.logger.info(f"Statistical analysis: ✅ PASSED ({supported_hypotheses}/{total_hypotheses} hypotheses supported)")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Statistical analysis validation failed: {e}")
            self.validation_results['checks']['statistical_analysis'] = {
                'status': 'failed', 
                'error': str(e)
            }
            self.logger.error(f"Statistical analysis: ❌ FAILED ({e})")
            return False
    
    def validate_figures(self) -> bool:
        """Validate figure generation and quality."""
        self.logger.info("Validating figures...")
        
        figures_dir = self.results_dir / 'figures'
        manifest_path = figures_dir / 'figure_manifest.json'
        
        if not figures_dir.exists():
            self.validation_results['errors'].append("Figures directory not found")
            self.validation_results['checks']['figures'] = {
                'status': 'failed', 
                'reason': 'directory_not_found'
            }
            return False
        
        if not manifest_path.exists():
            self.validation_results['warnings'].append("Figure manifest not found")
            # Count figures directly
            figure_files = list(figures_dir.glob('*.pdf')) + list(figures_dir.glob('*.png'))
            figure_count = len(figure_files)
        else:
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                
                figure_count = manifest.get('metadata', {}).get('total_figures', 0)
                
                # Validate figure files exist
                missing_figures = []
                for category_data in manifest.get('figures', {}).values():
                    for figure_name, figure_info in category_data.items():
                        figure_path = Path(figure_info['file_path'])
                        if not figure_path.exists():
                            missing_figures.append(figure_name)
                
                if missing_figures:
                    self.validation_results['warnings'].append(f"Missing figure files: {missing_figures}")
                
            except Exception as e:
                self.validation_results['warnings'].append(f"Failed to load figure manifest: {e}")
                figure_count = len(list(figures_dir.glob('*.pdf')) + list(figures_dir.glob('*.png')))
        
        # Check minimum figure requirements
        min_required_figures = 4
        figures_adequate = figure_count >= min_required_figures
        
        if not figures_adequate:
            self.validation_results['warnings'].append(f"Insufficient figures: {figure_count} (minimum: {min_required_figures})")
        
        # Record results
        self.validation_results['checks']['figures'] = {
            'status': 'passed' if figures_adequate else 'warning',
            'figure_count': figure_count,
            'has_manifest': manifest_path.exists(),
            'adequate_count': figures_adequate
        }
        
        self.logger.info(f"Figures validation: {'✅ PASSED' if figures_adequate else '⚠️  WARNING'} ({figure_count} figures)")
        return figures_adequate
    
    def validate_paper_generation(self) -> bool:
        """Validate paper generation if paper directory exists."""
        self.logger.info("Validating paper generation...")
        
        paper_dir = self.results_dir / 'paper'
        
        if not paper_dir.exists():
            self.validation_results['warnings'].append("Paper directory not found")
            self.validation_results['checks']['paper'] = {
                'status': 'skipped', 
                'reason': 'directory_not_found'
            }
            return True  # Not a failure if paper generation wasn't run
        
        # Check for essential paper files
        latex_file = paper_dir / 'lethe_paper.tex'
        bib_file = paper_dir / 'references.bib'
        pdf_file = paper_dir / 'lethe_paper.pdf'
        
        has_latex = latex_file.exists()
        has_bibliography = bib_file.exists()
        has_pdf = pdf_file.exists()
        
        # Record results
        self.validation_results['checks']['paper'] = {
            'status': 'passed' if has_latex else 'failed',
            'has_latex_source': has_latex,
            'has_bibliography': has_bibliography,
            'has_compiled_pdf': has_pdf
        }
        
        if not has_latex:
            self.validation_results['errors'].append("LaTeX source file not found")
            
        if not has_bibliography:
            self.validation_results['warnings'].append("Bibliography file not found")
        
        status = '✅ PASSED' if has_latex else '❌ FAILED'
        self.logger.info(f"Paper validation: {status}")
        
        return has_latex
    
    def assess_publication_readiness(self) -> Dict[str, Any]:
        """Comprehensive publication readiness assessment."""
        self.logger.info("Assessing publication readiness...")
        
        # Scoring criteria with weights
        criteria = {
            'directory_structure': {'weight': 0.10, 'score': 0.0},
            'dataset_integrity': {'weight': 0.15, 'score': 0.0},
            'experimental_results': {'weight': 0.25, 'score': 0.0},
            'statistical_analysis': {'weight': 0.30, 'score': 0.0},
            'figures': {'weight': 0.15, 'score': 0.0},
            'paper': {'weight': 0.05, 'score': 0.0}
        }
        
        # Calculate scores for each criterion
        for criterion, info in criteria.items():
            check_result = self.validation_results['checks'].get(criterion, {})
            status = check_result.get('status', 'failed')
            
            if status == 'passed':
                info['score'] = 1.0
            elif status == 'warning':
                info['score'] = 0.7
            elif status == 'skipped':
                info['score'] = 1.0  # Don't penalize for optional components
            else:
                info['score'] = 0.0
        
        # Calculate weighted overall score
        total_score = sum(info['weight'] * info['score'] for info in criteria.values())
        
        # Determine readiness status
        if total_score >= 0.8:
            readiness_status = 'ready'
            readiness_message = "Results are publication-ready"
        elif total_score >= 0.6:
            readiness_status = 'nearly_ready' 
            readiness_message = "Minor issues need to be addressed"
        elif total_score >= 0.4:
            readiness_status = 'needs_work'
            readiness_message = "Significant work needed before publication"
        else:
            readiness_status = 'not_ready'
            readiness_message = "Extensive work required"
        
        # Identify blocking issues
        blocking_issues = []
        for error in self.validation_results['errors']:
            blocking_issues.append(error)
        
        # Generate recommendations
        recommendations = []
        
        if criteria['experimental_results']['score'] < 0.8:
            recommendations.append("Increase number of experimental configurations")
        
        if criteria['statistical_analysis']['score'] < 0.8:
            recommendations.append("Strengthen statistical validation and hypothesis testing")
        
        if criteria['figures']['score'] < 0.8:
            recommendations.append("Generate additional publication-quality figures")
        
        if len(self.validation_results['warnings']) > 5:
            recommendations.append("Address validation warnings for improved quality")
        
        # Update publication readiness in validation results
        self.validation_results['publication_readiness'] = {
            'overall_score': total_score,
            'status': readiness_status,
            'message': readiness_message,
            'criteria_scores': {k: v['score'] for k, v in criteria.items()},
            'blocking_issues': blocking_issues,
            'recommendations': recommendations
        }
        
        self.logger.info(f"Publication readiness: {readiness_status.upper()} (score: {total_score:.2f})")
        
        return self.validation_results['publication_readiness']
    
    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        self.logger.info("Generating validation report...")
        
        # Determine overall status
        error_count = len(self.validation_results['errors'])
        warning_count = len(self.validation_results['warnings'])
        
        if error_count == 0:
            self.validation_results['overall_status'] = 'passed'
        elif error_count > 0:
            self.validation_results['overall_status'] = 'failed'
        else:
            self.validation_results['overall_status'] = 'warning'
        
        # Save validation results
        validation_file = self.results_dir / 'validation_report.json'
        with open(validation_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Generate human-readable summary
        summary_lines = [
            "# Results Validation Report",
            f"**Generated**: {self.validation_results['validation_timestamp']}",
            f"**Overall Status**: {self.validation_results['overall_status'].upper()}",
            f"**Errors**: {error_count}",
            f"**Warnings**: {warning_count}",
            "",
            "## Validation Checks",
            ""
        ]
        
        for check_name, result in self.validation_results['checks'].items():
            status = result.get('status', 'unknown')
            status_emoji = {'passed': '✅', 'failed': '❌', 'warning': '⚠️', 'skipped': '⏭️'}.get(status, '❓')
            
            summary_lines.append(f"- **{check_name.replace('_', ' ').title()}**: {status_emoji} {status.upper()}")
        
        if self.validation_results['errors']:
            summary_lines.extend([
                "",
                "## ❌ Errors",
                ""
            ])
            for error in self.validation_results['errors']:
                summary_lines.append(f"- {error}")
        
        if self.validation_results['warnings']:
            summary_lines.extend([
                "",
                "## ⚠️  Warnings", 
                ""
            ])
            for warning in self.validation_results['warnings']:
                summary_lines.append(f"- {warning}")
        
        # Publication readiness summary
        pub_readiness = self.validation_results['publication_readiness']
        summary_lines.extend([
            "",
            "## 📊 Publication Readiness",
            "",
            f"**Score**: {pub_readiness['overall_score']:.2f}/1.00",
            f"**Status**: {pub_readiness['status'].replace('_', ' ').title()}",
            f"**Message**: {pub_readiness['message']}",
            ""
        ])
        
        if pub_readiness['recommendations']:
            summary_lines.extend([
                "### Recommendations",
                ""
            ])
            for rec in pub_readiness['recommendations']:
                summary_lines.append(f"- {rec}")
        
        # Save human-readable summary
        summary_file = self.results_dir / 'VALIDATION_SUMMARY.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"Validation report saved: {validation_file}")
        self.logger.info(f"Human-readable summary: {summary_file}")
    
    def run_validation(self) -> bool:
        """Run complete validation pipeline."""
        self.logger.info("🔍 Starting comprehensive results validation...")
        
        # Run all validation checks
        validation_checks = [
            self.validate_directory_structure,
            self.validate_dataset_integrity,
            self.validate_experimental_results,
            self.validate_statistical_analysis,
            self.validate_figures,
            self.validate_paper_generation
        ]
        
        validation_results = []
        for check in validation_checks:
            try:
                result = check()
                validation_results.append(result)
            except Exception as e:
                self.logger.error(f"Validation check failed: {e}")
                validation_results.append(False)
                self.validation_results['errors'].append(f"Validation check failed: {e}")
        
        # Assess publication readiness
        self.assess_publication_readiness()
        
        # Generate comprehensive report
        self.generate_validation_report()
        
        # Determine overall success
        overall_success = all(validation_results) and len(self.validation_results['errors']) == 0
        
        # Print summary
        error_count = len(self.validation_results['errors'])
        warning_count = len(self.validation_results['warnings'])
        pub_score = self.validation_results['publication_readiness']['overall_score']
        
        self.logger.info("")
        self.logger.info("🏁 Validation Summary:")
        self.logger.info(f"   Overall Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        self.logger.info(f"   Errors: {error_count}")
        self.logger.info(f"   Warnings: {warning_count}")
        self.logger.info(f"   Publication Score: {pub_score:.2f}/1.00")
        self.logger.info(f"   Publication Status: {self.validation_results['publication_readiness']['status'].upper()}")
        
        return overall_success

def main():
    parser = argparse.ArgumentParser(description='Validate experimental results')
    parser.add_argument('--results-dir', type=Path, required=True,
                       help='Directory containing experimental results')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Create validator
        validator = ResultsValidator(
            results_dir=args.results_dir,
            logger=logger
        )
        
        # Run validation
        success = validator.run_validation()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()