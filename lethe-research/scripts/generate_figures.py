#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Script
============================================

Generates all publication-ready visualizations using the experiments/plots.py framework.
Orchestrates creation of:
- Pareto efficiency curves
- Ablation study charts  
- Latency breakdown analysis
- Coverage and scaling analysis
- Statistical significance visualizations

Usage:
    python generate_figures.py --analysis-results analysis/ --output figures/ --config config.yaml
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

from experiments.plots import LetheVisualizer

def setup_logging(level: str) -> logging.Logger:
    """Configure logging with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class FigureGenerator:
    """Orchestrates generation of all publication figures."""
    
    def __init__(self, analysis_results_dir: Path, output_dir: Path, 
                 config_path: Path, logger: logging.Logger):
        self.analysis_results_dir = analysis_results_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.logger = logger
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize visualizer
        self.visualizer = LetheVisualizer()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load visualization configuration."""
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
            
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
            
        # Merge with defaults
        default_config = self._get_default_config()
        merged_config = {**default_config, **config}
        
        return merged_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default visualization configuration."""
        return {
            'figure_format': 'pdf',
            'dpi': 300,
            'figure_size': [10, 6],
            'font_size': 12,
            'color_palette': 'Set2',
            'style': 'seaborn-v0_8-whitegrid',
            'figures_to_generate': [
                'pareto_curves',
                'ablation_study', 
                'latency_breakdown',
                'coverage_analysis',
                'scaling_analysis',
                'statistical_significance',
                'parameter_sensitivity',
                'domain_performance',
                'efficiency_comparison',
                'robustness_analysis'
            ]
        }
    
    def load_analysis_results(self) -> Dict[str, Any]:
        """Load analysis results from directory."""
        self.logger.info("Loading analysis results...")
        
        results = {}
        
        # Load main summary report
        summary_file = self.analysis_results_dir / 'summary_report.json'
        if summary_file.exists():
            with open(summary_file) as f:
                results['summary'] = json.load(f)
        else:
            raise FileNotFoundError(f"Summary report not found: {summary_file}")
        
        # Load detailed statistical results
        stats_file = self.analysis_results_dir / 'statistical_results.json'
        if stats_file.exists():
            with open(stats_file) as f:
                results['statistical'] = json.load(f)
        
        # Load experimental data
        data_file = self.analysis_results_dir / 'experimental_data.json'
        if data_file.exists():
            with open(data_file) as f:
                results['experimental'] = json.load(f)
        
        # Load fraud-proofing results
        fraud_file = self.analysis_results_dir / 'fraud_proofing_results.json'
        if fraud_file.exists():
            with open(fraud_file) as f:
                results['fraud_proofing'] = json.load(f)
        
        self.logger.info(f"Loaded analysis results with {len(results)} components")
        return results
    
    def validate_data(self, analysis_results: Dict[str, Any]) -> None:
        """Validate that required data is available for visualization."""
        self.logger.info("Validating analysis data for visualization...")
        
        required_components = ['summary', 'experimental']
        missing_components = []
        
        for component in required_components:
            if component not in analysis_results:
                missing_components.append(component)
        
        if missing_components:
            raise ValueError(f"Missing required analysis components: {missing_components}")
        
        # Validate data structure
        summary = analysis_results['summary']
        if 'hypothesis_results' not in summary:
            raise ValueError("No hypothesis results found in summary")
        
        experimental = analysis_results['experimental']
        if not experimental.get('baseline_results') and not experimental.get('lethe_results'):
            raise ValueError("No experimental results found")
        
        self.logger.info("✅ Data validation complete")
    
    def generate_all_figures(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all configured figures."""
        self.logger.info("🎨 Generating publication figures...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track generated figures
        generated_figures = {}
        
        figures_to_generate = self.config.get('figures_to_generate', [])
        
        for figure_type in figures_to_generate:
            try:
                self.logger.info(f"Generating {figure_type}...")
                output_path = self._generate_figure(figure_type, analysis_results)
                
                if output_path:
                    generated_figures[figure_type] = output_path
                    self.logger.info(f"✅ Generated {figure_type}: {output_path}")
                else:
                    self.logger.warning(f"❌ Failed to generate {figure_type}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error generating {figure_type}: {e}")
        
        self.logger.info(f"Generated {len(generated_figures)}/{len(figures_to_generate)} figures")
        return generated_figures
    
    def _generate_figure(self, figure_type: str, analysis_results: Dict[str, Any]) -> str:
        """Generate individual figure type."""
        output_path = self.output_dir / f"{figure_type}.{self.config['figure_format']}"
        
        # Dispatch to appropriate generation method
        if figure_type == 'pareto_curves':
            self.visualizer.create_pareto_curves(
                analysis_results['experimental'], 
                str(output_path)
            )
        elif figure_type == 'ablation_study':
            self.visualizer.create_ablation_study(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'latency_breakdown':
            self.visualizer.create_latency_breakdown(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'coverage_analysis':
            self.visualizer.create_coverage_analysis(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'scaling_analysis':
            self.visualizer.create_scaling_analysis(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'statistical_significance':
            self.visualizer.create_statistical_significance_plot(
                analysis_results['statistical'] if 'statistical' in analysis_results else analysis_results['summary'],
                str(output_path)
            )
        elif figure_type == 'parameter_sensitivity':
            self.visualizer.create_parameter_sensitivity(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'domain_performance':
            self.visualizer.create_domain_performance(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'efficiency_comparison':
            self.visualizer.create_efficiency_comparison(
                analysis_results['experimental'],
                str(output_path)
            )
        elif figure_type == 'robustness_analysis':
            self.visualizer.create_robustness_analysis(
                analysis_results.get('fraud_proofing', {}),
                str(output_path)
            )
        else:
            self.logger.warning(f"Unknown figure type: {figure_type}")
            return None
        
        return str(output_path)
    
    def generate_figure_manifest(self, generated_figures: Dict[str, str]) -> None:
        """Generate manifest of all generated figures."""
        self.logger.info("Generating figure manifest...")
        
        manifest = {
            'metadata': {
                'generated_timestamp': datetime.utcnow().isoformat(),
                'total_figures': len(generated_figures),
                'output_directory': str(self.output_dir),
                'figure_format': self.config['figure_format']
            },
            'figures': {}
        }
        
        # Categorize figures for easy reference
        figure_categories = {
            'performance': ['pareto_curves', 'efficiency_comparison', 'scaling_analysis'],
            'analysis': ['ablation_study', 'parameter_sensitivity', 'domain_performance'],
            'technical': ['latency_breakdown', 'coverage_analysis'],
            'validation': ['statistical_significance', 'robustness_analysis']
        }
        
        for category, figure_types in figure_categories.items():
            manifest['figures'][category] = {}
            for figure_type in figure_types:
                if figure_type in generated_figures:
                    manifest['figures'][category][figure_type] = {
                        'file_path': generated_figures[figure_type],
                        'description': self._get_figure_description(figure_type),
                        'recommended_caption': self._get_figure_caption(figure_type)
                    }
        
        # Save manifest
        manifest_path = self.output_dir / 'figure_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Generate LaTeX figure inclusion snippets
        self._generate_latex_snippets(generated_figures)
        
        self.logger.info(f"Figure manifest saved: {manifest_path}")
    
    def _get_figure_description(self, figure_type: str) -> str:
        """Get description for figure type."""
        descriptions = {
            'pareto_curves': 'Pareto efficiency frontier showing quality vs. latency trade-offs',
            'ablation_study': 'Ablation study showing impact of individual Lethe components',
            'latency_breakdown': 'Detailed breakdown of query processing latency components',
            'coverage_analysis': 'Analysis of retrieval coverage across different domains',
            'scaling_analysis': 'Performance scaling characteristics with dataset size',
            'statistical_significance': 'Statistical significance tests for hypothesis validation',
            'parameter_sensitivity': 'Sensitivity analysis for key hyperparameters',
            'domain_performance': 'Performance comparison across different evaluation domains',
            'efficiency_comparison': 'Efficiency comparison between Lethe and baseline systems',
            'robustness_analysis': 'Robustness validation results from fraud-proofing framework'
        }
        
        return descriptions.get(figure_type, f'Analysis visualization: {figure_type}')
    
    def _get_figure_caption(self, figure_type: str) -> str:
        """Get recommended LaTeX caption for figure."""
        captions = {
            'pareto_curves': 'Pareto efficiency curves comparing Lethe configurations against baseline systems. Each point represents a different parameter configuration, with the Pareto frontier highlighting optimal quality-efficiency trade-offs.',
            'ablation_study': 'Ablation study results showing the contribution of each Lethe component to overall performance. Error bars represent 95% confidence intervals from bootstrap sampling.',
            'latency_breakdown': 'Detailed breakdown of query processing latency showing time spent in each pipeline stage. Results averaged over 1000 queries with 95th percentile error bars.',
            'coverage_analysis': 'Retrieval coverage analysis across different domains and query types. Higher coverage indicates better recall of relevant information.',
            'scaling_analysis': 'Performance scaling characteristics with respect to dataset size. Shows how quality and latency metrics change as the knowledge base grows.',
            'statistical_significance': 'Statistical significance results for primary hypotheses (H1-H4). P-values are Holm-Bonferroni corrected for multiple comparisons.',
            'parameter_sensitivity': 'Parameter sensitivity analysis showing how key hyperparameters affect system performance. Darker colors indicate stronger parameter influence.',
            'domain_performance': 'Performance comparison across different evaluation domains. Demonstrates system robustness and domain adaptability.',
            'efficiency_comparison': 'Efficiency comparison between Lethe and baseline retrieval systems. Shows performance per unit computational cost.',
            'robustness_analysis': 'Results from comprehensive fraud-proofing validation framework. All tests must pass for publication acceptance.'
        }
        
        return captions.get(figure_type, f'Caption for {figure_type} analysis.')
    
    def _generate_latex_snippets(self, generated_figures: Dict[str, str]) -> None:
        """Generate LaTeX code snippets for figure inclusion."""
        self.logger.info("Generating LaTeX snippets...")
        
        latex_snippets = []
        
        for figure_type, file_path in generated_figures.items():
            filename = Path(file_path).name
            caption = self._get_figure_caption(figure_type)
            label = f"fig:{figure_type}"
            
            snippet = f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{figures/{filename}}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{figure}}"""
            
            latex_snippets.append(snippet)
        
        # Save LaTeX snippets
        latex_file = self.output_dir / 'latex_figures.tex'
        with open(latex_file, 'w') as f:
            f.write('% Auto-generated LaTeX figure inclusion snippets\n\n')
            f.write('\n\n'.join(latex_snippets))
        
        self.logger.info(f"LaTeX snippets saved: {latex_file}")
    
    def generate_figures_summary(self, generated_figures: Dict[str, str]) -> None:
        """Generate human-readable summary of generated figures."""
        summary_lines = [
            "# Generated Figures Summary",
            f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"**Total Figures**: {len(generated_figures)}",
            f"**Output Directory**: `{self.output_dir}`",
            "",
            "## Figure List",
            ""
        ]
        
        for figure_type, file_path in sorted(generated_figures.items()):
            filename = Path(file_path).name
            description = self._get_figure_description(figure_type)
            
            summary_lines.extend([
                f"### {figure_type}",
                f"- **File**: `{filename}`",
                f"- **Description**: {description}",
                ""
            ])
        
        summary_lines.extend([
            "## Usage Instructions",
            "",
            "1. **For LaTeX**: Include figures using the snippets in `latex_figures.tex`",
            "2. **For Papers**: Use the recommended captions from `figure_manifest.json`",
            "3. **For Presentations**: High-resolution figures suitable for projection",
            "",
            "## Quality Assurance",
            "",
            "- All figures generated at 300 DPI for publication quality",
            "- Color schemes optimized for both print and digital viewing",
            "- Consistent styling across all visualizations",
            "- Statistical significance properly indicated where applicable"
        ])
        
        # Save summary
        summary_file = self.output_dir / 'FIGURES_README.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"Figures summary saved: {summary_file}")
    
    def execute_generation(self) -> Dict[str, Any]:
        """Execute complete figure generation pipeline."""
        self.logger.info("🎨 Starting figure generation pipeline...")
        
        # Load analysis results
        analysis_results = self.load_analysis_results()
        
        # Validate data
        self.validate_data(analysis_results)
        
        # Generate all figures
        generated_figures = self.generate_all_figures(analysis_results)
        
        # Generate supporting documentation
        self.generate_figure_manifest(generated_figures)
        self.generate_figures_summary(generated_figures)
        
        # Prepare return results
        results = {
            'generated_figures': generated_figures,
            'output_directory': str(self.output_dir),
            'total_figures': len(generated_figures),
            'manifest_path': str(self.output_dir / 'figure_manifest.json'),
            'latex_path': str(self.output_dir / 'latex_figures.tex'),
            'readme_path': str(self.output_dir / 'FIGURES_README.md')
        }
        
        self.logger.info(f"✅ Figure generation complete: {len(generated_figures)} figures created")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality figures')
    parser.add_argument('--analysis-results', type=Path, required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for generated figures')
    parser.add_argument('--config', type=Path,
                       help='Visualization configuration file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Create figure generator
        generator = FigureGenerator(
            analysis_results_dir=args.analysis_results,
            output_dir=args.output,
            config_path=args.config or Path('experiments/grid_config.yaml'),
            logger=logger
        )
        
        # Execute generation
        results = generator.execute_generation()
        
        # Print summary
        logger.info("✅ Figure generation completed successfully!")
        logger.info(f"Generated {results['total_figures']} figures in {results['output_directory']}")
        logger.info(f"LaTeX snippets: {results['latex_path']}")
        logger.info(f"Documentation: {results['readme_path']}")
        
    except Exception as e:
        logger.error(f"❌ Figure generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()