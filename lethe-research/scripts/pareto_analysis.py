#!/usr/bin/env python3
"""
Pareto Frontier Analysis for Lethe Multi-Objective Optimization
===============================================================

Implements multi-objective Pareto analysis for:
- nDCG@10 vs p95 latency vs memory usage
- Budget constraint analysis
- Trade-off visualization
- Non-dominated configuration identification

Key Features:
- Multi-objective Pareto frontier computation
- Budget constraint enforcement
- Trade-off visualization
- Publication-quality plots
- Configuration ranking
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


class ParetoFrontierAnalyzer:
    """Multi-objective Pareto frontier analyzer for IR system optimization"""
    
    def __init__(self, output_dir: Path = Path("analysis")):
        """
        Initialize Pareto analyzer
        
        Args:
            output_dir: Directory for output files and figures
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Objective definitions (name, direction, weight)
        # direction: 1 for maximize, -1 for minimize
        self.objectives = {
            'ndcg_at_10': {'direction': 1, 'weight': 1.0, 'label': 'nDCG@10', 'unit': ''},
            'latency_p95': {'direction': -1, 'weight': 1.0, 'label': 'P95 Latency', 'unit': 'ms'},
            'memory_peak_mb': {'direction': -1, 'weight': 1.0, 'label': 'Peak Memory', 'unit': 'MB'}
        }
        
        # Budget constraints
        self.budget_constraints = {
            'latency_p95_budget': 3000,  # ms
            'memory_peak_budget': 1500,  # MB
            'quality_floor': 0.5  # minimum nDCG@10
        }
        
        print(f"Initialized Pareto Frontier Analyzer")
        print(f"Objectives: {list(self.objectives.keys())}")
        print(f"Output directory: {self.output_dir}")
    
    def is_dominated(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """
        Check if point1 is dominated by point2
        
        Args:
            point1: First point (configuration)
            point2: Second point (configuration)
            
        Returns:
            True if point1 is dominated by point2
        """
        # Transform objectives (maximize -> positive, minimize -> negative)
        directions = np.array([obj['direction'] for obj in self.objectives.values()])
        
        transformed_p1 = point1 * directions
        transformed_p2 = point2 * directions
        
        # point1 is dominated if point2 is better or equal in all objectives
        # and strictly better in at least one
        better_or_equal = np.all(transformed_p2 >= transformed_p1)
        strictly_better = np.any(transformed_p2 > transformed_p1)
        
        return better_or_equal and strictly_better
    
    def find_pareto_frontier(self, points: np.ndarray, 
                           labels: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Find Pareto frontier (non-dominated points)
        
        Args:
            points: Array of points (n_points, n_objectives)
            labels: Optional labels for points
            
        Returns:
            Tuple of (pareto_points, pareto_indices)
        """
        n_points = points.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j and is_pareto[i]:
                    if self.is_dominated(points[i], points[j]):
                        is_pareto[i] = False
                        break
        
        pareto_indices = np.where(is_pareto)[0].tolist()
        pareto_points = points[is_pareto]
        
        return pareto_points, pareto_indices
    
    def apply_budget_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply budget constraints to filter feasible configurations
        
        Args:
            df: Configuration dataframe
            
        Returns:
            Filtered dataframe with feasible configurations
        """
        feasible = df.copy()
        
        # Latency constraint
        if 'latency_p95' in feasible.columns:
            latency_feasible = feasible['latency_p95'] <= self.budget_constraints['latency_p95_budget']
            feasible = feasible[latency_feasible]
        elif 'latency_ms_total' in feasible.columns:
            # Use total latency as proxy for P95
            latency_feasible = feasible['latency_ms_total'] <= self.budget_constraints['latency_p95_budget']
            feasible = feasible[latency_feasible]
        
        # Memory constraint
        if 'memory_peak_mb' in feasible.columns:
            memory_feasible = feasible['memory_peak_mb'] <= self.budget_constraints['memory_peak_budget']
            feasible = feasible[memory_feasible]
        
        # Quality floor constraint
        if 'ndcg_at_10' in feasible.columns:
            quality_feasible = feasible['ndcg_at_10'] >= self.budget_constraints['quality_floor']
            feasible = feasible[quality_feasible]
        
        print(f"Budget constraints: {len(df)} -> {len(feasible)} feasible configurations")
        
        return feasible
    
    def compute_hypervolume(self, pareto_points: np.ndarray, 
                          reference_point: Optional[np.ndarray] = None) -> float:
        """
        Compute hypervolume indicator for Pareto frontier
        
        Args:
            pareto_points: Pareto frontier points
            reference_point: Reference point (nadir point)
            
        Returns:
            Hypervolume value
        """
        if len(pareto_points) == 0:
            return 0.0
        
        # Transform for maximization
        directions = np.array([obj['direction'] for obj in self.objectives.values()])
        transformed_points = pareto_points * directions
        
        if reference_point is None:
            # Use worst values as reference point
            reference_point = np.min(transformed_points, axis=0) - 1e-6
        else:
            reference_point = reference_point * directions
        
        # Simple hypervolume computation for 3D (can be extended)
        if transformed_points.shape[1] == 2:
            # Sort by first objective
            sorted_indices = np.argsort(transformed_points[:, 0])
            sorted_points = transformed_points[sorted_indices]
            
            hypervolume = 0.0
            prev_x = reference_point[0]
            
            for point in sorted_points:
                width = point[0] - prev_x
                height = point[1] - reference_point[1]
                hypervolume += width * height
                prev_x = point[0]
            
            return hypervolume
        
        elif transformed_points.shape[1] == 3:
            # Simplified 3D hypervolume (Monte Carlo approximation)
            n_samples = 100000
            
            # Define bounding box
            min_bounds = reference_point
            max_bounds = np.max(transformed_points, axis=0)
            
            # Generate random points
            random_points = np.random.uniform(
                min_bounds, max_bounds, size=(n_samples, 3)
            )
            
            # Check domination
            dominated_count = 0
            for sample in random_points:
                dominated = False
                for pareto_point in transformed_points:
                    if np.all(pareto_point >= sample):
                        dominated = True
                        break
                if dominated:
                    dominated_count += 1
            
            # Estimate hypervolume
            box_volume = np.prod(max_bounds - min_bounds)
            hypervolume = (dominated_count / n_samples) * box_volume
            
            return hypervolume
        
        else:
            # Fallback for higher dimensions
            return len(pareto_points)  # Just count non-dominated solutions
    
    def rank_configurations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank configurations using multiple criteria
        
        Args:
            df: Configuration dataframe
            
        Returns:
            Dataframe with ranking information
        """
        ranked_df = df.copy()
        
        # Extract objective values
        objective_cols = [col for col in self.objectives.keys() if col in df.columns]
        
        if not objective_cols:
            print("Warning: No objective columns found for ranking")
            return ranked_df
        
        # Handle missing columns with fallbacks
        if 'latency_p95' not in objective_cols and 'latency_ms_total' in df.columns:
            ranked_df['latency_p95'] = ranked_df['latency_ms_total'] * 1.5  # Approximate P95
            objective_cols.append('latency_p95')
        
        if 'memory_peak_mb' not in objective_cols:
            # Estimate memory from latency (rough heuristic)
            if 'latency_ms_total' in df.columns:
                ranked_df['memory_peak_mb'] = 500 + ranked_df['latency_ms_total'] * 0.3
                objective_cols.append('memory_peak_mb')
        
        objective_data = ranked_df[objective_cols].values
        
        # Find Pareto frontier
        pareto_points, pareto_indices = self.find_pareto_frontier(objective_data)
        
        # Mark Pareto optimal configurations
        ranked_df['is_pareto_optimal'] = False
        ranked_df.iloc[pareto_indices, ranked_df.columns.get_loc('is_pareto_optimal')] = True
        
        # Compute composite scores
        scaler = MinMaxScaler()
        normalized_objectives = scaler.fit_transform(objective_data)
        
        # Apply direction weights
        directions = np.array([self.objectives[col]['direction'] for col in objective_cols])
        weighted_objectives = normalized_objectives * directions
        
        # Simple additive utility
        ranked_df['composite_score'] = np.mean(weighted_objectives, axis=1)
        
        # Distance to ideal point
        ideal_point = np.max(weighted_objectives, axis=0)
        distances = cdist(weighted_objectives, ideal_point.reshape(1, -1), metric='euclidean')
        ranked_df['distance_to_ideal'] = distances.flatten()
        
        # Final ranking (Pareto optimal first, then by composite score)
        ranked_df = ranked_df.sort_values([
            'is_pareto_optimal',
            'composite_score'
        ], ascending=[False, False])
        
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        return ranked_df
    
    def analyze_tradeoffs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trade-offs between objectives
        
        Args:
            df: Configuration dataframe
            
        Returns:
            Trade-off analysis results
        """
        objective_cols = [col for col in self.objectives.keys() if col in df.columns]
        
        if len(objective_cols) < 2:
            print("Warning: Need at least 2 objectives for trade-off analysis")
            return {}
        
        # Handle missing columns
        if 'latency_p95' not in objective_cols and 'latency_ms_total' in df.columns:
            df = df.copy()
            df['latency_p95'] = df['latency_ms_total'] * 1.5
            objective_cols.append('latency_p95')
        
        if 'memory_peak_mb' not in objective_cols and 'latency_ms_total' in df.columns:
            df = df.copy()
            df['memory_peak_mb'] = 500 + df['latency_ms_total'] * 0.3
            objective_cols.append('memory_peak_mb')
        
        objective_data = df[objective_cols].values
        pareto_points, pareto_indices = self.find_pareto_frontier(objective_data)
        
        # Correlation analysis
        correlations = {}
        for i, obj1 in enumerate(objective_cols):
            for j, obj2 in enumerate(objective_cols):
                if i < j:
                    corr = np.corrcoef(df[obj1], df[obj2])[0, 1]
                    correlations[f"{obj1}_vs_{obj2}"] = float(corr)
        
        # Hypervolume
        hypervolume = self.compute_hypervolume(pareto_points)
        
        # Pareto frontier spread
        if len(pareto_points) > 1:
            pareto_spread = np.std(pareto_points, axis=0).mean()
        else:
            pareto_spread = 0.0
        
        return {
            'pareto_frontier_size': len(pareto_indices),
            'total_configurations': len(df),
            'pareto_efficiency': len(pareto_indices) / len(df),
            'hypervolume': float(hypervolume),
            'pareto_spread': float(pareto_spread),
            'objective_correlations': correlations,
            'pareto_indices': pareto_indices,
            'objective_ranges': {
                col: {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                } for col in objective_cols
            }
        }
    
    def create_pareto_plots(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                          title_prefix: str = "Lethe") -> List[Path]:
        """
        Create publication-quality Pareto frontier plots
        
        Args:
            df: Configuration dataframe with rankings
            analysis_results: Trade-off analysis results
            title_prefix: Prefix for plot titles
            
        Returns:
            List of generated plot file paths
        """
        plot_files = []
        
        # Set up publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Handle missing columns
        df = df.copy()
        if 'latency_p95' not in df.columns and 'latency_ms_total' in df.columns:
            df['latency_p95'] = df['latency_ms_total'] * 1.5
        
        if 'memory_peak_mb' not in df.columns and 'latency_ms_total' in df.columns:
            df['memory_peak_mb'] = 500 + df['latency_ms_total'] * 0.3
        
        # 1. 2D Pareto frontier plots
        objective_pairs = [
            ('ndcg_at_10', 'latency_p95'),
            ('ndcg_at_10', 'memory_peak_mb'),
            ('latency_p95', 'memory_peak_mb')
        ]
        
        for obj1, obj2 in objective_pairs:
            if obj1 in df.columns and obj2 in df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot all configurations
                scatter = ax.scatter(df[obj1], df[obj2], 
                                   c=df.get('composite_score', 'blue'),
                                   alpha=0.6, s=50, cmap='viridis')
                
                # Highlight Pareto frontier
                pareto_df = df[df.get('is_pareto_optimal', False)]
                if not pareto_df.empty:
                    ax.scatter(pareto_df[obj1], pareto_df[obj2],
                             c='red', s=100, alpha=0.8, marker='*',
                             label=f'Pareto Optimal (n={len(pareto_df)})')
                
                # Budget constraint lines
                if obj1 == 'ndcg_at_10' and 'quality_floor' in self.budget_constraints:
                    ax.axvline(x=self.budget_constraints['quality_floor'], 
                             color='orange', linestyle='--', alpha=0.7,
                             label='Quality Floor')
                
                if obj2 == 'latency_p95' and 'latency_p95_budget' in self.budget_constraints:
                    ax.axhline(y=self.budget_constraints['latency_p95_budget'],
                             color='orange', linestyle='--', alpha=0.7,
                             label='Latency Budget')
                
                if obj2 == 'memory_peak_mb' and 'memory_peak_budget' in self.budget_constraints:
                    ax.axhline(y=self.budget_constraints['memory_peak_budget'],
                             color='orange', linestyle='--', alpha=0.7,
                             label='Memory Budget')
                
                # Labels and formatting
                ax.set_xlabel(self.objectives.get(obj1, {}).get('label', obj1))
                ax.set_ylabel(self.objectives.get(obj2, {}).get('label', obj2))
                ax.set_title(f'{title_prefix}: {obj1.replace("_", " ").title()} vs {obj2.replace("_", " ").title()}')
                
                if len(df) > 10:  # Only add colorbar if we have composite scores
                    plt.colorbar(scatter, label='Composite Score')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_file = self.output_dir / "figures" / f"pareto_{obj1}_vs_{obj2}.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_file)
        
        # 2. 3D Pareto frontier plot
        if all(col in df.columns for col in ['ndcg_at_10', 'latency_p95', 'memory_peak_mb']):
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all configurations
            scatter = ax.scatter(df['ndcg_at_10'], df['latency_p95'], df['memory_peak_mb'],
                               c=df.get('composite_score', 'blue'),
                               alpha=0.6, s=50, cmap='viridis')
            
            # Highlight Pareto frontier
            pareto_df = df[df.get('is_pareto_optimal', False)]
            if not pareto_df.empty:
                ax.scatter(pareto_df['ndcg_at_10'], pareto_df['latency_p95'], 
                         pareto_df['memory_peak_mb'],
                         c='red', s=100, alpha=0.8, marker='*',
                         label=f'Pareto Optimal (n={len(pareto_df)})')
            
            # Labels
            ax.set_xlabel('nDCG@10')
            ax.set_ylabel('P95 Latency (ms)')
            ax.set_zlabel('Peak Memory (MB)')
            ax.set_title(f'{title_prefix}: 3D Pareto Frontier')
            
            plt.colorbar(scatter, shrink=0.5, aspect=5)
            ax.legend()
            
            plot_file = self.output_dir / "figures" / "pareto_3d_frontier.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        # 3. Trade-off summary plot
        if analysis_results and 'objective_correlations' in analysis_results:
            correlations = analysis_results['objective_correlations']
            
            if correlations:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                pairs = list(correlations.keys())
                values = list(correlations.values())
                colors = ['red' if v < -0.5 else 'orange' if v < -0.2 else 'green' 
                         for v in values]
                
                bars = ax.bar(range(len(pairs)), values, color=colors, alpha=0.7)
                ax.set_xticks(range(len(pairs)))
                ax.set_xticklabels([p.replace('_vs_', ' vs ').replace('_', ' ').title() 
                                  for p in pairs], rotation=45, ha='right')
                ax.set_ylabel('Correlation Coefficient')
                ax.set_title(f'{title_prefix}: Objective Trade-off Correlations')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                           f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top')
                
                plot_file = self.output_dir / "figures" / "pareto_tradeoff_correlations.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_file)
        
        print(f"Generated {len(plot_files)} Pareto frontier plots")
        return plot_files
    
    def run_full_analysis(self, df: pd.DataFrame, 
                         title_prefix: str = "Lethe") -> Dict[str, Any]:
        """
        Run complete Pareto frontier analysis
        
        Args:
            df: Input dataframe with configuration data
            title_prefix: Prefix for plots and titles
            
        Returns:
            Complete analysis results
        """
        print("Starting Pareto frontier analysis...")
        
        # Apply budget constraints
        feasible_df = self.apply_budget_constraints(df)
        
        if len(feasible_df) == 0:
            print("Warning: No feasible configurations after applying budget constraints")
            feasible_df = df  # Continue with all configurations
        
        # Rank configurations
        ranked_df = self.rank_configurations(feasible_df)
        
        # Analyze trade-offs
        analysis_results = self.analyze_tradeoffs(ranked_df)
        
        # Create visualizations
        plot_files = self.create_pareto_plots(ranked_df, analysis_results, title_prefix)
        
        # Compile full results
        results = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analyzer': 'ParetoFrontierAnalyzer',
                'version': '1.0.0',
                'input_configurations': len(df),
                'feasible_configurations': len(feasible_df)
            },
            'objectives': self.objectives,
            'budget_constraints': self.budget_constraints,
            'trade_off_analysis': analysis_results,
            'pareto_configurations': ranked_df[ranked_df.get('is_pareto_optimal', False)].to_dict('records'),
            'top_10_configurations': ranked_df.head(10).to_dict('records'),
            'generated_plots': [str(p) for p in plot_files],
            'recommendations': self._generate_recommendations(ranked_df, analysis_results)
        }
        
        return results
    
    def _generate_recommendations(self, ranked_df: pd.DataFrame, 
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on Pareto analysis"""
        
        recommendations = {
            'best_overall': {},
            'best_quality': {},
            'best_latency': {},
            'best_memory': {},
            'pareto_summary': {}
        }
        
        if len(ranked_df) == 0:
            return recommendations
        
        # Best overall (top ranked)
        best_overall = ranked_df.iloc[0]
        recommendations['best_overall'] = {
            'configuration': best_overall.get('method', 'unknown'),
            'rank': int(best_overall.get('rank', 0)),
            'composite_score': float(best_overall.get('composite_score', 0)),
            'is_pareto_optimal': bool(best_overall.get('is_pareto_optimal', False))
        }
        
        # Best in each objective
        if 'ndcg_at_10' in ranked_df.columns:
            best_quality = ranked_df.loc[ranked_df['ndcg_at_10'].idxmax()]
            recommendations['best_quality'] = {
                'configuration': best_quality.get('method', 'unknown'),
                'ndcg_at_10': float(best_quality['ndcg_at_10']),
                'is_pareto_optimal': bool(best_quality.get('is_pareto_optimal', False))
            }
        
        if 'latency_p95' in ranked_df.columns:
            best_latency = ranked_df.loc[ranked_df['latency_p95'].idxmin()]
            recommendations['best_latency'] = {
                'configuration': best_latency.get('method', 'unknown'),
                'latency_p95': float(best_latency['latency_p95']),
                'is_pareto_optimal': bool(best_latency.get('is_pareto_optimal', False))
            }
        elif 'latency_ms_total' in ranked_df.columns:
            best_latency = ranked_df.loc[ranked_df['latency_ms_total'].idxmin()]
            recommendations['best_latency'] = {
                'configuration': best_latency.get('method', 'unknown'),
                'latency_ms_total': float(best_latency['latency_ms_total']),
                'is_pareto_optimal': bool(best_latency.get('is_pareto_optimal', False))
            }
        
        if 'memory_peak_mb' in ranked_df.columns:
            best_memory = ranked_df.loc[ranked_df['memory_peak_mb'].idxmin()]
            recommendations['best_memory'] = {
                'configuration': best_memory.get('method', 'unknown'),
                'memory_peak_mb': float(best_memory['memory_peak_mb']),
                'is_pareto_optimal': bool(best_memory.get('is_pareto_optimal', False))
            }
        
        # Pareto summary
        pareto_configs = ranked_df[ranked_df.get('is_pareto_optimal', False)]
        recommendations['pareto_summary'] = {
            'count': len(pareto_configs),
            'efficiency': len(pareto_configs) / len(ranked_df),
            'configurations': pareto_configs.get('method', pd.Series([])).tolist()
        }
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_file: Path) -> None:
        """Save analysis results to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Pareto analysis results saved to: {output_file}")


def load_experimental_data(artifacts_dir: Path) -> pd.DataFrame:
    """Load experimental data from artifacts directory"""
    
    data_sources = [
        artifacts_dir / "synthetic_dataset.json",
        artifacts_dir / "statistical_analysis_results.json", 
        artifacts_dir / "publication_statistical_results.json"
    ]
    
    for source in data_sources:
        if source.exists():
            print(f"Loading data from {source}")
            with open(source, 'r') as f:
                data = json.load(f)
            
            if 'datapoints' in data:
                return pd.DataFrame(data['datapoints'])
            elif 'summary_statistics' in data:
                return reconstruct_from_summary_stats(data)
    
    raise FileNotFoundError("No experimental data found")


def reconstruct_from_summary_stats(data: Dict[str, Any]) -> pd.DataFrame:
    """Reconstruct approximate dataframe from summary statistics"""
    
    summary_stats = data.get('summary_statistics', {})
    all_data = []
    
    for method, method_stats in summary_stats.items():
        # Create base row
        base_row = {
            'method': method,
            'iteration': 0 if method.startswith('baseline') else int(method[-1]) if method[-1].isdigit() else 0,
            'domain': 'mixed'
        }
        
        # Add metrics with mean values (deterministic)
        for metric, metric_stats in method_stats.items():
            if 'mean' in metric_stats:
                base_row[metric] = metric_stats['mean']
        
        all_data.append(base_row)
    
    return pd.DataFrame(all_data)


def main():
    """Main execution function"""
    artifacts_dir = Path("artifacts")
    output_dir = Path("analysis")
    
    # Initialize analyzer
    analyzer = ParetoFrontierAnalyzer(output_dir)
    
    # Load data
    print("Loading experimental data...")
    df = load_experimental_data(artifacts_dir)
    
    print(f"Loaded {len(df)} configurations")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Available metrics: {[col for col in df.columns if col not in ['method', 'iteration', 'domain']]}")
    
    # Run full Pareto analysis
    results = analyzer.run_full_analysis(df, title_prefix="Lethe Hybrid System")
    
    # Save results
    output_file = output_dir / "pareto_analysis_results.json"
    analyzer.save_results(results, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("PARETO FRONTIER ANALYSIS SUMMARY")
    print("="*60)
    
    analysis = results.get('trade_off_analysis', {})
    print(f"Total configurations analyzed: {analysis.get('total_configurations', 0)}")
    print(f"Pareto optimal configurations: {analysis.get('pareto_frontier_size', 0)}")
    print(f"Pareto efficiency: {analysis.get('pareto_efficiency', 0):.1%}")
    print(f"Hypervolume indicator: {analysis.get('hypervolume', 0):.4f}")
    
    recommendations = results.get('recommendations', {})
    if 'best_overall' in recommendations:
        best = recommendations['best_overall']
        print(f"Best overall configuration: {best.get('configuration', 'unknown')}")
        print(f"  Composite score: {best.get('composite_score', 0):.4f}")
        print(f"  Pareto optimal: {best.get('is_pareto_optimal', False)}")
    
    plots = results.get('generated_plots', [])
    print(f"Generated {len(plots)} visualization plots")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()