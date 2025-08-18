"""Model comparison utilities for evaluation pipeline."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class ModelComparator:
    """Compare and visualize performance between models."""
    
    def __init__(self, results_dir: str = "logs/evaluation"):
        """Initialize comparator.
        
        Args:
            results_dir: Directory to save comparison results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_comparison_metrics(self, evaluation_results: List[Dict[str, Any]], 
                                 model_names: List[str]) -> Dict[str, Any]:
        """Extract metrics for comparison from evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            model_names: List of model names
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison_metrics = {
            'model_names': model_names,
            'word_overlap': [],
            'bertscore_f1': [],
            'length_stats': []
        }
        
        for results in evaluation_results:
            metrics = results['metrics']
            
            # Word overlap metrics
            comparison_metrics['word_overlap'].append({
                'mean': metrics['word_overlap']['mean'],
                'std': metrics['word_overlap']['std'],
                'median': metrics['word_overlap']['median']
            })
            
            # BERTScore metrics (if available)
            if 'bertscore' in metrics and metrics['bertscore']:
                comparison_metrics['bertscore_f1'].append({
                    'mean': metrics['bertscore']['f1']['mean'],
                    'std': metrics['bertscore']['f1']['std'],
                    'median': metrics['bertscore']['f1']['median']
                })
            else:
                comparison_metrics['bertscore_f1'].append(None)
            
            # Performance breakdown - removed, we focus on core metrics
            
            # Length statistics
            comparison_metrics['length_stats'].append({
                'pred_mean_length': metrics['length_stats']['predictions']['mean_length'],
                'ref_mean_length': metrics['length_stats']['references']['mean_length']
            })
        
        return comparison_metrics
    
    def plot_score_comparison(self, comparison_metrics: Dict[str, Any], 
                            metric_type: str = 'word_overlap', save_plot: bool = True):
        """Plot score comparison between models.
        
        Args:
            comparison_metrics: Comparison metrics dictionary
            metric_type: Type of metric to plot ('word_overlap' or 'bertscore_f1')
            save_plot: Whether to save the plot
        """
        model_names = comparison_metrics['model_names']
        metrics_data = comparison_metrics[metric_type]
        
        # Filter out None values for BERTScore
        valid_data = [(name, data) for name, data in zip(model_names, metrics_data) if data is not None]
        
        if not valid_data:
            print(f"No valid {metric_type} data for plotting")
            return
        
        valid_names, valid_metrics = zip(*valid_data)
        
        means = [m['mean'] for m in valid_metrics]
        stds = [m['std'] for m in valid_metrics]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(valid_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=['blue', 'orange', 'green', 'red', 'purple'][:len(valid_names)])
        
        # Customize plot
        ax.set_xlabel('Models')
        ax.set_ylabel(f'{metric_type.replace("_", " ").title()} Score')
        ax.set_title(f'Model Performance Comparison: {metric_type.replace("_", " ").title()}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.results_dir / f"{metric_type}_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Score comparison plot saved to: {plot_file}")
        
        plt.show()
        return fig
    
    def plot_performance_breakdown(self, comparison_metrics: Dict[str, Any], save_plot: bool = True):
        """Plot performance breakdown comparison.
        
        Args:
            comparison_metrics: Comparison metrics dictionary
            save_plot: Whether to save the plot
        """
        model_names = comparison_metrics['model_names']
        breakdown_data = comparison_metrics['performance_breakdown']
        
        # Extract rates
        excellent_rates = [bd['excellent_rate'] for bd in breakdown_data]
        good_rates = [bd['good_rate'] for bd in breakdown_data]
        acceptable_rates = [bd['acceptable_rate'] for bd in breakdown_data]
        poor_rates = [bd['poor_rate'] for bd in breakdown_data]
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(model_names))
        width = 0.2
        
        bars1 = ax.bar(x_pos - 1.5*width, excellent_rates, width, label='Excellent (≥0.9)', alpha=0.8, color='green')
        bars2 = ax.bar(x_pos - 0.5*width, good_rates, width, label='Good (≥0.7)', alpha=0.8, color='blue')
        bars3 = ax.bar(x_pos + 0.5*width, acceptable_rates, width, label='Acceptable (≥0.5)', alpha=0.8, color='orange')
        bars4 = ax.bar(x_pos + 1.5*width, poor_rates, width, label='Poor (≤0.3)', alpha=0.8, color='red')
        
        # Customize plot
        ax.set_xlabel('Models')
        ax.set_ylabel('Percentage of Samples')
        ax.set_title('Performance Breakdown Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.results_dir / "performance_breakdown_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Performance breakdown plot saved to: {plot_file}")
        
        plt.show()
        return fig
    
    def plot_score_distribution(self, evaluation_results: List[Dict[str, Any]], 
                              model_names: List[str], metric_type: str = 'word_overlap',
                              save_plot: bool = True):
        """Plot score distribution comparison.
        
        Args:
            evaluation_results: List of evaluation results
            model_names: List of model names
            metric_type: Type of metric to plot
            save_plot: Whether to save the plot
        """
        fig, axes = plt.subplots(1, len(evaluation_results), figsize=(5*len(evaluation_results), 5))
        
        if len(evaluation_results) == 1:
            axes = [axes]
        
        for i, (results, name) in enumerate(zip(evaluation_results, model_names)):
            metrics = results['metrics']
            
            # Get scores based on metric type
            if metric_type == 'bertscore_f1' and 'bertscore' in metrics and metrics['bertscore']:
                scores = metrics['bertscore']['f1']['scores']
                title_suffix = "BERTScore F1"
            else:
                scores = metrics['word_overlap']['scores']
                title_suffix = "Word Overlap"
            
            # Plot histogram
            axes[i].hist(scores, bins=30, alpha=0.7, edgecolor='black')
            axes[i].axvline(np.mean(scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(scores):.3f}')
            axes[i].axvline(np.median(scores), color='green', linestyle='--', 
                           label=f'Median: {np.median(scores):.3f}')
            
            axes[i].set_xlabel(f'{title_suffix} Score')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{name}\n{title_suffix} Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.results_dir / f"{metric_type}_distributions.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Score distribution plot saved to: {plot_file}")
        
        plt.show()
        return fig
    
    def create_comparison_report(self, evaluation_results: List[Dict[str, Any]], 
                               model_names: List[str]) -> str:
        """Create a comprehensive comparison report.
        
        Args:
            evaluation_results: List of evaluation results
            model_names: List of model names
            
        Returns:
            Path to the generated report file
        """
        comparison_metrics = self.extract_comparison_metrics(evaluation_results, model_names)
        
        # Generate all plots (silently - no duplicate comparison output)
        self.plot_score_comparison(comparison_metrics, 'word_overlap')
        
        # Plot BERTScore if available
        has_bertscore = any(bd is not None for bd in comparison_metrics['bertscore_f1'])
        if has_bertscore:
            self.plot_score_comparison(comparison_metrics, 'bertscore_f1')
        
        # Plot score distribution
        self.plot_score_distribution(evaluation_results, model_names, 'word_overlap')
        
        if has_bertscore:
            self.plot_score_distribution(evaluation_results, model_names, 'bertscore_f1')
        
        # Save comparison metrics
        import json
        report_file = self.results_dir / "model_comparison_report.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_metrics = convert_numpy(comparison_metrics)
        
        with open(report_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\nComparison report saved to: {report_file}")
        return str(report_file)