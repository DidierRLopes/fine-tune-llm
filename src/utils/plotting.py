"""Plotting utilities for visualization and analysis."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class PlottingUtils:
    """Utilities for creating various plots and visualizations."""
    
    def __init__(self, output_dir: str = "logs/plots"):
        """Initialize plotting utilities.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_training_curves(self, train_losses: List[Tuple[int, float]], 
                           val_losses: List[Tuple[int, float]] = None,
                           save_name: str = "training_curves.png") -> str:
        """Plot training and validation loss curves.
        
        Args:
            train_losses: List of (iteration, loss) tuples
            val_losses: List of (iteration, loss) tuples (optional)
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        if train_losses:
            iterations, losses = zip(*train_losses)
            ax.plot(iterations, losses, 'o-', label='Training Loss', 
                   color=self.colors[0], markersize=4, linewidth=2)
        
        # Plot validation loss
        if val_losses:
            iterations, losses = zip(*val_losses)
            ax.plot(iterations, losses, 's-', label='Validation Loss', 
                   color=self.colors[1], markersize=4, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add trend line for training loss if enough points
        if train_losses and len(train_losses) > 5:
            iterations, losses = zip(*train_losses)
            z = np.polyfit(iterations, losses, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), '--', alpha=0.7, 
                   color=self.colors[0], label='Training Trend')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def plot_perplexity(self, perplexities: List[Tuple[int, float]], 
                       save_name: str = "perplexity.png") -> str:
        """Plot perplexity over training.
        
        Args:
            perplexities: List of (iteration, perplexity) tuples
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        if not perplexities:
            print("No perplexity data to plot")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations, perp_values = zip(*perplexities)
        ax.plot(iterations, perp_values, 'o-', color=self.colors[2], 
               markersize=4, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Perplexity')
        ax.set_title('Validation Perplexity')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at the minimum
        min_perp = min(perp_values)
        ax.axhline(y=min_perp, color='red', linestyle='--', alpha=0.7, 
                  label=f'Best: {min_perp:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def plot_score_distributions(self, scores_dict: Dict[str, List[float]], 
                               title: str = "Score Distributions",
                               save_name: str = "score_distributions.png") -> str:
        """Plot score distributions for multiple models or metrics.
        
        Args:
            scores_dict: Dictionary mapping names to score lists
            title: Plot title
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, len(scores_dict), figsize=(5*len(scores_dict), 5))
        
        if len(scores_dict) == 1:
            axes = [axes]
        
        for i, (name, scores) in enumerate(scores_dict.items()):
            ax = axes[i]
            
            # Histogram
            ax.hist(scores, bins=30, alpha=0.7, edgecolor='black', color=self.colors[i % len(self.colors)])
            
            # Statistics lines
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            
            ax.axvline(mean_score, color='red', linestyle='--', 
                      label=f'Mean: {mean_score:.3f}')
            ax.axvline(median_score, color='green', linestyle='--', 
                      label=f'Median: {median_score:.3f}')
            
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\nDistribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def plot_performance_comparison(self, comparison_data: Dict[str, Dict[str, float]], 
                                  save_name: str = "performance_comparison.png") -> str:
        """Plot performance comparison between models.
        
        Args:
            comparison_data: Nested dict with model names and performance metrics
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        models = list(comparison_data.keys())
        
        # Extract performance categories
        categories = ['excellent_rate', 'good_rate', 'acceptable_rate', 'poor_rate']
        category_labels = ['Excellent (≥0.9)', 'Good (≥0.7)', 'Acceptable (≥0.5)', 'Poor (≤0.3)']
        category_colors = ['green', 'blue', 'orange', 'red']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(models))
        width = 0.2
        
        for i, (category, label, color) in enumerate(zip(categories, category_labels, category_colors)):
            values = [comparison_data[model].get(category, 0) for model in models]
            offset = (i - 1.5) * width
            bars = ax.bar(x_pos + offset, values, width, label=label, 
                         alpha=0.8, color=color)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0.02:  # Only show label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.1%}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Percentage of Samples')
        ax.set_title('Performance Breakdown Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def plot_score_vs_length(self, scores: List[float], text_lengths: List[int],
                           title: str = "Score vs Text Length",
                           save_name: str = "score_vs_length.png") -> str:
        """Plot scores against text lengths to analyze correlations.
        
        Args:
            scores: List of evaluation scores
            text_lengths: List of corresponding text lengths
            title: Plot title
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(text_lengths, scores, alpha=0.6, color=self.colors[0])
        
        # Add trend line
        if len(scores) > 5:
            z = np.polyfit(text_lengths, scores, 1)
            p = np.poly1d(z)
            ax.plot(sorted(text_lengths), p(sorted(text_lengths)), 
                   'r--', alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
        
        # Calculate correlation
        correlation = np.corrcoef(text_lengths, scores)[0, 1]
        
        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Score')
        ax.set_title(f'{title}\nCorrelation: {correlation:.3f}')
        ax.grid(True, alpha=0.3)
        
        if len(scores) > 5:
            ax.legend()
        
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def plot_confusion_matrix_style(self, data: np.ndarray, labels: List[str],
                                  title: str = "Comparison Matrix",
                                  save_name: str = "comparison_matrix.png") -> str:
        """Plot a confusion matrix style heatmap.
        
        Args:
            data: 2D numpy array of data
            labels: Labels for rows and columns
            title: Plot title
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(data, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                              ha="center", va="center", color="black" if data[i, j] < 0.5 else "white")
        
        ax.set_title(title)
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def create_subplot_grid(self, plots_config: List[Dict[str, Any]], 
                          grid_shape: Tuple[int, int] = None,
                          figsize: Tuple[int, int] = (15, 10),
                          save_name: str = "combined_plots.png") -> str:
        """Create a grid of subplots.
        
        Args:
            plots_config: List of plot configuration dictionaries
            grid_shape: Shape of subplot grid (auto-calculated if None)
            figsize: Figure size
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot
        """
        n_plots = len(plots_config)
        
        if grid_shape is None:
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
        else:
            rows, cols = grid_shape
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes for easier indexing
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, config in enumerate(plots_config):
            ax = axes[i]
            plot_type = config.get('type', 'line')
            
            if plot_type == 'line':
                ax.plot(config['x'], config['y'], 'o-', color=self.colors[i % len(self.colors)])
            elif plot_type == 'bar':
                ax.bar(config['x'], config['y'], color=self.colors[i % len(self.colors)])
            elif plot_type == 'hist':
                ax.hist(config['data'], bins=config.get('bins', 30), 
                       color=self.colors[i % len(self.colors)], alpha=0.7)
            
            ax.set_title(config.get('title', f'Plot {i+1}'))
            ax.set_xlabel(config.get('xlabel', 'X'))
            ax.set_ylabel(config.get('ylabel', 'Y'))
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return str(plot_path)
    
    def save_all_plots_summary(self) -> str:
        """Create a summary of all saved plots.
        
        Returns:
            Path to summary file
        """
        summary_file = self.output_dir / "plots_summary.txt"
        
        plot_files = list(self.output_dir.glob("*.png"))
        plot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        with open(summary_file, 'w') as f:
            f.write("PLOTS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total plots: {len(plot_files)}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            for plot_file in plot_files:
                stat = plot_file.stat()
                size_kb = stat.st_size / 1024
                f.write(f"- {plot_file.name} ({size_kb:.1f} KB)\n")
        
        print(f"Plots summary saved to: {summary_file}")
        return str(summary_file)