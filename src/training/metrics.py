"""Training metrics collection for fine-tuning pipeline."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from pathlib import Path


class TrainingMetrics:
    """Enhanced metrics class with early stopping and comprehensive tracking."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, logs_dir: str = "logs/training"):
        """Initialize metrics collector.
        
        Args:
            patience: Number of iterations to wait for improvement
            min_delta: Minimum change to qualify as improvement
            logs_dir: Directory to save logs
        """
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.best_val_loss = float('inf')
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.should_stop = False
        
        # Setup logging directory
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nMetrics logging to: {self.logs_dir}")
    
    def on_train_loss_report(self, info: Dict[str, Any]):
        """Callback for training loss reporting.
        
        Args:
            info: Training information dictionary with 'iteration' and 'train_loss'
        """
        iteration = info["iteration"]
        train_loss = info["train_loss"]
        self.train_losses.append((iteration, train_loss))
        
        print(f"  Iteration {iteration}: Train loss = {train_loss:.4f}")
    
    def on_val_loss_report(self, info: Dict[str, Any]):
        """Callback for validation loss reporting.
        
        Args:
            info: Validation information dictionary with 'iteration' and 'val_loss'
        """
        iteration = info["iteration"]
        val_loss = info["val_loss"]
        self.val_losses.append((iteration, val_loss))
        
        # Calculate perplexity
        perplexity = np.exp(val_loss)
        self.perplexities.append((iteration, perplexity))
        
        print(f"  Iteration {iteration}: Val loss = {val_loss:.4f}, Perplexity = {perplexity:.2f}")
        
        # Early stopping logic
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"  → New best validation loss: {val_loss:.4f} (perplexity: {perplexity:.2f})")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                print(f"  → Early stopping triggered (patience: {self.patience})")
    
    def save_metrics(self):
        """Save metrics to JSON files."""
        import json
        
        metrics_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "perplexities": self.perplexities,
            "best_val_loss": self.best_val_loss,
            "final_patience_counter": self.patience_counter,
            "early_stopped": self.should_stop
        }
        
        metrics_file = self.logs_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Training metrics saved to: {metrics_file}")
        return metrics_file
    
    def plot_metrics(self, save_plot: bool = True):
        """Plot training metrics including perplexity.
        
        Args:
            save_plot: Whether to save plot to file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot losses
        if self.train_losses:
            train_its, train_losses = zip(*self.train_losses)
            ax1.plot(train_its, train_losses, '-o', label='Train Loss', markersize=4)
        
        if self.val_losses:
            val_its, val_losses = zip(*self.val_losses)
            ax1.plot(val_its, val_losses, '-o', label='Validation Loss', markersize=4)
        
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot perplexity
        if self.perplexities:
            perp_its, perp_values = zip(*self.perplexities)
            ax2.plot(perp_its, perp_values, '-o', color='green', markersize=4)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Validation Perplexity")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.logs_dir / "training_metrics.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to: {plot_file}")
        
        plt.show()
        return fig
    
    def get_final_summary(self) -> Dict[str, Any]:
        """Get final training summary.
        
        Returns:
            Dictionary with training summary
        """
        summary = {
            "total_iterations": len(self.train_losses),
            "early_stopped": self.should_stop,
            "best_val_loss": self.best_val_loss,
            "final_patience_counter": self.patience_counter
        }
        
        if self.val_losses:
            final_val_loss = self.val_losses[-1][1]
            final_perplexity = np.exp(final_val_loss)
            summary.update({
                "final_val_loss": final_val_loss,
                "final_perplexity": final_perplexity,
                "best_perplexity": np.exp(self.best_val_loss)
            })
        
        if self.train_losses:
            final_train_loss = self.train_losses[-1][1]
            summary["final_train_loss"] = final_train_loss
        
        return summary
    
    def print_final_summary(self):
        """Print final training summary."""
        summary = self.get_final_summary()
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print(f"\n📊 Training Completion:")
        print(f"  Total iterations:     {summary['total_iterations']}")
        print(f"  Early stopped:        {summary['early_stopped']}")
        
        if 'final_train_loss' in summary:
            print(f"  Final train loss:     {summary['final_train_loss']:.4f}")
        
        if 'final_val_loss' in summary:
            print(f"  Final val loss:       {summary['final_val_loss']:.4f}")
            print(f"  Best val loss:        {summary['best_val_loss']:.4f}")
            print(f"  Final perplexity:     {summary['final_perplexity']:.2f}")
            print(f"  Best perplexity:      {summary['best_perplexity']:.2f}")
        
        if summary['early_stopped']:
            print(f"\n⚠️  Training stopped early due to lack of improvement")
            print(f"   Patience counter reached: {summary['final_patience_counter']}/{self.patience}")
        else:
            print(f"\n✅ Training completed all iterations")
        print("")
        
        return summary