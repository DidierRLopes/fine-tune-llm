"""Metrics calculation utilities for model evaluation."""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


class MetricsCalculator:
    """Calculate various evaluation metrics for model performance."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.bertscore_available = False
        self._try_import_bertscore()
    
    def _try_import_bertscore(self):
        """Try to import BERTScore for semantic evaluation."""
        try:
            from evaluate import load as load_metric
            self.load_metric = load_metric
            self.bertscore_available = True
        except ImportError:
            print("Note: 'evaluate' library not available. BERTScore evaluation disabled.")
            self.load_metric = None
    
    def calculate_word_overlap(self, predictions: List[str], references: List[str]) -> List[float]:
        """Calculate word overlap scores between predictions and references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            List of word overlap scores (0.0 to 1.0)
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Extract words and convert to lowercase
            pred_words = set(re.findall(r'\w+', pred.lower()))
            ref_words = set(re.findall(r'\w+', ref.lower()))
            
            if not ref_words and not pred_words:
                scores.append(1.0)  # Both empty
            elif not ref_words or not pred_words:
                scores.append(0.0)  # One empty
            else:
                # Jaccard similarity
                overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
                scores.append(overlap)
        
        return scores
    
    def calculate_bertscore(self, predictions: List[str], references: List[str], 
                          model_type: str = "microsoft/deberta-xlarge-mnli") -> Dict[str, List[float]]:
        """Calculate BERTScore for semantic similarity.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            model_type: Model to use for BERTScore calculation
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not self.bertscore_available:
            raise ValueError("BERTScore not available. Install 'evaluate' and 'bert-score' packages.")
        
        print("Computing BERTScore (this may take a while)...")
        bertscore = self.load_metric("bertscore")
        
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type=model_type,
            verbose=False
        )
        
        return {
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1']
        }
    
    def calculate_length_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate length statistics for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with length statistics
        """
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        
        return {
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'total_chars': np.sum(lengths)
        }
    
    def calculate_comprehensive_metrics(self, predictions: List[str], references: List[str],
                                      use_bertscore: bool = False, 
                                      bertscore_model: str = "microsoft/deberta-xlarge-mnli") -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            use_bertscore: Whether to calculate BERTScore
            bertscore_model: Model to use for BERTScore
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic validation
        if len(predictions) != len(references):
            raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
        
        print(f"Calculating metrics for {len(predictions)} samples...")
        
        # Word overlap scores
        print("Calculating word overlap scores...")
        word_overlap_scores = self.calculate_word_overlap(predictions, references)
        metrics['word_overlap'] = {
            'scores': word_overlap_scores,
            'mean': np.mean(word_overlap_scores),
            'std': np.std(word_overlap_scores),
            'median': np.median(word_overlap_scores),
            'min': np.min(word_overlap_scores),
            'max': np.max(word_overlap_scores)
        }
        
        # BERTScore if requested and available
        if use_bertscore and self.bertscore_available:
            try:
                bertscore_results = self.calculate_bertscore(predictions, references, bertscore_model)
                metrics['bertscore'] = {
                    'precision': {
                        'scores': bertscore_results['precision'],
                        'mean': np.mean(bertscore_results['precision']),
                        'std': np.std(bertscore_results['precision']),
                        'median': np.median(bertscore_results['precision'])
                    },
                    'recall': {
                        'scores': bertscore_results['recall'],
                        'mean': np.mean(bertscore_results['recall']),
                        'std': np.std(bertscore_results['recall']),
                        'median': np.median(bertscore_results['recall'])
                    },
                    'f1': {
                        'scores': bertscore_results['f1'],
                        'mean': np.mean(bertscore_results['f1']),
                        'std': np.std(bertscore_results['f1']),
                        'median': np.median(bertscore_results['f1'])
                    }
                }
            except Exception as e:
                print(f"Warning: BERTScore calculation failed: {e}")
                metrics['bertscore'] = None
        
        # Length statistics
        pred_length_stats = self.calculate_length_statistics(predictions)
        ref_length_stats = self.calculate_length_statistics(references)
        
        metrics['length_stats'] = {
            'predictions': pred_length_stats,
            'references': ref_length_stats
        }
        
        # Remove performance breakdown - just use core metrics
        
        return metrics
    
    
    def save_metrics(self, metrics: Dict[str, Any], save_path: str):
        """Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save JSON file
        """
        import json
        from pathlib import Path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        metrics_serializable = convert_numpy(metrics)
        
        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")
    
    def print_metrics_summary(self, metrics: Dict[str, Any], title: str = "Evaluation Metrics"):
        """Print a formatted summary of metrics.
        
        Args:
            metrics: Metrics dictionary
            title: Title for the summary
        """
        print("\n" + "="*60)
        print(f"{title.upper()}")
        print("="*60)
        
        # Word overlap metrics
        if 'word_overlap' in metrics:
            wo = metrics['word_overlap']
            print(f"\n📊 Word Overlap Metrics:")
            print(f"  Mean:    {wo['mean']:.4f}")
            print(f"  Median:  {wo['median']:.4f}")
            print(f"  Std:     {wo['std']:.4f}")
            print(f"  Range:   [{wo['min']:.4f}, {wo['max']:.4f}]")
        
        # BERTScore metrics
        if 'bertscore' in metrics and metrics['bertscore']:
            bs = metrics['bertscore']
            print(f"\n🎯 BERTScore Metrics:")
            print(f"  Precision: {bs['precision']['mean']:.4f} (±{bs['precision']['std']:.4f})")
            print(f"  Recall:    {bs['recall']['mean']:.4f} (±{bs['recall']['std']:.4f})")
            print(f"  F1:        {bs['f1']['mean']:.4f} (±{bs['f1']['std']:.4f})")
        
        # Length statistics summary
        if 'length_stats' in metrics:
            pred_stats = metrics['length_stats']['predictions']
            ref_stats = metrics['length_stats']['references']
            print(f"\n📏 Length Statistics:")
            print(f"  Predictions: {pred_stats['mean_length']:.1f} chars (±{pred_stats['std_length']:.1f})")
            print(f"  References:  {ref_stats['mean_length']:.1f} chars (±{ref_stats['std_length']:.1f})")
        
        print("="*60)