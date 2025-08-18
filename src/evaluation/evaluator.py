"""Model evaluation utilities for fine-tuning pipeline."""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten
import mlx.core as mx
from .metrics_calculator import MetricsCalculator


class ModelEvaluator:
    """Evaluate model performance on test dataset."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration.
        
        Args:
            config_path: Path to evaluation configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.metrics_config = self.config['metrics']
        self.comparison_config = self.config['comparison']
        self.paths_config = self.config['paths']
        
        self.metrics_calculator = MetricsCalculator()
    
    def load_test_data(self, test_data_path: str = None) -> Tuple[List[str], List[str]]:
        """Load and parse test data.
        
        Args:
            test_data_path: Path to test data JSON file (optional, uses config default)
            
        Returns:
            Tuple of (questions, answers)
        """
        if test_data_path is None:
            test_data_path = self.paths_config['test_data']
        
        print(f"Loading test data from: {test_data_path}")
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        questions = []
        answers = []
        
        for sample in test_data:
            # Split by assistant tag to separate prompt from answer
            if "<|assistant|>" in sample:
                parts = sample.split("<|assistant|>")
                prompt_part = parts[0] + "<|assistant|>"  # Include the assistant tag
                answer_part = parts[1] if len(parts) > 1 else ""
                
                # Remove the closing <|end|> from answer
                if "<|end|>" in answer_part:
                    answer_part = answer_part.replace("<|end|>", "").strip()
                
                questions.append(prompt_part)
                answers.append(answer_part)
        
        print(f"Extracted {len(questions)} test questions")
        return questions, answers
    
    def generate_predictions(self, model, tokenizer, questions: List[str]) -> List[str]:
        """Generate predictions for test questions.
        
        Args:
            model: Model to evaluate
            tokenizer: Model tokenizer
            questions: List of formatted questions
            
        Returns:
            List of generated predictions
        """
        predictions = []
        max_tokens = self.eval_config['max_tokens']
        temperature = self.eval_config['temperature']
        
        print(f"Generating predictions for {len(questions)} questions...")
        
        for question in tqdm(questions, desc="Generating predictions"):
            try:
                # Generate response
                response = generate(
                    model, tokenizer, question, 
                    max_tokens=max_tokens, 
                    temp=temperature
                )
                
                # Clean response - extract only the assistant part
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1]
                if "<|end|>" in response:
                    response = response.split("<|end|>")[0]
                
                predictions.append(response.strip())
                
            except Exception as e:
                print(f"Error generating prediction: {e}")
                predictions.append("")  # Empty prediction for failed generation
        
        return predictions
    
    def evaluate_model(self, model, tokenizer, questions: List[str], references: List[str]) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            model: Model to evaluate
            tokenizer: Model tokenizer
            questions: List of test questions
            references: List of reference answers
            
        Returns:
            Dictionary with evaluation results
        """
        # Generate predictions
        predictions = self.generate_predictions(model, tokenizer, questions)
        
        # Calculate metrics
        use_bertscore = (self.eval_config['method'] == 'bertscore')
        bertscore_model = self.metrics_config.get('bertscore', {}).get('model_type', 'microsoft/deberta-xlarge-mnli')
        
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            predictions, references, 
            use_bertscore=use_bertscore,
            bertscore_model=bertscore_model
        )
        
        # Add evaluation metadata
        eval_results = {
            'evaluation_config': self.eval_config,
            'metrics': metrics,
            'predictions': predictions,
            'references': references,
            'questions': questions
        }
        
        return eval_results
    
    def save_evaluation_results(self, results: Dict[str, Any], model_name: str) -> str:
        """Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the evaluated model
            
        Returns:
            Path to saved results file
        """
        results_dir = Path(self.paths_config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{model_name}_evaluation.json"
        
        self.metrics_calculator.save_metrics(results, str(results_file))
        
        return str(results_file)
    
    def load_model(self, model_path: str):
        """Load model and tokenizer.
        
        Args:
            model_path: Path to model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {model_path}")
        model, tokenizer = load(model_path)
        print("Model loaded successfully")
        return model, tokenizer
    
    def load_model_with_adapters(self, base_model_path: str, adapter_path: str):
        """Load base model with LoRA adapters applied at runtime.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory
            
        Returns:
            Tuple of (model_with_adapters, tokenizer)
        """
        print(f"Loading base model: {base_model_path}")
        model, tokenizer = load(base_model_path)
        
        # Load adapter configuration
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")
        
        print(f"Loading adapter config: {adapter_config_path}")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Freeze base model
        print("Freezing base model parameters...")
        model.freeze()
        
        # Apply LoRA layers
        print("Applying LoRA adapters...")
        linear_to_lora_layers(
            model,
            adapter_config["lora_layers"],
            adapter_config["lora_parameters"]
        )
        
        # Load adapter weights
        adapter_file = Path(adapter_path) / "adapters.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"Adapter weights not found: {adapter_file}")
        
        print(f"Loading adapter weights: {adapter_file}")
        model.load_weights(str(adapter_file), strict=False)
        
        # Count parameters
        num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
        
        print(f"✅ Model with LoRA adapters loaded successfully")
        print(f"  Trainable parameters: {num_train_params:,} ({num_train_params/total_params*100:.2f}%)")
        print(f"  Total parameters:     {total_params:,}")
        
        return model, tokenizer
    
    def evaluate_model_from_path(self, model_path: str, model_name: str = None, 
                                test_data_path: str = None) -> Dict[str, Any]:
        """Evaluate a model from its path.
        
        Args:
            model_path: Path to model directory
            model_name: Name for the model (defaults to path basename)
            test_data_path: Path to test data (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if model_name is None:
            model_name = Path(model_path).name
        
        print("="*60)
        print(f"EVALUATING MODEL: {model_name}")
        print("="*60)
        
        # Load model
        model, tokenizer = self.load_model(model_path)
        
        # Load test data
        questions, references = self.load_test_data(test_data_path)
        
        # Evaluate model
        results = self.evaluate_model(model, tokenizer, questions, references)
        
        # Print metrics summary
        self.metrics_calculator.print_metrics_summary(
            results['metrics'], 
            f"{model_name} Evaluation Results"
        )
        
        # Save results
        results_file = self.save_evaluation_results(results, model_name)
        
        print(f"\nEvaluation completed! Results saved to: {results_file}")
        
        return results
    
    def evaluate_model_with_adapters(self, base_model_path: str, adapter_path: str, 
                                   model_name: str = "lora_runtime", test_data_path: str = None) -> Dict[str, Any]:
        """Evaluate base model with LoRA adapters applied at runtime.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory
            model_name: Name for the model (defaults to "lora_runtime")
            test_data_path: Path to test data (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        print("="*60)
        print(f"EVALUATING MODEL: {model_name} (Base + LoRA Adapters)")
        print("="*60)
        
        # Load base model with adapters
        model, tokenizer = self.load_model_with_adapters(base_model_path, adapter_path)
        
        # Load test data
        questions, references = self.load_test_data(test_data_path)
        
        # Evaluate model
        results = self.evaluate_model(model, tokenizer, questions, references)
        
        # Print metrics summary
        self.metrics_calculator.print_metrics_summary(
            results['metrics'], 
            f"{model_name} Evaluation Results"
        )
        
        # Save results
        results_file = self.save_evaluation_results(results, model_name)
        
        print(f"\nEvaluation completed! Results saved to: {results_file}")
        
        return results
    
    def compare_models(self, model_results: List[Dict[str, Any]], model_names: List[str]):
        """Compare multiple model evaluation results.
        
        Args:
            model_results: List of evaluation results dictionaries
            model_names: List of model names
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Extract primary metrics for comparison
        comparison_data = []
        
        for i, (results, name) in enumerate(zip(model_results, model_names)):
            metrics = results['metrics']
            
            # Use BERTScore F1 if available, otherwise word overlap
            if 'bertscore' in metrics and metrics['bertscore']:
                primary_score = metrics['bertscore']['f1']['mean']
                primary_metric = "BERTScore F1"
            else:
                primary_score = metrics['word_overlap']['mean']
                primary_metric = "Word Overlap"
            
            comparison_data.append({
                'name': name,
                'primary_score': primary_score,
                'primary_metric': primary_metric
            })
        
        # Sort by primary score (descending)
        comparison_data.sort(key=lambda x: x['primary_score'], reverse=True)
        
        # Print comparison table
        print(f"\n📊 Model Performance Comparison:")
        print("-" * 60)
        print(f"{'Rank':<6} {'Model':<25} {'Score':<15} {'Metric':<15}")
        print("-" * 60)
        
        for i, data in enumerate(comparison_data, 1):
            print(f"{i:<6} {data['name']:<25} {data['primary_score']:.4f}        {data['primary_metric']}")
        
        print("-" * 60)
        
        # Print improvement analysis
        if len(comparison_data) >= 2:
            best = comparison_data[0]
            baseline = comparison_data[-1] if len(comparison_data) > 1 else comparison_data[0]
            
            if best['name'] != baseline['name']:
                improvement = ((best['primary_score'] - baseline['primary_score']) / baseline['primary_score']) * 100
                print(f"\n🎯 Best Model ({best['name']}) vs Baseline ({baseline['name']}):")
                print(f"  Score Improvement: +{improvement:.1f}%")
                print(f"  {baseline['name']}: {baseline['primary_score']:.4f}")
                print(f"  {best['name']}: {best['primary_score']:.4f}")
        
        print("="*60)
    
    def comprehensive_model_comparison(self, base_model_path: str, adapter_path: str, 
                                     fused_model_path: str = None, test_data_path: str = None) -> Dict[str, Any]:
        """Comprehensive comparison of base model, runtime adapters, and fused model.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory  
            fused_model_path: Path to fused model (optional)
            test_data_path: Path to test data (optional)
            
        Returns:
            Dictionary with all evaluation results and comparison
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("Base Model vs Runtime LoRA vs Fused Model")
        print("="*60)
        
        results = {}
        model_names = []
        
        # 1. Evaluate base model
        print("\n🔸 Step 1: Evaluating Base Model")
        try:
            base_results = self.evaluate_model_from_path(base_model_path, "base_model", test_data_path)
            results["base_model"] = base_results
            model_names.append("base_model")
        except Exception as e:
            print(f"❌ Failed to evaluate base model: {e}")
        
        # 2. Evaluate base model + runtime adapters
        print("\n🔸 Step 2: Evaluating Base Model + Runtime LoRA Adapters")
        try:
            runtime_results = self.evaluate_model_with_adapters(
                base_model_path, adapter_path, "lora_runtime", test_data_path
            )
            results["lora_runtime"] = runtime_results
            model_names.append("lora_runtime")
        except Exception as e:
            print(f"❌ Failed to evaluate runtime adapters: {e}")
        
        # 3. Evaluate fused model (if available)
        if fused_model_path and Path(fused_model_path).exists():
            print("\n🔸 Step 3: Evaluating Fused Model")
            try:
                fused_results = self.evaluate_model_from_path(fused_model_path, "lora_fused", test_data_path)
                results["lora_fused"] = fused_results
                model_names.append("lora_fused")
            except Exception as e:
                print(f"❌ Failed to evaluate fused model: {e}")
        else:
            print("\n🔸 Step 3: Skipping Fused Model (not available)")
            if fused_model_path:
                print(f"  Path not found: {fused_model_path}")
        
        # 4. Compare all models
        if len(results) >= 2:
            print("\n🔸 Step 4: Model Comparison")
            self.compare_models(
                [results[name] for name in model_names if name in results],
                [name for name in model_names if name in results]
            )
            
            # 5. Fusion verification (if we have both runtime and fused)
            if "lora_runtime" in results and "lora_fused" in results:
                print("\n🔸 Step 5: Fusion Verification")
                self._verify_fusion_quality(results["lora_runtime"], results["lora_fused"])
        else:
            print("\n⚠️  Not enough models evaluated for comparison")
        
        # Save comprehensive results
        self._save_comprehensive_results(results)
        
        return results
    
    def _verify_fusion_quality(self, runtime_results: Dict[str, Any], fused_results: Dict[str, Any]):
        """Verify that fusion preserved the adapter behavior.
        
        Args:
            runtime_results: Results from base + runtime adapters
            fused_results: Results from fused model
        """
        print("\n" + "="*60)
        print("FUSION QUALITY VERIFICATION")
        print("="*60)
        
        runtime_metrics = runtime_results['metrics']
        fused_metrics = fused_results['metrics']
        
        # Compare primary scores
        if 'bertscore' in runtime_metrics and runtime_metrics['bertscore']:
            runtime_score = runtime_metrics['bertscore']['f1']['mean']
            fused_score = fused_metrics['bertscore']['f1']['mean']
            metric_name = "BERTScore F1"
        else:
            runtime_score = runtime_metrics['word_overlap']['mean']
            fused_score = fused_metrics['word_overlap']['mean']
            metric_name = "Word Overlap"
        
        score_diff = abs(runtime_score - fused_score)
        relative_diff = (score_diff / runtime_score) * 100 if runtime_score > 0 else 0
        
        print(f"📊 {metric_name} Comparison:")
        print(f"  Runtime Adapters: {runtime_score:.4f}")
        print(f"  Fused Model:      {fused_score:.4f}")
        print(f"  Absolute Diff:    {score_diff:.4f}")
        print(f"  Relative Diff:    {relative_diff:.2f}%")
        
        # Fusion quality assessment
        if relative_diff < 1.0:  # Less than 1% difference
            print("✅ Excellent fusion quality - models are essentially identical")
        elif relative_diff < 3.0:  # Less than 3% difference
            print("✅ Good fusion quality - minor differences within acceptable range")
        elif relative_diff < 5.0:  # Less than 5% difference
            print("⚠️  Acceptable fusion quality - some degradation detected")
        else:
            print("❌ Poor fusion quality - significant differences detected")
            print("   Consider checking fusion process or adapter scale parameter")
        
        # Compare additional metrics
        print(f"\n📈 Additional Metrics Comparison:")
        print(f"  Word Overlap Std Dev: Runtime {runtime_metrics['word_overlap']['std']:.4f} | Fused {fused_metrics['word_overlap']['std']:.4f}")
        print(f"  Word Overlap Range:   Runtime [{runtime_metrics['word_overlap']['min']:.3f}, {runtime_metrics['word_overlap']['max']:.3f}] | Fused [{fused_metrics['word_overlap']['min']:.3f}, {fused_metrics['word_overlap']['max']:.3f}]")
        
        print("="*60)
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive comparison results.
        
        Args:
            results: Dictionary with all evaluation results
        """
        from datetime import datetime
        
        results_dir = Path(self.paths_config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_file = results_dir / f"comprehensive_comparison_{timestamp}.json"
        
        # Create comparison summary
        comparison_summary = {
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": list(results.keys()),
            "evaluation_config": self.eval_config,
            "results": results
        }
        
        with open(comp_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        print(f"\n📁 Comprehensive results saved to: {comp_file}")
        return str(comp_file)