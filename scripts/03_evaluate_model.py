#!/usr/bin/env python3
"""
Model evaluation script for fine-tuning pipeline.
Evaluates base model and runtime LoRA adapters.

This script evaluates:
1. Base model (baseline performance)  
2. Base model + runtime LoRA adapters (your fine-tuned performance)

For fused model evaluation, use 04_fuse_and_evaluate.py
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import ModelEvaluator
from evaluation.comparator import ModelComparator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/evaluation_config.yaml",
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--model-path", 
        type=str,
        help="Path to model to evaluate (if not specified, evaluates adapters and base model)"
    )
    parser.add_argument(
        "--base-model", 
        type=str, 
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Path to base model for comparison"
    )
    parser.add_argument(
        "--adapters-path", 
        type=str, 
        default="models/adapters",
        help="Path to LoRA adapters directory"
    )
    parser.add_argument(
        "--test-data", 
        type=str, 
        default="data/processed/test.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--compare-only", 
        action="store_true",
        help="Only compare existing evaluation results"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # Check if test data exists
    if not Path(args.test_data).exists():
        print(f"❌ Test data file not found: {args.test_data}")
        print("Run '01_prepare_data.py' first to prepare the data.")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config)
    
    if args.compare_only:
        print("Compare-only mode: Loading existing evaluation results...")
        
        # Try to load existing results
        try:
            import json
            results_dir = Path("logs/evaluation")
            
            result_files = list(results_dir.glob("*_evaluation.json"))
            if not result_files:
                print("❌ No evaluation results found in logs/evaluation/")
                print("Run evaluation first without --compare-only flag.")
                sys.exit(1)
            
            results = []
            model_names = []
            
            for result_file in result_files:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                results.append(result)
                model_names.append(result_file.stem.replace("_evaluation", ""))
            
            # Compare models
            evaluator.compare_models(results, model_names)
            
            # Create detailed comparison report
            comparator = ModelComparator()
            comparator.create_comparison_report(results, model_names)
            
            print("✅ Comparison completed!")
            
        except Exception as e:
            print(f"❌ Error loading evaluation results: {e}")
            sys.exit(1)
        
        return
    
    evaluation_results = []
    model_names = []
    
    # Evaluate specific model if provided
    if args.model_path:
        print(f"\\nEvaluating specific model: {args.model_path}")
        
        if not Path(args.model_path).exists():
            print(f"❌ Model path does not exist: {args.model_path}")
            sys.exit(1)
        
        model_name = Path(args.model_path).name
        results = evaluator.evaluate_model_from_path(args.model_path, model_name, args.test_data)
        evaluation_results.append(results)
        model_names.append(model_name)
    
    else:
        # Evaluate base model + runtime adapters only (no fused model)
        print("Standard evaluation: Base + Runtime LoRA Adapters")
        
        base_model_path = args.base_model or "microsoft/Phi-3-mini-4k-instruct"
        
        # 1. Evaluate base model
        print("\n🔸 Step 1: Evaluating Base Model")
        try:
            base_results = evaluator.evaluate_model_from_path(base_model_path, "base_model", args.test_data)
            evaluation_results.append(base_results)
            model_names.append("base_model")
        except Exception as e:
            print(f"❌ Failed to evaluate base model: {e}")
        
        # 2. Evaluate base model + runtime adapters
        if Path(args.adapters_path).exists():
            print("\n🔸 Step 2: Evaluating Base Model + Runtime LoRA Adapters")
            try:
                runtime_results = evaluator.evaluate_model_with_adapters(
                    base_model_path, args.adapters_path, "lora_runtime", args.test_data
                )
                evaluation_results.append(runtime_results)
                model_names.append("lora_runtime")
            except Exception as e:
                print(f"❌ Failed to evaluate runtime adapters: {e}")
        else:
            print(f"\n❌ Adapters not found at {args.adapters_path}")
            print("Run '02_train_model.py' first to train the model.")
        
        # 3. Compare models
        if len(evaluation_results) >= 2:
            print("\n🔸 Step 3: Model Comparison")
            evaluator.compare_models(evaluation_results, model_names)
            
            # Create detailed comparison report with plots (but don't print comparison again)
            from evaluation.comparator import ModelComparator
            comparator = ModelComparator()
            report_path = comparator.create_comparison_report(evaluation_results, model_names)
            print(f"\n📊 Detailed comparison plots saved to: {report_path}")
        
        elif len(evaluation_results) == 1:
            print("\n✅ Single model evaluation completed!")
            print("For comparison, run again with multiple models or use existing evaluation results.")
    
        else:
            print("\n❌ No models were successfully evaluated.")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("EVALUATION PIPELINE COMPLETED!")
    print("="*60)
    print("📁 Output files:")
    print("  Evaluation results: logs/evaluation/")
    print("  Comparison plots: logs/evaluation/")
    print("\n🎉 Evaluation completed! Check the results in the logs directory.")


if __name__ == "__main__":
    main()