#!/usr/bin/env python3
"""
Comprehensive evaluation script.
Evaluates and compares all three model states:
1. Base model
2. Base model + runtime adapters  
3. Fused model (if available)

This script focuses only on evaluation and comparison.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation and comparison")
    parser.add_argument(
        "--eval-config", 
        type=str, 
        default="config/evaluation_config.yaml",
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--test-data", 
        type=str, 
        default="data/processed/test.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--base-model", 
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--adapters-path", 
        type=str,
        default="models/adapters",
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--fused-model-path", 
        type=str,
        default="models/fused",
        help="Path to fused model"
    )
    parser.add_argument(
        "--skip-base", 
        action="store_true",
        help="Skip base model evaluation"
    )
    parser.add_argument(
        "--skip-runtime", 
        action="store_true",
        help="Skip runtime adapter evaluation"
    )
    parser.add_argument(
        "--skip-fused", 
        action="store_true",
        help="Skip fused model evaluation"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Adapters: {args.adapters_path}")
    print(f"Fused model: {args.fused_model_path}")
    print(f"Test data: {args.test_data}")
    
    # Check if test data exists
    if not Path(args.test_data).exists():
        print(f"❌ Test data file not found: {args.test_data}")
        print("Run '01_prepare_data.py' first to prepare the data.")
        sys.exit(1)
    
    # Check if adapters exist (unless skipping runtime)
    if not args.skip_runtime and not Path(args.adapters_path).exists():
        print(f"❌ Adapters not found at {args.adapters_path}")
        print("Run '02_train_model.py' first to train the model.")
        print("Or use --skip-runtime to skip adapter evaluation.")
        sys.exit(1)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.eval_config)
        
        # Determine what to evaluate based on flags
        if args.skip_base and args.skip_runtime and args.skip_fused:
            print("❌ All evaluations skipped. Nothing to do.")
            sys.exit(1)
        
        # Override fused model path to None if skipping
        fused_path = None if args.skip_fused else args.fused_model_path
        
        # Run comprehensive comparison
        results = evaluator.comprehensive_model_comparison(
            base_model_path=args.base_model,
            adapter_path=args.adapters_path,
            fused_model_path=fused_path,
            test_data_path=args.test_data
        )
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION COMPLETED!")
        print("="*80)
        
        # Print summary
        models_evaluated = list(results.keys())
        print(f"\n📊 Models Evaluated: {', '.join(models_evaluated)}")
        
        if len(results) >= 2:
            # Show key results
            print(f"\n🎯 Key Results:")
            for model_name, result in results.items():
                metrics = result['metrics']
                if 'bertscore' in metrics and metrics['bertscore']:
                    score = metrics['bertscore']['f1']['mean']
                    metric_type = "BERTScore F1"
                else:
                    score = metrics['word_overlap']['mean']
                    metric_type = "Word Overlap"
                print(f"  {model_name}: {score:.4f} ({metric_type})")
        
        print(f"\n📁 Detailed results saved to: logs/evaluation/")
        
        # Recommendations based on results
        if "lora_runtime" in results and "lora_fused" in results:
            print("\n💡 Fusion Quality Assessment:")
            runtime_score = results["lora_runtime"]['metrics']['word_overlap']['mean']
            fused_score = results["lora_fused"]['metrics']['word_overlap']['mean']
            diff = abs(runtime_score - fused_score) / runtime_score * 100
            
            if diff < 1.0:
                print("  ✅ Excellent - Fusion preserved adapter performance perfectly")
            elif diff < 3.0:
                print("  ✅ Good - Minor differences, fusion quality is acceptable")
            else:
                print("  ⚠️  Warning - Significant differences detected in fusion")
                print("     Consider re-running fusion or adjusting adapter scale")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()