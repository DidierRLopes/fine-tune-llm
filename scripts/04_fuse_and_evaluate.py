#!/usr/bin/env python3
"""
Fusion and comprehensive evaluation script.
This script fuses adapters and then evaluates all three model states:
1. Base model (baseline)
2. Base model + runtime adapters (fine-tuned performance)
3. Fused model (deployment-ready)

This verifies that fusion preserves the fine-tuning correctly and provides
numerical precision verification between runtime and fused models.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.fusion import AdapterFusion
from evaluation.evaluator import ModelEvaluator
import yaml


def main():
    parser = argparse.ArgumentParser(description="Fuse adapters and run comprehensive evaluation")
    parser.add_argument(
        "--model-config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
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
        help="Path to base model (overrides config)"
    )
    parser.add_argument(
        "--adapters-path", 
        type=str,
        help="Path to LoRA adapters (overrides config)"
    )
    parser.add_argument(
        "--output-path", 
        type=str,
        help="Path for fused model output (overrides config)"
    )
    parser.add_argument(
        "--skip-fusion", 
        action="store_true",
        help="Skip fusion step (assume fused model already exists)"
    )
    parser.add_argument(
        "--force-fusion", 
        action="store_true",
        help="Force fusion even if output directory exists"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("FUSION AND COMPREHENSIVE EVALUATION PIPELINE")
    print("="*60)
    print("This script: 1) Fuses adapters  2) Evaluates Base + Runtime + Fused")
    
    # Load model configuration
    try:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Model config file not found: {args.model_config}")
        sys.exit(1)
    
    # Get paths from config or arguments
    base_model_path = args.base_model or model_config['base_model']['path']
    adapters_path = args.adapters_path or model_config['paths']['adapter_dir']
    output_path = args.output_path or model_config['paths']['fused_model_dir']
    
    print(f"Base model: {base_model_path}")
    print(f"Adapters: {adapters_path}")
    print(f"Fused output: {output_path}")
    print(f"Test data: {args.test_data}")
    
    # Check if adapters exist
    if not Path(adapters_path).exists():
        print(f"❌ Adapters not found at {adapters_path}")
        print("Run '02_train_model.py' first to train the model.")
        sys.exit(1)
    
    # Check if test data exists
    if not Path(args.test_data).exists():
        print(f"❌ Test data file not found: {args.test_data}")
        print("Run '01_prepare_data.py' first to prepare the data.")
        sys.exit(1)
    
    # Step 1: Fusion (unless skipped)
    if not args.skip_fusion:
        print("\n" + "="*60)
        print("STEP 1: ADAPTER FUSION")
        print("="*60)
        
        # Check if output exists
        if Path(output_path).exists() and not args.force_fusion:
            print(f"⚠️  Fused model already exists at: {output_path}")
            response = input("Overwrite? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Using existing fused model...")
            else:
                # Remove existing output directory
                import shutil
                shutil.rmtree(output_path)
                print(f"Removed existing directory: {output_path}")
        
        # Perform fusion if needed
        if not Path(output_path).exists() or args.force_fusion:
            try:
                fusion = AdapterFusion()
                
                # Validate inputs
                if not fusion.validate_fusion_inputs(base_model_path, adapters_path):
                    print("❌ Fusion input validation failed.")
                    sys.exit(1)
                
                # Perform fusion
                fused_model_path = fusion.fuse_adapters(
                    base_model_path, adapters_path, output_path, verbose=True
                )
                
                print(f"✅ Fusion completed: {fused_model_path}")
                
            except Exception as e:
                print(f"❌ Fusion failed: {e}")
                sys.exit(1)
        else:
            print(f"✅ Using existing fused model: {output_path}")
    else:
        print(f"\nSkipping fusion step - using existing model at: {output_path}")
        if not Path(output_path).exists():
            print(f"❌ Fused model not found at {output_path}")
            print("Remove --skip-fusion flag to create it.")
            sys.exit(1)
    
    # Step 2: Comprehensive Evaluation (Base + Runtime + Fused)
    print("\n" + "="*60)
    print("STEP 2: COMPREHENSIVE EVALUATION")
    print("="*60)
    print("Evaluating: Base + Runtime LoRA + Fused")
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.eval_config)
        
        # Run comprehensive comparison (this is what makes 04 different from 03)
        results = evaluator.comprehensive_model_comparison(
            base_model_path=base_model_path,
            adapter_path=adapters_path,
            fused_model_path=output_path,
            test_data_path=args.test_data
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
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
        print(f"📁 Fused model available at: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()