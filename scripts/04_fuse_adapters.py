#!/usr/bin/env python3
"""
Adapter fusion script for fine-tuning pipeline.
Fuses LoRA adapters into the base model.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.fusion import AdapterFusion
import yaml


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters into base model")
    parser.add_argument(
        "--model-config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to model configuration file"
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
        "--test-fusion", 
        action="store_true",
        help="Test fusion quality with sample prompts"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="Clean up intermediate checkpoint files after fusion"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force fusion even if output directory exists"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ADAPTER FUSION PIPELINE")
    print("="*60)
    
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
    print(f"Output: {output_path}")
    
    # Check if output exists and handle force flag
    if Path(output_path).exists() and not args.force:
        print(f"\\n⚠️  Output directory already exists: {output_path}")
        response = input("Overwrite? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Fusion cancelled.")
            sys.exit(0)
        
        # Remove existing output directory
        import shutil
        shutil.rmtree(output_path)
        print(f"Removed existing directory: {output_path}")
    
    # Initialize fusion utility
    fusion = AdapterFusion()
    
    # Validate inputs
    print("\nStep 1: Validating fusion inputs...")
    if not fusion.validate_fusion_inputs(base_model_path, adapters_path):
        print("❌ Fusion input validation failed. Please check the paths and files.")
        sys.exit(1)
    
    # Perform fusion
    print("\nStep 2: Fusing LoRA adapters into base model...")
    try:
        fused_model_path = fusion.fuse_adapters(
            base_model_path, adapters_path, output_path, verbose=True
        )
        
        print(f"\\n✅ Fusion completed successfully!")
        print(f"Fused model saved to: {fused_model_path}")
        
    except Exception as e:
        print(f"\\n❌ Fusion failed: {e}")
        sys.exit(1)
    
    # Test fusion quality if requested
    if args.test_fusion:
        print("\nStep 3: Testing fusion quality...")
        try:
            test_results = fusion.compare_fusion_quality(
                adapters_path,  # Original model path (adapters)
                fused_model_path,  # Fused model path
            )
            
            if "error" in test_results:
                print(f"⚠️  Warning: Fusion test failed: {test_results['error']}")
            else:
                print("✅ Fusion quality test completed")
                
                for result in test_results["test_results"]:
                    prompt_preview = result["prompt"][:50] + "..."
                    success = "✅" if result["fused_success"] else "❌"
                    print(f"  {success} Prompt: {prompt_preview}")
                    
        except Exception as e:
            print(f"⚠️  Warning: Could not test fusion quality: {e}")
    
    # Cleanup checkpoint files if requested
    if args.cleanup:
        print("\nStep 4: Cleaning up checkpoint files...")
        try:
            fusion.cleanup_fusion_artifacts(adapters_path, keep_final=True)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Warning: Cleanup failed: {e}")
    
    # Create fusion report
    print("\nStep 5: Creating fusion report...")
    try:
        report_path = fusion.create_fusion_report(
            base_model_path, adapters_path, fused_model_path
        )
        print(f"✅ Fusion report created: {report_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create fusion report: {e}")
    
    # Get fused model info
    print("\nStep 6: Getting fused model information...")
    try:
        model_info = fusion.get_fusion_info(fused_model_path)
        print(f"\\n📊 Fused Model Info:")
        print(f"  Total files: {model_info['total_files']}")
        print(f"  Total size: {model_info['total_size_mb']} MB")
        print(f"  Model directory: {model_info['path']}")
    except Exception as e:
        print(f"⚠️  Warning: Could not get model info: {e}")
    
    print("\n" + "="*60)
    print("ADAPTER FUSION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"🎉 Fused model ready at: {fused_model_path}")
    print("\nNext steps:")
    print("  - Run '03_evaluate_model.py' to evaluate the fused model")
    print("  - Run '05_upload_model.py' to upload to HuggingFace (optional)")
    print("  - Use the fused model for inference")
    print("="*60)


if __name__ == "__main__":
    main()