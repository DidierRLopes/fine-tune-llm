#!/usr/bin/env python3
"""
Model training script for fine-tuning pipeline.
Sets up LoRA configuration and trains the model.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import FineTuner


def main():
    parser = argparse.ArgumentParser(description="Train model with LoRA fine-tuning")
    parser.add_argument(
        "--model-config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--training-config", 
        type=str, 
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--train-data", 
        type=str, 
        default="data/processed/train.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--val-data", 
        type=str, 
        default="data/processed/val.json",
        help="Path to validation data JSON file"
    )
    parser.add_argument(
        "--test-data", 
        type=str, 
        default="data/processed/test.json",
        help="Path to test data JSON file (only used for final evaluation)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not Path(args.train_data).exists():
        print(f"❌ Training data file not found: {args.train_data}")
        print("Run '01_prepare_data.py' first to prepare the data.")
        sys.exit(1)
    
    if not Path(args.val_data).exists():
        print(f"❌ Validation data file not found: {args.val_data}")
        print("Run '01_prepare_data.py' first to prepare the data.")
        sys.exit(1)
    
    print("")
    print("="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"\nModel config: {args.model_config}")
    print(f"Training config: {args.training_config}")
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Test data: {args.test_data} (for final evaluation only)")
    
    if args.resume:
        print("Resume mode: Will attempt to resume from last checkpoint")
    
    try:
        # Initialize fine-tuner
        fine_tuner = FineTuner(args.model_config, args.training_config)
        
        # Run the complete training pipeline
        model, tokenizer, training_summary = fine_tuner.run_full_training_pipeline(
            args.train_data, args.val_data
        )
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n📁 Output Files:")
        print("  Adapters: models/adapters/")
        print("  Training logs: logs/training/")
        print("  Metrics plot: logs/training/training_metrics.png")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()