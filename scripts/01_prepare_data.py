#!/usr/bin/env python3
"""
Data preparation script for fine-tuning pipeline.
Loads, processes, validates, and saves training data.
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.validator import DataValidator


def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--preview", 
        action="store_true",
        help="Show preview of processed samples"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate existing processed data"
    )
    
    args = parser.parse_args()
    
    print("")
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Initialize components
    data_loader = DataLoader(args.config)
    
    if args.validate_only:
        # Only validate existing data
        print("Validation-only mode: checking existing processed data...")
        
        try:
            
            with open("data/processed/train.json", 'r') as f:
                train_data = json.load(f)
            with open("data/processed/val.json", 'r') as f:
                val_data = json.load(f)
            with open("data/processed/test.json", 'r') as f:
                test_data = json.load(f)
            
            # Initialize validator
            validation_config = data_loader.config['validation']
            validator = DataValidator(validation_config['required_tokens'])
            
            # Validate data
            print("Validating training data...")
            train_validation = validator.validate_dataset(
                train_data, validation_config['sample_validation_size']
            )
            validator.print_validation_report(train_validation)
            
            print("Validating validation data...")
            val_validation = validator.validate_dataset(
                val_data, validation_config['sample_validation_size']
            )
            validator.print_validation_report(val_validation)
            
            print("Validating test data...")
            test_validation = validator.validate_dataset(
                test_data, validation_config['sample_validation_size']
            )
            validator.print_validation_report(test_validation)
            
            print("✅ Validation completed!")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Processed data files not found: {e}")
            print("Run without --validate-only to process raw data first.")
            sys.exit(1)
        
        return
    
    # Load raw data
    print("\n>>> Step 1: Loading raw dataset...\n")
    raw_data = data_loader.load_raw_dataset()
    
    # Initialize preprocessor
    formatting_config = data_loader.get_formatting_config()
    test_split_ratio = data_loader.get_test_split_ratio()
    val_split_ratio = data_loader.get_val_split_ratio()
    
    preprocessor = DataPreprocessor(
        system_prompt=formatting_config['system_prompt'],
        test_split_ratio=test_split_ratio,
        val_split_ratio=val_split_ratio
    )
    
    # Process data
    print("\n>>> Step 2: Processing and formatting data...\n")
    train_samples, val_samples, test_samples = preprocessor.process_dataset(raw_data)
    
    # Preview samples if requested
    if args.preview:
        print("\nPreviewing processed samples...")
        preprocessor.preview_samples(train_samples, num_samples=2)
        preprocessor.preview_samples(val_samples, num_samples=1)
        preprocessor.preview_samples(test_samples, num_samples=1)
    
    # Validate processed data
    print("\n>>> Step 3: Validating processed data...\n")
    validation_config = data_loader.config['validation']
    validator = DataValidator(validation_config['required_tokens'])
    
    # Validate training data
    print("Validating training data...")
    train_validation = validator.validate_dataset(
        train_samples, validation_config['sample_validation_size']
    )
    validator.print_validation_report(train_validation)
    
    # Validate validation data
    print("Validating validation data...")
    val_validation = validator.validate_dataset(
        val_samples, validation_config['sample_validation_size']
    )
    validator.print_validation_report(val_validation)
    
    # Validate test data
    print("Validating test data...")
    test_validation = validator.validate_dataset(
        test_samples, validation_config['sample_validation_size']
    )
    validator.print_validation_report(test_validation)
    
    # Check if validation passed
    if not (train_validation['validation_rate'] == 1.0 and 
            val_validation['validation_rate'] == 1.0 and 
            test_validation['validation_rate'] == 1.0):
        print("⚠️  Warning: Some samples failed validation. Please review the data format.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Data preparation aborted.")
            sys.exit(1)
    
    # Print sample statistics
    print("Sample statistics:")
    validator.print_sample_statistics(train_samples + val_samples + test_samples)
    
    # Save processed data
    print("\n>>> Step 4: Saving processed data...\n")
    train_path, val_path, test_path, stats_path = data_loader.save_processed_data(train_samples, val_samples, test_samples)
    
    print("")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Test data: {test_path}")
    print(f"Statistics: {stats_path}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print("")


if __name__ == "__main__":
    main()