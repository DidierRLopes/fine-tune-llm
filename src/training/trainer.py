"""Core training functionality for fine-tuning pipeline."""

import json
from pathlib import Path
from mlx_lm.tuner import train
from typing import List, Any, Dict
from .metrics import TrainingMetrics
from .lora_setup import LoRASetup


class FineTuner:
    """Main class for handling the fine-tuning process."""
    
    def __init__(self, model_config_path: str, training_config_path: str):
        """Initialize fine-tuner.
        
        Args:
            model_config_path: Path to model configuration YAML file
            training_config_path: Path to training configuration YAML file
        """
        self.lora_setup = LoRASetup(model_config_path, training_config_path)
        
        # Load training config for metrics setup
        import yaml
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        
        # Setup metrics with configuration
        metrics_config = training_config.get('metrics', {})
        logs_dir = training_config.get('paths', {}).get('logs_dir', 'logs/training')
        
        self.metrics = TrainingMetrics(
            patience=metrics_config.get('patience', 5),
            min_delta=metrics_config.get('min_delta', 0.001),
            logs_dir=logs_dir
        )
    
    def load_training_data(self, train_data_path: str, val_data_path: str) -> tuple:
        """Load training and validation data from JSON files.
        
        Args:
            train_data_path: Path to training data JSON file
            val_data_path: Path to validation data JSON file
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print(f"\nLoading training data from: {train_data_path}")
        with open(train_data_path, 'r') as f:
            train_dataset = json.load(f)
        
        print(f"Loading validation data from: {val_data_path}")
        with open(val_data_path, 'r') as f:
            val_dataset = json.load(f)
        
        print(f"Loaded {len(train_dataset)} training samples")
        print(f"Loaded {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset
    
    def setup_model_and_training(self, train_size: int, val_size: int):
        """Setup model and training configuration.
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            
        Returns:
            Tuple of (model, tokenizer, lora_config, training_args, optimizer)
        """
        return self.lora_setup.complete_setup(train_size, val_size)
    
    def train_model(self, model, tokenizer, training_args, optimizer, train_dataset: List[str], val_dataset: List[str] = None):
        """Train the model with LoRA.
        
        Args:
            model: Model with LoRA layers
            tokenizer: Model tokenizer
            training_args: Training arguments
            optimizer: MLX optimizer
            train_dataset: Training dataset (list of strings)
            val_dataset: Validation dataset (list of strings), optional
        """
        print("\nStarting training...")
        print(f"Training on {len(train_dataset)} samples")
        if val_dataset:
            print(f"Validating on {len(val_dataset)} samples")
        
        # Use MLX LM train function
        train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_callback=self.metrics,
        )
        
        print("Training completed!")
        
        # Save and display metrics
        self.metrics.save_metrics()
        self.metrics.plot_metrics()
        summary = self.metrics.print_final_summary()
        
        return summary
    
    def run_full_training_pipeline(self, train_data_path: str, val_data_path: str):
        """Run the complete training pipeline.
        
        Args:
            train_data_path: Path to training data JSON file
            val_data_path: Path to validation data JSON file
            
        Returns:
            Tuple of (model, tokenizer, training_summary)
        """
        print("")
        print("="*60)
        print("STARTING FINE-TUNING PIPELINE")
        print("="*60)
        
        # Load data
        train_dataset, val_dataset = self.load_training_data(train_data_path, val_data_path)
        
        # Setup model and training
        model, tokenizer, lora_config, training_args, optimizer = self.setup_model_and_training(
            len(train_dataset), len(val_dataset)
        )
        
        # Train model using proper validation set
        training_summary = self.train_model(
            model, tokenizer, training_args, optimizer, 
            train_dataset, val_dataset
        )
        
        return model, tokenizer, training_summary