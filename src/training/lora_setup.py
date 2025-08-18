"""LoRA setup utilities for fine-tuning pipeline."""

import json
import yaml
from pathlib import Path
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers, TrainingArgs
from mlx.utils import tree_flatten
import mlx.optimizers as optim
from typing import Dict, Any, Tuple


class LoRASetup:
    """Setup and configure LoRA for fine-tuning."""
    
    def __init__(self, model_config_path: str, training_config_path: str):
        """Initialize LoRA setup.
        
        Args:
            model_config_path: Path to model configuration YAML file
            training_config_path: Path to training configuration YAML file
        """
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        with open(training_config_path, 'r') as f:
            self.training_config = yaml.safe_load(f)
    
    def load_base_model(self):
        """Load the base model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.model_config['base_model']['path']
        print(f"Loading base model: {model_path}\n")
        
        model, tokenizer = load(model_path)
        print(f"\nModel loaded successfully\n")
        
        return model, tokenizer
    
    def setup_lora_config(self) -> Dict[str, Any]:
        """Setup LoRA configuration.
        
        Returns:
            LoRA configuration dictionary
        """
        lora_config = {
            "num_layers": self.model_config['lora']['num_layers'],
            "lora_layers": self.model_config['lora']['lora_layers'],
            "lora_parameters": {
                "rank": self.model_config['lora']['rank'],
                "scale": self.model_config['lora']['scale'],
                "dropout": self.model_config['lora']['dropout'],
            }
        }
        
        # Add keys if specified
        if 'keys' in self.model_config['lora']:
            lora_config["lora_parameters"]["keys"] = self.model_config['lora']['keys']
        
        return lora_config
    
    def save_lora_config(self, lora_config: Dict[str, Any]):
        """Save LoRA configuration to adapter directory.
        
        Args:
            lora_config: LoRA configuration dictionary
        """
        adapter_dir = Path(self.model_config['paths']['adapter_dir'])
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = adapter_dir / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(lora_config, f, indent=4)
        
        print(f"LoRA config saved to: {config_path}")
        return config_path
    
    def setup_training_args(self) -> TrainingArgs:
        """Setup training arguments.
        
        Returns:
            TrainingArgs object
        """
        adapter_dir = Path(self.model_config['paths']['adapter_dir'])
        adapter_file = adapter_dir / "adapters.safetensors"
        
        training_args = TrainingArgs(
            adapter_file=str(adapter_file),
            iters=self.training_config['training']['iters'],
            steps_per_eval=self.training_config['training']['steps_per_eval'],
            batch_size=self.training_config['training']['batch_size'],
        )
        
        # Add grad_checkpoint if specified and supported
        if self.training_config['training'].get('grad_checkpoint', False):
            try:
                training_args.grad_checkpoint = True
            except AttributeError:
                print("Warning: grad_checkpoint not supported in this MLX version")
        
        return training_args
    
    def setup_optimizer(self):
        """Setup optimizer.
        
        Returns:
            MLX optimizer
        """
        learning_rate = float(self.training_config['training']['learning_rate'])
        optimizer_type = self.training_config['optimizer']['type'].lower()
        
        print(f"Setting up {optimizer_type} optimizer with learning rate: {learning_rate}")
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(learning_rate=learning_rate)
        else:
            print(f"Warning: Unknown optimizer type '{optimizer_type}', using Adam")
            optimizer = optim.Adam(learning_rate=learning_rate)
        
        return optimizer
    
    def apply_lora_to_model(self, model, lora_config: Dict[str, Any]):
        """Apply LoRA layers to the model.
        
        Args:
            model: Base model
            lora_config: LoRA configuration
        
        Returns:
            Model with LoRA layers applied
        """
        print("Freezing base model parameters...")
        model.freeze()
        
        print("Converting linear layers to LoRA layers...")
        linear_to_lora_layers(
            model,
            lora_config["lora_layers"],
            lora_config["lora_parameters"]
        )
        
        # Count trainable parameters
        num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
        
        print(f"LoRA setup completed:")
        print(f"  Trainable parameters: {num_train_params:,} ({num_train_params/total_params*100:.2f}%)")
        print(f"  Total parameters:     {total_params:,}")
        
        return model
    
    def print_configuration_summary(self, lora_config: Dict[str, Any], train_size: int, val_size: int):
        """Print a summary of the training configuration.
        
        Args:
            lora_config: LoRA configuration
            train_size: Number of training samples
            val_size: Number of validation samples
        """
        print("")
        print("="*60)
        print("TRAINING CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Dataset:")
        print(f"  Training samples:   {train_size}")
        print(f"  Validation samples: {val_size}")
        
        print(f"\n🔧 LoRA Configuration:")
        print(f"  Layers to adapt:  {lora_config['lora_layers']}/{lora_config['num_layers']}")
        print(f"  LoRA rank:        {lora_config['lora_parameters']['rank']}")
        print(f"  LoRA scale:       {lora_config['lora_parameters']['scale']}")
        print(f"  Dropout:          {lora_config['lora_parameters']['dropout']}")
        
        if 'keys' in lora_config['lora_parameters']:
            print(f"  Target layers:    {', '.join(lora_config['lora_parameters']['keys'])}")
        
        print(f"\n📈 Training Parameters:")
        print(f"  Iterations:       {self.training_config['training']['iters']}")
        print(f"  Batch size:       {self.training_config['training']['batch_size']}")
        print(f"  Learning rate:    {self.training_config['training']['learning_rate']}")
        print(f"  Eval frequency:   {self.training_config['training']['steps_per_eval']}")
        print(f"  Grad checkpoint:  {self.training_config['training'].get('grad_checkpoint', False)}")
    
    def complete_setup(self, train_size: int, val_size: int) -> Tuple[Any, Any, Dict[str, Any], TrainingArgs, Any]:
        """Complete setup process for LoRA fine-tuning.
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
        
        Returns:
            Tuple of (model, tokenizer, lora_config, training_args, optimizer)
        """
        # Load base model
        model, tokenizer = self.load_base_model()
        
        # Setup LoRA configuration
        lora_config = self.setup_lora_config()
        self.save_lora_config(lora_config)
        
        # Apply LoRA to model
        model = self.apply_lora_to_model(model, lora_config)
        
        # Setup training arguments and optimizer
        training_args = self.setup_training_args()
        optimizer = self.setup_optimizer()
        
        # Put model in training mode
        model.train()
        
        # Print configuration summary
        self.print_configuration_summary(lora_config, train_size, val_size)
        
        return model, tokenizer, lora_config, training_args, optimizer