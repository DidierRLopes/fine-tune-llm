"""Data loading utilities for fine-tuning pipeline."""

import yaml
from datasets import load_dataset
from pathlib import Path
from typing import Dict, Any, List


class DataLoader:
    """Load and manage datasets for fine-tuning."""
    
    def __init__(self, config_path: str):
        """Initialize data loader with configuration.
        
        Args:
            config_path: Path to data configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.formatting_config = self.config['formatting']
    
    def load_raw_dataset(self) -> Dict[str, Any]:
        """Load raw dataset from HuggingFace.
        
        Returns:
            Raw dataset dictionary
        """
        print(f"Loading dataset: {self.dataset_config['name']}")
        dataset = load_dataset(self.dataset_config['name'])
        print(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
        
        # Get the train dataset (assuming it's the main split)
        if 'train' in dataset:
            data = dataset['train']
        else:
            # If no train split, use the first available split
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"No 'train' split found, using '{split_name}' split")
        
        print(f"Dataset size: {len(data)} samples")
        print(f"Dataset features: {data.features}")
        
        return data
    
    def get_formatting_config(self) -> Dict[str, Any]:
        """Get formatting configuration.
        
        Returns:
            Formatting configuration dictionary
        """
        return self.formatting_config
    
    def get_test_split_ratio(self) -> float:
        """Get test split ratio.
        
        Returns:
            Test split ratio as float
        """
        return self.dataset_config['test_split']
    
    def get_val_split_ratio(self) -> float:
        """Get validation split ratio.
        
        Returns:
            Validation split ratio as float
        """
        return self.dataset_config.get('val_split', 0.1)
    
    def save_processed_data(self, train_data: List[str], val_data: List[str], test_data: List[str]):
        """Save processed data to JSON files.
        
        Args:
            train_data: List of formatted training samples
            val_data: List of formatted validation samples
            test_data: List of formatted test samples
        """
        import json
        
        # Ensure processed data directory exists
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        train_path = processed_dir / "train.json"
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Training data saved to: {train_path}")
        
        # Save validation data
        val_path = processed_dir / "val.json"
        with open(val_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Validation data saved to: {val_path}")
        
        # Save test data
        test_path = processed_dir / "test.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Test data saved to: {test_path}")
        
        # Save data statistics
        total_size = len(train_data) + len(val_data) + len(test_data)
        stats = {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "total_size": total_size,
            "train_split_ratio": len(train_data) / total_size,
            "val_split_ratio": len(val_data) / total_size,
            "test_split_ratio": len(test_data) / total_size
        }
        
        stats_path = processed_dir / "data_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Data statistics saved to: {stats_path}")
        
        return train_path, val_path, test_path, stats_path