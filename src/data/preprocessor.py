"""Data preprocessing utilities for fine-tuning pipeline."""

import random
from typing import List, Tuple, Dict, Any


class DataPreprocessor:
    """Preprocess raw data into training format."""
    
    def __init__(self, system_prompt: str, test_split_ratio: float = 0.1, val_split_ratio: float = 0.1):
        """Initialize preprocessor.
        
        Args:
            system_prompt: System prompt for chat format
            test_split_ratio: Ratio of data to use for testing
            val_split_ratio: Ratio of data to use for validation
        """
        self.system_prompt = system_prompt
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio = val_split_ratio
    
    def format_conversation_sample(self, question: str, answer: str) -> str:
        """Format a single Q&A pair into Phi-3 chat format.
        
        Args:
            question: User question
            answer: Assistant answer
            
        Returns:
            Formatted string in Phi-3 format
        """
        formatted_sample = (
            f"<|system|>\n{self.system_prompt}<|end|>\n"
            f"<|user|>\n{question}<|end|>\n"
            f"<|assistant|>\n{answer}<|end|>"
        )
        return formatted_sample
    
    def extract_conversations(self, raw_data) -> List[str]:
        """Extract and format conversations from raw dataset.
        
        Args:
            raw_data: Raw dataset from HuggingFace
            
        Returns:
            List of formatted conversation samples
        """
        all_samples = []
        
        # Convert to list of dictionaries for easier processing
        data_list = raw_data.to_list()
        
        for sample in data_list:
            conversation = sample["conversation"]
            conversation_samples = []
            
            # Process conversation pairs (user, assistant)
            for i in range(0, len(conversation), 2):
                user_message = conversation[i]
                assistant_message = conversation[i+1] if i+1 < len(conversation) else None
                
                if assistant_message:
                    question = user_message["content"]
                    answer = assistant_message["content"]
                    
                    formatted_sample = self.format_conversation_sample(question, answer)
                    conversation_samples.append(formatted_sample)
            
            all_samples.extend(conversation_samples)
        
        # Shuffle all samples for better distribution
        random.shuffle(all_samples)
        
        print(f"Extracted {len(all_samples)} conversation samples")
        return all_samples
    
    def create_train_val_test_split(self, samples: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Create train/validation/test split from samples.
        
        Args:
            samples: List of formatted samples
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        total_size = len(samples)
        test_size = int(total_size * self.test_split_ratio)
        val_size = int(total_size * self.val_split_ratio)
        train_size = total_size - test_size - val_size
        
        # Split the data
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        print(f"Data split created:")
        print(f"  Training samples: {len(train_samples)} ({len(train_samples)/total_size:.1%})")
        print(f"  Validation samples: {len(val_samples)} ({len(val_samples)/total_size:.1%})")
        print(f"  Test samples: {len(test_samples)} ({len(test_samples)/total_size:.1%})")
        
        return train_samples, val_samples, test_samples
    
    def process_dataset(self, raw_data) -> Tuple[List[str], List[str], List[str]]:
        """Complete processing pipeline from raw data to train/val/test splits.
        
        Args:
            raw_data: Raw dataset from HuggingFace
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        print("Starting data preprocessing...")
        
        # Extract and format conversations
        samples = self.extract_conversations(raw_data)
        
        # Create train/val/test split
        train_samples, val_samples, test_samples = self.create_train_val_test_split(samples)
        
        print("Data preprocessing completed successfully!")
        return train_samples, val_samples, test_samples
    
    def preview_samples(self, samples: List[str], num_samples: int = 3):
        """Preview formatted samples.
        
        Args:
            samples: List of formatted samples
            num_samples: Number of samples to preview
        """
        print(f"\n📝 Sample previews (first {num_samples}):")
        print("=" * 60)
        
        for i, sample in enumerate(samples[:num_samples]):
            print(f"\nSample {i+1}:")
            print("-" * 40)
            # Show first 200 characters of each sample
            preview_text = sample[:200] + "..." if len(sample) > 200 else sample
            print(preview_text)
        
        print("=" * 60)