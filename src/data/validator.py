"""Data validation utilities for fine-tuning pipeline."""

from typing import List, Dict, Any


class DataValidator:
    """Validate data format and quality."""
    
    def __init__(self, required_tokens: List[str]):
        """Initialize validator.
        
        Args:
            required_tokens: List of required tokens for format validation
        """
        self.required_tokens = required_tokens
    
    def validate_sample_format(self, sample: str) -> Dict[str, Any]:
        """Validate a single sample format.
        
        Args:
            sample: Formatted sample string
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        
        if not isinstance(sample, str):
            errors.append(f"Sample is not a string: {type(sample)}")
            return {"valid": False, "errors": errors}
        
        # Check for all required tokens
        missing_tokens = []
        for token in self.required_tokens:
            if token not in sample:
                missing_tokens.append(token)
        
        if missing_tokens:
            errors.append(f"Missing required tokens: {missing_tokens}")
        
        # Check token order for Phi-3 format
        token_positions = {}
        for token in self.required_tokens:
            if token in sample:
                token_positions[token] = sample.index(token)
        
        # Expected order: system, user, assistant, end
        expected_order = ["<|system|>", "<|user|>", "<|assistant|>"]
        for i in range(len(expected_order) - 1):
            current_token = expected_order[i]
            next_token = expected_order[i + 1]
            
            if (current_token in token_positions and 
                next_token in token_positions and
                token_positions[current_token] > token_positions[next_token]):
                errors.append(f"Incorrect token order: {current_token} should come before {next_token}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "token_positions": token_positions
        }
    
    def validate_dataset(self, samples: List[str], sample_size: int = None) -> Dict[str, Any]:
        """Validate a dataset of samples.
        
        Args:
            samples: List of formatted samples
            sample_size: Number of samples to validate (None for all)
            
        Returns:
            Dictionary with validation results
        """
        if sample_size is None:
            samples_to_check = samples
        else:
            samples_to_check = samples[:sample_size]
        
        total_samples = len(samples_to_check)
        valid_samples = 0
        all_errors = []
        
        print(f"Validating {total_samples} samples...")
        
        for i, sample in enumerate(samples_to_check):
            result = self.validate_sample_format(sample)
            
            if result["valid"]:
                valid_samples += 1
            else:
                all_errors.extend([f"Sample {i}: {error}" for error in result["errors"]])
        
        validation_result = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "invalid_samples": total_samples - valid_samples,
            "validation_rate": valid_samples / total_samples if total_samples > 0 else 0,
            "errors": all_errors
        }
        
        return validation_result
    
    def print_validation_report(self, validation_result: Dict[str, Any]):
        """Print a formatted validation report.
        
        Args:
            validation_result: Result from validate_dataset
        """       
        total = validation_result["total_samples"]
        valid = validation_result["valid_samples"]
        invalid = validation_result["invalid_samples"]
        rate = validation_result["validation_rate"]
        
        print(f"\n📊 Validation Summary:")
        print(f"  Total samples checked: {total}")
        print(f"  Valid samples:         {valid}")
        print(f"  Invalid samples:       {invalid}")
        print(f"  Validation rate:       {rate:.1%}")
        
        if invalid > 0:
            print(f"\n⚠️  Found {invalid} invalid samples:")
            errors = validation_result["errors"]
            
            # Show first 10 errors
            for error in errors[:10]:
                print(f"  - {error}")
            
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("\n✅ All samples passed validation!")
        print("")
    
    def get_sample_statistics(self, samples: List[str]) -> Dict[str, Any]:
        """Get basic statistics about the samples.
        
        Args:
            samples: List of formatted samples
            
        Returns:
            Dictionary with sample statistics
        """
        if not samples:
            return {"error": "No samples provided"}
        
        lengths = [len(sample) for sample in samples]
        
        stats = {
            "total_samples": len(samples),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_characters": sum(lengths)
        }
        
        return stats
    
    def print_sample_statistics(self, samples: List[str]):
        """Print sample statistics.
        
        Args:
            samples: List of formatted samples
        """
        stats = self.get_sample_statistics(samples)
        
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return
        
        print(f"\n📈 Sample Statistics:")
        print(f"  Total samples:     {stats['total_samples']:,}")
        print(f"  Average length:    {stats['avg_length']:.0f} characters")
        print(f"  Min length:        {stats['min_length']:,} characters")
        print(f"  Max length:        {stats['max_length']:,} characters")
        print(f"  Total characters:  {stats['total_characters']:,}")