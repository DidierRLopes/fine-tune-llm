"""LoRA adapter fusion utilities."""

import os
import subprocess
from pathlib import Path
from typing import Optional


class AdapterFusion:
    """Handle LoRA adapter fusion operations."""
    
    def __init__(self):
        """Initialize adapter fusion utility."""
        pass
    
    def fuse_adapters(self, base_model_path: str, adapter_path: str, 
                     output_path: str, verbose: bool = True) -> str:
        """Fuse LoRA adapters into base model.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory
            output_path: Path for fused model output
            verbose: Whether to show detailed output
            
        Returns:
            Path to fused model
        """
        # Disable tokenizer parallelism to prevent warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Prepare the fusion command
        cmd = [
            "mlx_lm.fuse",
            "--model", base_model_path,
            "--adapter-path", adapter_path,
            "--save-path", output_path
        ]
        
        if verbose:
            print("="*60)
            print("FUSING LORA ADAPTERS")
            print("="*60)
            print(f"Base model: {base_model_path}")
            print(f"Adapters: {adapter_path}")
            print(f"Output: {output_path}")
            print(f"Command: {' '.join(cmd)}")
            print("-" * 60)
        
        try:
            # Run the fusion command
            if verbose:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if verbose:
                print("-" * 60)
                print("✅ Fusion completed successfully!")
                print(f"Fused model saved to: {output_path}")
                print("="*60)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Fusion failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError: {e.stderr}"
            if e.stdout:
                error_msg += f"\nOutput: {e.stdout}"
            
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        except FileNotFoundError:
            error_msg = ("mlx_lm.fuse command not found. Make sure MLX LM is properly installed.\n"
                        "Try: pip install mlx-lm")
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def validate_fusion_inputs(self, base_model_path: str, adapter_path: str) -> bool:
        """Validate inputs before fusion.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory
            
        Returns:
            True if inputs are valid
        """
        errors = []
        
        # Check base model path
        if not Path(base_model_path).exists():
            errors.append(f"Base model path does not exist: {base_model_path}")
        
        # Check adapter path
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            errors.append(f"Adapter path does not exist: {adapter_path}")
        elif adapter_dir.is_dir():
            # Check for required adapter files
            adapter_config = adapter_dir / "adapter_config.json"
            adapter_weights = adapter_dir / "adapters.safetensors"
            
            if not adapter_config.exists():
                errors.append(f"adapter_config.json not found in {adapter_path}")
            
            if not adapter_weights.exists():
                errors.append(f"adapters.safetensors not found in {adapter_path}")
        
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Fusion inputs validated successfully")
        return True
    
    def compare_fusion_quality(self, original_model_path: str, fused_model_path: str,
                             test_prompts: Optional[list] = None) -> dict:
        """Compare quality between original LoRA model and fused model.
        
        Args:
            original_model_path: Path to model with LoRA adapters
            fused_model_path: Path to fused model
            test_prompts: Optional list of test prompts
            
        Returns:
            Dictionary with comparison results
        """
        from mlx_lm import load, generate
        
        if test_prompts is None:
            test_prompts = [
                "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat is OpenBB?<|end|>\n<|assistant|>",
                "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nExplain LoRA fine-tuning.<|end|>\n<|assistant|>"
            ]
        
        print("Loading models for comparison...")
        
        try:
            # Load original model (this might fail if it's just adapters)
            print("Loading fused model...")
            fused_model, fused_tokenizer = load(fused_model_path)
            
            comparison_results = {
                "fused_model_loaded": True,
                "original_model_loaded": False,
                "test_results": []
            }
            
            # Test fused model
            for i, prompt in enumerate(test_prompts):
                print(f"Testing prompt {i+1}...")
                
                try:
                    fused_response = generate(fused_model, fused_tokenizer, prompt, max_tokens=100)
                    
                    test_result = {
                        "prompt_index": i,
                        "prompt": prompt[:50] + "...",
                        "fused_response": fused_response,
                        "fused_success": True
                    }
                    
                except Exception as e:
                    test_result = {
                        "prompt_index": i,
                        "prompt": prompt[:50] + "...",
                        "fused_response": f"Error: {e}",
                        "fused_success": False
                    }
                
                comparison_results["test_results"].append(test_result)
            
            print("✅ Fused model comparison completed")
            return comparison_results
            
        except Exception as e:
            print(f"❌ Error during fusion comparison: {e}")
            return {"error": str(e)}
    
    def get_fusion_info(self, fused_model_path: str) -> dict:
        """Get information about a fused model.
        
        Args:
            fused_model_path: Path to fused model
            
        Returns:
            Dictionary with model information
        """
        import os
        
        model_path = Path(fused_model_path)
        
        if not model_path.exists():
            return {"error": f"Model path does not exist: {fused_model_path}"}
        
        # Get basic file information
        total_size = 0
        total_files = 0
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
                total_files += 1
        
        info = {
            "path": str(model_path),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "fusion_info": {
                "is_fused_model": True,
                "contains_adapters": False  # Fused models don't contain separate adapters
            }
        }
        
        return info
    
    def cleanup_fusion_artifacts(self, adapter_path: str, keep_final: bool = True):
        """Clean up intermediate fusion artifacts.
        
        Args:
            adapter_path: Path to adapter directory
            keep_final: Whether to keep the final adapters.safetensors file
        """
        adapter_dir = Path(adapter_path)
        
        if not adapter_dir.exists():
            print(f"Adapter directory does not exist: {adapter_path}")
            return
        
        # Find checkpoint files
        checkpoint_files = list(adapter_dir.glob("*_adapters.safetensors"))
        
        files_removed = 0
        
        for checkpoint_file in checkpoint_files:
            # Don't remove the final adapters.safetensors file if keep_final is True
            if keep_final and checkpoint_file.name == "adapters.safetensors":
                continue
            
            print(f"Removing checkpoint: {checkpoint_file.name}")
            checkpoint_file.unlink()
            files_removed += 1
        
        print(f"Cleaned up {files_removed} checkpoint files")
        
        if keep_final:
            final_adapter = adapter_dir / "adapters.safetensors"
            if final_adapter.exists():
                print(f"Kept final adapter file: {final_adapter}")
    
    def create_fusion_report(self, base_model_path: str, adapter_path: str, 
                           fused_model_path: str, output_file: str = None) -> str:
        """Create a report about the fusion process.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapters
            fused_model_path: Path to fused model
            output_file: Path to output report file
            
        Returns:
            Path to report file
        """
        import json
        from datetime import datetime
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"logs/fusion_report_{timestamp}.json"
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "fusion_timestamp": datetime.now().isoformat(),
            "base_model": {
                "path": base_model_path,
                "exists": Path(base_model_path).exists()
            },
            "adapters": {
                "path": adapter_path,
                "exists": Path(adapter_path).exists(),
                "info": self.get_fusion_info(adapter_path) if Path(adapter_path).exists() else None
            },
            "fused_model": {
                "path": fused_model_path,
                "exists": Path(fused_model_path).exists(),
                "info": self.get_fusion_info(fused_model_path) if Path(fused_model_path).exists() else None
            }
        }
        
        # Add fusion validation
        report["validation"] = {
            "inputs_valid": self.validate_fusion_inputs(base_model_path, adapter_path),
            "fused_model_exists": Path(fused_model_path).exists()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Fusion report saved to: {output_file}")
        return output_file