"""Model utilities for loading, saving, and managing models."""

import os
import shutil
from pathlib import Path
from typing import Tuple, Any, Dict
from mlx_lm import load


class ModelManager:
    """Manage model loading, saving, and organization."""
    
    def __init__(self, base_dir: str = "models"):
        """Initialize model manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: str, verbose: bool = True) -> Tuple[Any, Any]:
        """Load model and tokenizer.
        
        Args:
            model_path: Path to model (local or HuggingFace)
            verbose: Whether to print loading information
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if verbose:
            print(f"Loading model from: {model_path}")
        
        try:
            model, tokenizer = load(model_path)
            
            if verbose:
                print("Model loaded successfully")
                
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def copy_model(self, source_path: str, dest_path: str, overwrite: bool = False):
        """Copy model directory to new location.
        
        Args:
            source_path: Source model directory
            dest_path: Destination directory
            overwrite: Whether to overwrite existing destination
        """
        source = Path(source_path)
        dest = Path(dest_path)
        
        if dest.exists() and not overwrite:
            raise ValueError(f"Destination {dest} already exists. Use overwrite=True to replace.")
        
        if dest.exists() and overwrite:
            shutil.rmtree(dest)
        
        print(f"Copying model from {source} to {dest}")
        shutil.copytree(source, dest)
        print("Model copy completed")
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a model directory.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary with model information
        """
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            return {"error": f"Model directory {model_path} does not exist"}
        
        info = {
            "path": str(model_dir.absolute()),
            "exists": True,
            "files": []
        }
        
        # List all files in the model directory
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_dir)
                file_size = file_path.stat().st_size
                info["files"].append({
                    "name": str(relative_path),
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })
        
        # Calculate total size
        total_size = sum(f["size_bytes"] for f in info["files"])
        info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        info["total_files"] = len(info["files"])
        
        return info
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all models in the managed directory.
        
        Returns:
            Dictionary mapping model names to their info
        """
        models = {}
        
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                models[model_dir.name] = self.get_model_info(str(model_dir))
        
        return models
    
    def cleanup_checkpoints(self, adapter_dir: str, keep_last: int = 3):
        """Clean up old checkpoint files, keeping only the most recent ones.
        
        Args:
            adapter_dir: Directory containing adapter checkpoints
            keep_last: Number of recent checkpoints to keep
        """
        adapter_path = Path(adapter_dir)
        
        if not adapter_path.exists():
            print(f"Adapter directory {adapter_dir} does not exist")
            return
        
        # Find all checkpoint files
        checkpoint_files = list(adapter_path.glob("*_adapters.safetensors"))
        
        if len(checkpoint_files) <= keep_last:
            print(f"Found {len(checkpoint_files)} checkpoints, keeping all")
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove old checkpoints
        files_to_remove = checkpoint_files[:-keep_last]
        
        print(f"Cleaning up {len(files_to_remove)} old checkpoints, keeping {keep_last} most recent")
        
        for file_path in files_to_remove:
            print(f"  Removing: {file_path.name}")
            file_path.unlink()
        
        print("Checkpoint cleanup completed")
    
    def validate_model_structure(self, model_path: str) -> Dict[str, Any]:
        """Validate that a model directory has the expected structure.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary with validation results
        """
        model_dir = Path(model_path)
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "files_found": []
        }
        
        if not model_dir.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Model directory does not exist: {model_path}")
            return validation_result
        
        # Expected files for MLX models
        expected_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        # Check for model weight files
        weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("model*.npz"))
        
        if not weight_files:
            validation_result["errors"].append("No model weight files found (.safetensors or .npz)")
            validation_result["valid"] = False
        else:
            validation_result["files_found"].extend([f.name for f in weight_files])
        
        # Check for expected configuration files
        for expected_file in expected_files:
            file_path = model_dir / expected_file
            if file_path.exists():
                validation_result["files_found"].append(expected_file)
            else:
                validation_result["warnings"].append(f"Expected file not found: {expected_file}")
        
        return validation_result
    
    def print_model_info(self, model_path: str):
        """Print formatted model information.
        
        Args:
            model_path: Path to model directory
        """
        info = self.get_model_info(model_path)
        
        if "error" in info:
            print(f"Error: {info['error']}")
            return
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Path: {info['path']}")
        print(f"Total files: {info['total_files']}")
        print(f"Total size: {info['total_size_mb']} MB")
        
        # Group files by type
        file_types = {}
        for file_info in info["files"]:
            ext = Path(file_info["name"]).suffix.lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(file_info)
        
        print(f"\nFiles by type:")
        for ext, files in sorted(file_types.items()):
            total_size = sum(f["size_mb"] for f in files)
            print(f"  {ext or 'no extension'}: {len(files)} files, {total_size:.1f} MB")
        
        print("="*60)
    
    def create_model_archive(self, model_path: str, archive_path: str = None):
        """Create a compressed archive of a model directory.
        
        Args:
            model_path: Path to model directory
            archive_path: Path for archive file (auto-generated if None)
        """
        import tarfile
        from datetime import datetime
        
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory does not exist: {model_path}")
        
        if archive_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"{model_dir.name}_{timestamp}.tar.gz"
        
        print(f"Creating archive: {archive_path}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(model_dir, arcname=model_dir.name)
        
        archive_size = Path(archive_path).stat().st_size / (1024 * 1024)
        print(f"Archive created successfully: {archive_path} ({archive_size:.1f} MB)")
        
        return archive_path