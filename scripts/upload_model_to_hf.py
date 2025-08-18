#!/usr/bin/env python3
"""
Upload LoRA adapters to HuggingFace Hub.
Specialized script for uploading MLX LoRA adapter weights.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv


def validate_adapter_structure(adapter_path: Path) -> tuple[bool, list, float]:
    """Validate LoRA adapter directory structure.
    
    Args:
        adapter_path: Path to adapter directory
        
    Returns:
        Tuple of (is_valid, missing_files, total_size_mb)
    """
    required_files = [
        "adapter_config.json",
        "config.json",
        "adapters.safetensors"
    ]
    
    missing_files = []
    for file in required_files:
        if not (adapter_path / file).exists():
            missing_files.append(file)
    
    # Calculate total size
    total_size = 0
    for file in adapter_path.glob("*.safetensors"):
        total_size += file.stat().st_size
    for file in adapter_path.glob("*.json"):
        total_size += file.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    
    is_valid = len(missing_files) == 0
    return is_valid, missing_files, total_size_mb


def create_lora_model_card(adapter_path: Path, repo_name: str) -> str:
    """Create a model card for LoRA adapters.
    
    Args:
        adapter_path: Path to adapter directory
        repo_name: HuggingFace repository name
        
    Returns:
        Model card content as string
    """
    # Load adapter config
    with open(adapter_path / "adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    # Load config if exists
    config = {}
    config_path = adapter_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    base_model = config.get("base_model", adapter_config.get("base_model", "microsoft/Phi-3-mini-4k-instruct"))
    
    model_card = f"""---
license: mit
base_model: {base_model}
tags:
- lora
- mlx
- fine-tuned
library_name: mlx
---

# LoRA Adapters for {base_model.split('/')[-1]}

This repository contains LoRA adapter weights for fine-tuning {base_model} using MLX.

## Model Details

- **Base Model**: {base_model}
- **Training Framework**: MLX
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: {config.get('trainable_params', 'N/A'):,} ({config.get('trainable_percent', 'N/A')}% of total)
- **Total Model Parameters**: {config.get('total_params', 'N/A'):,}

## LoRA Configuration

- **Rank (r)**: {adapter_config.get('lora_parameters', {}).get('rank', 16)}
- **Scale**: {adapter_config.get('lora_parameters', {}).get('scale', 20.0)}
- **Dropout**: {adapter_config.get('lora_parameters', {}).get('dropout', 0.1)}
- **Target Modules**: {', '.join(adapter_config.get('lora_parameters', {}).get('keys', []))}
- **Number of Layers**: {adapter_config.get('lora_layers', 32)} (out of {adapter_config.get('num_layers', 32)} total)

## Usage

### Installation

```bash
pip install mlx-lm
```

### Loading the Adapters

#### Option 1: Load from HuggingFace Hub

```python
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
from huggingface_hub import snapshot_download
import json

# Download adapters from HuggingFace
adapter_path = snapshot_download(repo_id="{repo_name}")

# Load base model
model, tokenizer = load("{base_model}")

# Load adapter config
with open(f"{{adapter_path}}/adapter_config.json", "r") as f:
    adapter_config = json.load(f)

# Freeze base model and apply LoRA layers
model.freeze()
linear_to_lora_layers(
    model, 
    adapter_config["lora_layers"],
    adapter_config["lora_parameters"]
)

# Load the LoRA weights
model.load_weights(f"{{adapter_path}}/adapters.safetensors", strict=False)

# Generate text
prompt = "<|system|>\\nYou are a helpful assistant.<|end|>\\n<|user|>\\nHello!<|end|>\\n<|assistant|>"
response = generate(model, tokenizer, prompt, max_tokens=200)
print(response)
```

#### Option 2: Clone and Load Locally

```bash
git clone https://huggingface.co/{repo_name}
cd {repo_name.split('/')[-1]}
```

Then use the same Python code above, replacing `adapter_path` with your local directory path.

## Training Details

These adapters were trained using:
- **Framework**: MLX with LoRA fine-tuning
- **Hardware**: Apple Silicon
- **Training approach**: Parameter-efficient fine-tuning with gradient checkpointing

## Files

- `adapters.safetensors`: Final adapter weights
- `adapter_config.json`: LoRA configuration
- `config.json`: Training and model metadata
- `*.safetensors`: Training checkpoint files (optional)

## License

These adapters are released under the MIT License. The base model may have its own license requirements.
"""
    
    return model_card


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapters to HuggingFace Hub")
    parser.add_argument(
        "--adapter-path", 
        type=str, 
        default="models/adapters",
        help="Path to LoRA adapters directory (default: models/adapters)"
    )
    parser.add_argument(
        "--repo-name", 
        type=str, 
        required=True,
        help="HuggingFace repository name (e.g., 'username/model-name-lora')"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include training checkpoint files (default: only final adapters)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Validate setup without uploading"
    )
    parser.add_argument(
        "--token", 
        type=str,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("LORA ADAPTER UPLOAD PIPELINE")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Get HuggingFace token
    hf_token = args.token or os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("❌ HuggingFace token not found!")
        print("Either:")
        print("  1. Set HF_TOKEN environment variable")
        print("  2. Add HF_TOKEN to .env file")
        print("  3. Use --token argument")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Check if adapter directory exists
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"❌ Adapter directory not found: {args.adapter_path}")
        print("Run '02_train_model.py' first to train the model.")
        sys.exit(1)
    
    print(f"Adapter path: {adapter_path}")
    print(f"Repository: {args.repo_name}")
    print(f"Private: {args.private}")
    print(f"Include checkpoints: {args.include_checkpoints}")
    print(f"Dry run: {args.dry_run}")
    
    # Step 1: Validate adapter structure
    print("\nStep 1: Validating adapter structure...")
    
    is_valid, missing_files, total_size = validate_adapter_structure(adapter_path)
    
    if missing_files:
        print(f"⚠️  Warning: Missing files: {missing_files}")
        if "adapters.safetensors" in missing_files:
            print("❌ Critical: adapters.safetensors not found!")
            sys.exit(1)
    
    # Count files to upload
    files_to_upload = []
    
    # Essential files
    for file in ["adapter_config.json", "config.json", "adapters.safetensors"]:
        file_path = adapter_path / file
        if file_path.exists():
            files_to_upload.append(file_path)
    
    # Optional checkpoint files
    if args.include_checkpoints:
        checkpoint_files = list(adapter_path.glob("*_adapters.safetensors"))
        files_to_upload.extend(checkpoint_files)
        print(f"  Including {len(checkpoint_files)} checkpoint files")
    
    print(f"✅ Adapter validation passed")
    print(f"  Files to upload: {len(files_to_upload)}")
    print(f"  Total size: {total_size:.1f} MB")
    
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")
        print("Files that would be uploaded:")
        for file in files_to_upload:
            print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
        sys.exit(0)
    
    # Step 2: Initialize HuggingFace API
    print("\nStep 2: Initializing HuggingFace API...")
    
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
    
    # Step 3: Create or get repository
    print("\nStep 3: Creating repository...")
    
    try:
        repo_url = create_repo(
            repo_id=args.repo_name,
            private=args.private,
            token=hf_token,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository ready: {repo_url}")
    except Exception as e:
        print(f"❌ Repository creation failed: {e}")
        sys.exit(1)
    
    # Step 4: Create model card
    print("\nStep 4: Creating model card...")
    
    try:
        model_card_content = create_lora_model_card(adapter_path, args.repo_name)
        
        model_card_path = adapter_path / "README.md"
        with open(model_card_path, 'w') as f:
            f.write(model_card_content)
        
        files_to_upload.append(model_card_path)
        
        print(f"✅ Model card created: {model_card_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create model card: {e}")
    
    # Step 5: Upload files
    print("\nStep 5: Uploading adapter files...")
    
    try:
        print(f"Uploading {len(files_to_upload)} files...")
        
        # Upload all files
        for file_path in files_to_upload:
            relative_path = file_path.name
            print(f"  Uploading {relative_path}... ", end="")
            
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=relative_path,
                repo_id=args.repo_name,
                token=hf_token
            )
            print("✅")
        
        print(f"✅ Upload completed successfully!")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
    
    # Step 6: Verify upload
    print("\nStep 6: Verifying upload...")
    
    try:
        files = api.list_repo_files(repo_id=args.repo_name, token=hf_token)
        print(f"✅ Upload verified: {len(files)} files in repository")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify upload: {e}")
    
    # Success message
    print("\n" + "="*60)
    print("LORA ADAPTER UPLOAD COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"🎉 Adapters uploaded to: https://huggingface.co/{args.repo_name}")
    print("\nYour LoRA adapters are now available for:")
    print("  - Direct download and usage with MLX")
    print("  - Integration with base models")
    print("  - Sharing with the community")
    print("\nExample usage:")
    print(f"  from mlx_lm.tuner import load_adapters")
    print(f"  model = load_adapters(base_model, '{args.repo_name}')")
    print("="*60)


if __name__ == "__main__":
    main()