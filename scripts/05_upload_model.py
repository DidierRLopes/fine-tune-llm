#!/usr/bin/env python3
"""
Model upload script for fine-tuning pipeline.
Uploads fused model to HuggingFace Hub.
"""

import sys
import argparse
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Upload fused model to HuggingFace Hub")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/fused",
        help="Path to fused model directory"
    )
    parser.add_argument(
        "--repo-name", 
        type=str, 
        required=True,
        help="HuggingFace repository name (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make repository private"
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
    print("MODEL UPLOAD PIPELINE")
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
    
    # Check if model directory exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model directory not found: {args.model_path}")
        print("Run '04_fuse_adapters.py' first to create the fused model.")
        sys.exit(1)
    
    print(f"Model path: {model_path}")
    print(f"Repository: {args.repo_name}")
    print(f"Private: {args.private}")
    print(f"Dry run: {args.dry_run}")
    
    # Validate model structure
    print("\nStep 1: Validating model structure...")
    
    required_files = ["config.json"]
    model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.npz"))
    
    missing_files = []
    for req_file in required_files:
        if not (model_path / req_file).exists():
            missing_files.append(req_file)
    
    if missing_files:
        print(f"⚠️  Warning: Missing recommended files: {missing_files}")
    
    if not model_files:
        print("❌ No model weight files found (.safetensors or .npz)")
        sys.exit(1)
    
    print(f"✅ Model validation passed")
    print(f"  Found {len(model_files)} model weight files")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    print(f"  Total model size: {total_size_mb:.1f} MB")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be uploaded")
        print("✅ Model structure validation completed")
        print(f"Ready to upload {len(list(model_path.rglob('*')))} files to {args.repo_name}")
        return
    
    # Initialize HuggingFace API
    print("\nStep 2: Initializing HuggingFace API...")
    try:
        api = HfApi(token=hf_token)
        
        # Test token by getting user info
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your HuggingFace token.")
        sys.exit(1)
    
    # Create repository
    print("\nStep 3: Creating repository...")
    try:
        repo_info = create_repo(
            repo_id=args.repo_name,
            token=hf_token,
            private=args.private,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_info}")
        
    except Exception as e:
        print(f"❌ Repository creation failed: {e}")
        sys.exit(1)
    
    # Create model card
    print("\nStep 4: Creating model card...")
    try:
        model_card_content = f\"\"\"---
license: mit
base_model: microsoft/Phi-3-mini-4k-instruct
tags:
- fine-tuned
- lora
- mlx
---

# Fine-tuned Phi-3 Mini Model

This model is a fine-tuned version of microsoft/Phi-3-mini-4k-instruct using LoRA (Low-Rank Adaptation) and MLX.

## Model Details

- **Base Model**: microsoft/Phi-3-mini-4k-instruct
- **Fine-tuning Method**: LoRA with MLX
- **Model Size**: {total_size_mb:.1f} MB

## Usage

```python
from mlx_lm import load, generate

# Load the model
model, tokenizer = load("{args.repo_name}")

# Generate text
prompt = "<|system|>\\nYou are a helpful assistant.<|end|>\\n<|user|>\\nHello!<|end|>\\n<|assistant|>"
response = generate(model, tokenizer, prompt, max_tokens=100)
print(response)
```

## Training

This model was fine-tuned using the MLX framework with LoRA adapters. The training process involved:

1. Data preprocessing and validation
2. LoRA configuration and setup
3. Fine-tuning with custom training loop
4. Adapter fusion into base model

## License

This model is released under the MIT license.
\"\"\"

        model_card_path = model_path / "README.md"
        with open(model_card_path, 'w') as f:
            f.write(model_card_content)
        
        print(f"✅ Model card created: {model_card_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create model card: {e}")
    
    # Upload model
    print("\nStep 5: Uploading model files...")
    try:
        print(f"Uploading {len(list(model_path.rglob('*')))} files...")
        
        # Upload the entire directory
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_name,
            repo_type="model",
            token=hf_token
        )
        
        print(f"✅ Upload completed successfully!")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
    
    # Verify upload
    print("\nStep 6: Verifying upload...")
    try:
        repo_files = api.list_repo_files(args.repo_name, token=hf_token)
        print(f"✅ Upload verified: {len(repo_files)} files in repository")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not verify upload: {e}")
    
    print("\n" + "="*60)
    print("MODEL UPLOAD COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"🎉 Model uploaded to: https://huggingface.co/{args.repo_name}")
    print("\nYour model is now available for:")
    print("  - Direct download and usage")
    print("  - Integration with transformers library")
    print("  - Sharing with the community")
    print("\nExample usage:")
    print(f'  from mlx_lm import load')
    print(f'  model, tokenizer = load("{args.repo_name}")')
    print("="*60)


if __name__ == "__main__":
    main()