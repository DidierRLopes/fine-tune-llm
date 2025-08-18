# Fine-Tune LLM Pipeline

A modular, production-ready pipeline for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation) with MLX on Apple Silicon.

## 🎯 Overview

This project provides a complete end-to-end pipeline for fine-tuning LLMs, from data preparation to model deployment. It's specifically optimized for Apple Silicon devices using the MLX framework and includes comprehensive evaluation and monitoring tools.

## ✨ Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, training, evaluation, and inference
- **Configuration-Driven**: YAML-based configuration for easy experimentation and reproducibility
- **LoRA Fine-Tuning**: Parameter-efficient fine-tuning using Low-Rank Adaptation
- **Three-Way Data Split**: Proper train/validation/test split for robust evaluation
- **Runtime & Fused Adapters**: Compare runtime adapters vs fused models for deployment
- **Comprehensive Evaluation**: Word overlap and BERTScore metrics with fusion quality verification
- **Interactive Chat Interface**: Test your models with a command-line chat interface
- **Model Comparison**: Compare performance between base, runtime adapters, and fused models
- **HuggingFace Integration**: Seamless upload to HuggingFace Hub

## 🏗️ Project Structure

```
fine-tune-llm/
├── config/                   # Configuration files
│   ├── data_config.yaml     # Dataset and formatting settings
│   ├── model_config.yaml    # Model and LoRA configuration
│   ├── training_config.yaml # Training parameters
│   └── evaluation_config.yaml # Evaluation settings
├── src/                     # Source code modules
│   ├── data/               # Data loading and preprocessing
│   ├── training/           # Training and LoRA setup
│   ├── evaluation/         # Model evaluation and comparison
│   ├── inference/          # Text generation and chat interface
│   └── utils/              # Utilities for models, plotting, fusion
├── scripts/                # Main pipeline scripts
│   ├── 01_prepare_data.py  # Data preparation and three-way split
│   ├── 02_train_model.py   # LoRA fine-tuning with validation
│   ├── 03_evaluate_model.py # Evaluate base + runtime adapters
│   ├── 04_fuse_and_evaluate.py # Fusion + comprehensive evaluation
│   ├── 05_upload_model.py  # HuggingFace upload
│   └── interactive_chat.py # Chat interface
├── data/                   # Data storage
├── models/                 # Model storage
├── logs/                   # Training and evaluation logs
└── notebooks/              # Jupyter notebooks for exploration
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
CONDA_SUBDIR=osx-arm64 conda create -n fine-tune-llm python=3.11
conda activate fine-tune-llm
conda config --env --set subdir osx-arm64

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit the configuration files in `config/` to match your dataset and training preferences:

- `data_config.yaml`: Dataset name, formatting, and validation settings
- `model_config.yaml`: Base model path and LoRA configuration
- `training_config.yaml`: Training parameters and optimization settings
- `evaluation_config.yaml`: Evaluation methods and metrics

### 3. Run the Pipeline

```bash
# Step 1: Prepare data (creates train/val/test split)
python scripts/01_prepare_data.py

# Step 2: Train model with LoRA adapters
python scripts/02_train_model.py

# Step 3: Evaluate base model + runtime adapters
python scripts/03_evaluate_model.py

# Step 4: Fuse adapters and comprehensive evaluation (optional)
python scripts/04_fuse_and_evaluate.py

# Step 5: Upload to HuggingFace (optional)
python scripts/05_upload_model.py --repo-name "your-username/your-model"
```

### 4. Test Your Model

```bash
# Interactive chat
python scripts/interactive_chat.py

# Quick test
python scripts/interactive_chat.py --quick-test
```

## 📊 Configuration Examples

### LoRA Configuration (model_config.yaml)

```yaml
lora:
  num_layers: 32
  lora_layers: 32  # Use all layers for full adaptation
  rank: 16         # Higher rank for more expressive power
  scale: 20.0      # LoRA scaling factor
  dropout: 0.1     # Dropout for regularization
```

### Training Configuration (training_config.yaml)

```yaml
training:
  iters: 2000         # Training iterations
  batch_size: 4       # Batch size (adjust for your hardware)
  learning_rate: 1e-5 # Learning rate
  steps_per_eval: 50  # Evaluation frequency
  grad_checkpoint: true # Memory optimization
```

## 📋 Script Reference

### 01_prepare_data.py
**Purpose**: Load dataset and create three-way split (train/validation/test)

**What it does**:
- Downloads and loads specified dataset from HuggingFace
- Applies conversation formatting with system prompts
- Creates train/validation/test split (default: 80%/10%/10%)
- Validates data format and saves processed files
- Generates data statistics and sample previews

**Parameters**:
```bash
python scripts/01_prepare_data.py \
  --data-config config/data_config.yaml \
  --model-config config/model_config.yaml \
  --output-dir data/processed
```

- `--data-config`: Dataset configuration (source, formatting, splits)
- `--model-config`: Model configuration (needed for tokenizer)
- `--output-dir`: Where to save processed data files

### 02_train_model.py  
**Purpose**: Fine-tune model using LoRA adapters with validation

**What it does**:
- Loads base model and applies LoRA configuration
- Sets up training with gradient checkpointing and early stopping
- Trains on training set, validates on validation set
- Saves adapter weights and configuration
- Logs training metrics and validation perplexity

**Parameters**:
```bash
python scripts/02_train_model.py \
  --model-config config/model_config.yaml \
  --training-config config/training_config.yaml \
  --train-data data/processed/train.json \
  --val-data data/processed/val.json \
  --output-dir models/adapters
```

- `--model-config`: Model and LoRA configuration
- `--training-config`: Training parameters (learning rate, batch size, etc.)
- `--train-data`: Training dataset JSON file
- `--val-data`: Validation dataset JSON file  
- `--output-dir`: Where to save trained adapters

### 03_evaluate_model.py
**Purpose**: Evaluate base model and runtime LoRA adapters

**What it does**:
- Evaluates base model performance on test set
- Loads base model + runtime adapters and evaluates
- Compares base vs fine-tuned performance
- Generates evaluation metrics and saves results
- Creates comparison plots and analysis

**Parameters**:
```bash
python scripts/03_evaluate_model.py \
  --config config/evaluation_config.yaml \
  --test-data data/processed/test.json \
  --adapters-path models/adapters \
  --base-model microsoft/Phi-3-mini-4k-instruct
```

- `--config`: Evaluation configuration (metrics, temperature, etc.)
- `--test-data`: Test dataset JSON file
- `--adapters-path`: Directory containing trained adapters
- `--base-model`: Base model path or HuggingFace model ID
- `--model-path`: Evaluate specific model (optional)
- `--compare-only`: Only compare existing evaluation results

### 04_fuse_and_evaluate.py
**Purpose**: Fuse adapters into base model and comprehensive evaluation

**What it does**:
- Fuses LoRA adapters into base model weights (for deployment)
- Evaluates all three models: base, runtime adapters, fused
- Verifies fusion quality (mathematical equivalence check)
- Provides comprehensive comparison and analysis
- Saves fused model for deployment

**Parameters**:
```bash
python scripts/04_fuse_and_evaluate.py \
  --model-config config/model_config.yaml \
  --eval-config config/evaluation_config.yaml \
  --test-data data/processed/test.json \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --adapters-path models/adapters \
  --output-path lora_fused_model
```

- `--model-config`: Model configuration file
- `--eval-config`: Evaluation configuration file
- `--test-data`: Test dataset JSON file
- `--base-model`: Base model path override
- `--adapters-path`: Adapters directory override
- `--output-path`: Where to save fused model
- `--skip-fusion`: Skip fusion step (use existing fused model)
- `--force-fusion`: Force fusion even if output exists

### 05_upload_model.py
**Purpose**: Upload trained model to HuggingFace Hub

**What it does**:
- Uploads fused model or adapters to HuggingFace
- Creates model card with training details
- Sets up repository with proper licensing
- Includes usage examples and configuration

**Parameters**:
```bash
python scripts/05_upload_model.py \
  --repo-name "your-username/your-model" \
  --model-path lora_fused_model \
  --model-config config/model_config.yaml
```

## 🔧 Advanced Usage

### Custom Dataset

1. Modify `data_config.yaml` to point to your dataset
2. Ensure your data follows the expected conversation format
3. Adjust the system prompt for your use case

### Hyperparameter Tuning

The pipeline supports easy experimentation through configuration files:

```bash
# Train with different configurations
python scripts/02_train_model.py --training-config config/training_config_experimental.yaml
```

### Multi-Model Evaluation

Compare multiple models by running evaluation on different model paths:

```bash
# Evaluate specific model
python scripts/03_evaluate_model.py --model-path models/your-model

# Compare existing results
python scripts/03_evaluate_model.py --compare-only
```

## 📈 Evaluation Metrics

The pipeline includes comprehensive evaluation tools:

- **Word Overlap**: Jaccard similarity between predicted and reference text words
  - Primary metric for model comparison
  - Mean, median, standard deviation, and range statistics
- **BERTScore**: Semantic similarity using contextual embeddings (optional)
  - Precision, recall, and F1 scores using DeBERTa
  - More computationally expensive but semantically aware
- **Length Statistics**: Response length analysis (character counts)
- **Fusion Quality**: Verification that fused models preserve adapter behavior
- **Comparative Analysis**: Side-by-side model comparison with improvement percentages

**Key Insight**: Word overlap scores typically range 10-40% for good models due to multiple valid answer phrasings. Focus on relative improvements between models rather than absolute scores.

## 🎨 Understanding the Pipeline

### Runtime vs Fused Adapters

The pipeline supports two adapter deployment modes:

**Runtime Adapters** (`03_evaluate_model.py`):
- LoRA weights loaded at inference time  
- Base model stays frozen, adapters applied dynamically
- Slower inference but flexible (can swap adapters)
- Perfect for experimentation and A/B testing

**Fused Adapters** (`04_fuse_and_evaluate.py`):
- LoRA weights mathematically merged into base model
- Single model file for deployment
- Faster inference, smaller memory footprint  
- Perfect for production deployment

**Mathematical Equivalence**: Both approaches should produce identical outputs. The fusion process verification ensures this equivalence.

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or enable gradient checkpointing
2. **Data Format Errors**: Check data validation output and fix formatting
3. **Model Loading Errors**: Ensure model paths are correct and models exist
4. **Token Limit Issues**: Adjust max_tokens in generation settings

### Performance Optimization

- Use gradient checkpointing for memory efficiency
- Adjust LoRA rank vs. number of layers based on your hardware
- Monitor training metrics to detect overfitting/underfitting

## 📝 Model Card Template

When uploading to HuggingFace, the pipeline automatically creates a model card with:

- Model details and configuration
- Usage examples
- Training information
- License information

## 🤝 Contributing

This pipeline is designed to be modular and extensible. Key areas for contribution:

- Additional evaluation metrics
- Support for other model architectures
- Enhanced visualization tools
- Performance optimizations

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) - Language model utilities
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation technique
- [HuggingFace](https://huggingface.co/) - Model hub and tools

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review configuration files
3. Open an issue with detailed error messages and system information

---

**Happy Fine-Tuning! 🚀**
