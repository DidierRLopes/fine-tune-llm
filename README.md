# Fine-Tuning Phi-3 Mini on Blog Q&A Data using MLX LM

This notebook demonstrates how to fine-tune Microsoft's Phi-3 Mini model (3.8B parameters) to better answer questions about blog content using Low-Rank Adaptation (LoRA).

The process shows how to adapt a large language model for a specific task while maintaining efficiency on Apple Silicon hardware.

## What's Actually Happening Here

### 1. Dataset Preparation

- Uses a blog Q&A dataset containing conversations about blog posts
- Splits the data into train/validation/test sets (70/20/10 split)
- Formats conversations into question-answer pairs for training

### 2. Model Adaptation

- Uses LoRA to efficiently fine-tune the model by only modifying a small number of parameters
- Adapts only the attention layers (query, key, value matrices) in the top 8 transformer blocks
- Uses a low rank of 8 to keep the number of trainable parameters small
- Implements dropout (0.2) to prevent overfitting

### 3. Training Process

- Runs for 500 iterations with evaluation every 100 steps
- Uses Adam optimizer with a conservative learning rate (5e-6)
- Tracks both training and validation losses to monitor progress
- Saves adapter weights periodically

### 4. Evaluation Approach

- Uses LLaMA 3.2 as an independent judge to evaluate responses
- Compares semantic similarity between model outputs and ground truth
- Provides detailed metrics including:
  - Mean similarity scores
  - Distribution of good (≥0.7) vs poor (≤0.3) responses
  - Performance comparison with the base model

### 5. Model Export

- Fuses LoRA adapters into the base model for deployment
- Uploads the resulting model to Hugging Face for sharing

## Key Results

The notebook shows:

- How to adapt a large model with minimal computational resources
- The effectiveness of LoRA for task-specific fine-tuning
- A practical approach to evaluating model improvements
- Real-world deployment considerations

This is particularly useful for developers looking to customize large language models for specific domains or tasks while working within the constraints of local hardware.

## Technical Significance

The approach demonstrates:

- Efficient parameter-efficient fine-tuning (only modifying a small subset of model parameters)
- Practical evaluation methods for comparing model versions
- Integration with the MLX framework for Apple Silicon optimization
- End-to-end workflow from training to deployment

This serves as a practical template for similar fine-tuning projects, especially when working with domain-specific data or custom response patterns.
