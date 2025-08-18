#!/usr/bin/env python3
"""
Interactive chat script for testing fine-tuned models.
Provides comparison mode to test base, LoRA runtime, and fused models simultaneously.
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.chat_interface import ChatInterface


class ModelComparison:
    """Compare responses from different model configurations."""
    
    def __init__(self, base_model_path: str = "microsoft/Phi-3-mini-4k-instruct",
                 adapters_path: str = "models/adapters", 
                 fused_path: str = "models/fused"):
        """Initialize model comparison.
        
        Args:
            base_model_path: Path or HuggingFace ID for base model
            adapters_path: Path to LoRA adapters directory
            fused_path: Path to fused model directory
        """
        self.models = {}
        self.tokenizer = None
        
        # Load base model
        print("\n🔸 Loading base model...")
        try:
            model, tokenizer = load(base_model_path)
            self.models['base'] = model
            self.tokenizer = tokenizer
            print("  ✅ Base model loaded")
        except Exception as e:
            print(f"  ❌ Failed to load base model: {e}")
        
        # Load LoRA runtime if adapters exist
        if Path(adapters_path).exists():
            print("\n🔸 Loading LoRA runtime (base + adapters)...")
            try:
                # Load fresh base model for runtime adapters
                model, _ = load(base_model_path)
                
                # Load adapter config
                with open(Path(adapters_path) / "adapter_config.json", 'r') as f:
                    adapter_config = json.load(f)
                
                # Freeze and apply LoRA
                model.freeze()
                linear_to_lora_layers(
                    model,
                    adapter_config["lora_layers"],
                    adapter_config["lora_parameters"]
                )
                
                # Load adapter weights
                adapter_file = Path(adapters_path) / "adapters.safetensors"
                model.load_weights(str(adapter_file), strict=False)
                
                self.models['lora_runtime'] = model
                print("  ✅ LoRA runtime loaded")
            except Exception as e:
                print(f"  ❌ Failed to load LoRA runtime: {e}")
        else:
            print(f"\n⚠️  LoRA adapters not found at: {adapters_path}")
        
        # Load fused model if exists
        if Path(fused_path).exists():
            print("\n🔸 Loading fused model...")
            try:
                model, _ = load(fused_path)
                self.models['fused'] = model
                print("  ✅ Fused model loaded")
            except Exception as e:
                print(f"  ❌ Failed to load fused model: {e}")
        else:
            print(f"\n⚠️  Fused model not found at: {fused_path}")
    
    def compare_responses(self, prompt: str, max_tokens: int = 200, 
                         temperature: float = 0.7) -> Dict[str, str]:
        """Generate responses from all available models.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary of model_name -> response
        """
        if not self.tokenizer:
            print("❌ No models loaded successfully")
            return {}
        
        responses = {}
        
        for name, model in self.models.items():
            print(f"\n🤖 Generating from {name}...")
            try:
                response = generate(
                    model, self.tokenizer, prompt,
                    max_tokens=max_tokens,
                    temp=temperature
                )
                # Extract only the assistant response
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1]
                if "<|end|>" in response:
                    response = response.split("<|end|>")[0]
                responses[name] = response.strip()
            except Exception as e:
                print(f"  ❌ Error: {e}")
                responses[name] = f"[Error: {e}]"
        
        return responses
    
    def print_comparison(self, prompt: str, responses: Dict[str, str]):
        """Print formatted comparison of responses.
        
        Args:
            prompt: Original prompt
            responses: Dictionary of model responses
        """
        print("\n" + "="*80)
        print("MODEL RESPONSE COMPARISON")
        print("="*80)
        
        # Extract user message for cleaner display
        if "<|user|>" in prompt:
            user_msg = prompt.split("<|user|>")[1].split("<|end|>")[0].strip()
        else:
            user_msg = prompt
        
        print(f"\n📝 Prompt: {user_msg}")
        
        for name, response in responses.items():
            print(f"\n{'='*40}")
            model_label = {
                'base': '🔷 BASE MODEL',
                'lora_runtime': '🔶 LORA RUNTIME (Base + Adapters)',
                'fused': '🔴 FUSED MODEL'
            }.get(name, name.upper())
            
            print(f"{model_label}")
            print("-"*40)
            print(response)
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with model comparison")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "compare"],
        default="compare",
        help="Mode: single model or compare all models"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/fused",
        help="Path to model directory (for single mode)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model path or HuggingFace ID"
    )
    parser.add_argument(
        "--adapters-path",
        type=str,
        default="models/adapters",
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--fused-path",
        type=str,
        default="models/fused",
        help="Path to fused model"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str,
        help="Custom system prompt"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run quick test with predefined questions"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("INTERACTIVE CHAT INTERFACE")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    
    try:
        if args.mode == "compare":
            # Comparison mode - load all available models
            comparator = ModelComparison(
                base_model_path=args.base_model,
                adapters_path=args.adapters_path,
                fused_path=args.fused_path
            )
            
            if not comparator.models:
                print("❌ No models loaded successfully")
                sys.exit(1)
            
            print(f"\n✅ Loaded {len(comparator.models)} models: {list(comparator.models.keys())}")
            
            # Set up system prompt
            if args.system_prompt:
                system_prompt = args.system_prompt
            else:
                system_prompt = "You are a helpful AI assistant."
            
            if args.quick_test:
                # Quick test mode with predefined questions
                test_questions = [
                    "What is machine learning?",
                    "Explain neural networks in simple terms.",
                    "What are the benefits of open source?",
                    "How does fine-tuning work?",
                ]
                
                print("\n" + "="*60)
                print("QUICK TEST MODE")
                print("="*60)
                
                for question in test_questions:
                    # Format prompt
                    prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>"
                    
                    # Generate responses from all models
                    responses = comparator.compare_responses(
                        prompt, 
                        max_tokens=args.max_tokens,
                        temperature=args.temperature
                    )
                    
                    # Print comparison
                    comparator.print_comparison(prompt, responses)
                    
                    # Optional: pause between questions
                    if question != test_questions[-1]:
                        input("\nPress Enter for next question...")
            
            else:
                # Interactive mode
                print("\n" + "="*60)
                print("INTERACTIVE COMPARISON MODE")
                print("="*60)
                print("Type your prompts to compare model responses.")
                print("Commands: 'quit' to exit, 'clear' to clear screen")
                print("-"*60)
                
                while True:
                    try:
                        # Get user input
                        user_input = input("\n👤 You: ").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print("\nGoodbye! 👋")
                            break
                        
                        if user_input.lower() == 'clear':
                            import os
                            os.system('clear' if os.name == 'posix' else 'cls')
                            continue
                        
                        if not user_input:
                            continue
                        
                        # Format prompt
                        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>"
                        
                        # Generate responses
                        responses = comparator.compare_responses(
                            prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature
                        )
                        
                        # Print comparison
                        comparator.print_comparison(prompt, responses)
                    
                    except KeyboardInterrupt:
                        print("\n\nChat interrupted. Goodbye! 👋")
                        break
        
        else:
            # Single model mode (original behavior)
            if not Path(args.model_path).exists():
                print(f"❌ Model not found: {args.model_path}")
                sys.exit(1)
            
            chat = ChatInterface(args.model_path, args.system_prompt)
            
            if args.quick_test:
                test_questions = [
                    "What is machine learning?",
                    "Explain neural networks in simple terms.",
                    "What are the benefits of open source?",
                    "How does fine-tuning work?",
                ]
                
                results = chat.quick_test(
                    test_questions, 
                    max_tokens=args.max_tokens, 
                    temperature=args.temperature
                )
                
                save_choice = input("\\nSave test results? (y/n): ").lower().strip()
                if save_choice in ['y', 'yes']:
                    chat.save_conversation()
            else:
                chat.start_chat(
                    max_tokens=args.max_tokens, 
                    temperature=args.temperature
                )
    
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()