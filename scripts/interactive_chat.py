#!/usr/bin/env python3
"""
Interactive chat script for testing fine-tuned models.
Provides a command-line chat interface.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.chat_interface import ChatInterface


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with fine-tuned model")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/fused",
        help="Path to model directory"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str,
        help="Custom system prompt (uses default if not provided)"
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
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        print("Available options:")
        print("  - Run '04_fuse_adapters.py' to create fused model")
        print("  - Use --model-path to specify different model")
        sys.exit(1)
    
    print("="*60)
    print("INTERACTIVE CHAT INTERFACE")
    print("="*60)
    
    try:
        # Initialize chat interface
        chat = ChatInterface(args.model_path, args.system_prompt)
        
        if args.quick_test:
            # Run quick test mode
            test_questions = [
                "What is OpenBB?",
                "Why is open source important for fintech?",
                "How can AI improve financial research?",
                "What makes a good investment terminal?",
            ]
            
            results = chat.quick_test(
                test_questions, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature
            )
            
            # Optionally save results
            save_choice = input("\\nSave test results? (y/n): ").lower().strip()
            if save_choice in ['y', 'yes']:
                chat.save_conversation()
        
        else:
            # Start interactive chat
            chat.start_chat(
                max_tokens=args.max_tokens, 
                temperature=args.temperature
            )
    
    except KeyboardInterrupt:
        print("\n\\nChat interrupted by user. Goodbye!")
    
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()