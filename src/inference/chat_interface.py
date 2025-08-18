"""Interactive chat interface for testing fine-tuned models."""

import os
from typing import List, Dict, Any, Optional
from .generator import TextGenerator


class ChatInterface:
    """Interactive chat interface for model testing."""
    
    def __init__(self, model_path: str, system_prompt: str = None):
        """Initialize chat interface.
        
        Args:
            model_path: Path to model directory
            system_prompt: System prompt for the conversation
        """
        self.generator = TextGenerator(model_path, system_prompt)
        self.conversation_history = []
        self.session_active = False
        
        print("="*60)
        print("CHAT INTERFACE INITIALIZED")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"System prompt: {self.generator.default_system_prompt[:100]}...")
        print("\nType 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("Type 'save' to save conversation history")
        print("Type 'help' for more commands")
        print("="*60)
    
    def display_help(self):
        """Display help information."""
        print("\n" + "="*40)
        print("CHAT INTERFACE COMMANDS")
        print("="*40)
        print("quit, exit, q    - End the conversation")
        print("clear            - Clear conversation history")
        print("save             - Save conversation to file")
        print("history          - Show conversation history")
        print("info             - Show model information")
        print("temp <value>     - Set temperature (0.1-2.0)")
        print("tokens <value>   - Set max tokens (50-500)")
        print("help             - Show this help message")
        print("="*40 + "\n")
    
    def display_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("No conversation history.")
            return
        
        print("\n" + "="*40)
        print("CONVERSATION HISTORY")
        print("="*40)
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. User: {entry['user']}")
            print(f"   Assistant: {entry['assistant']}")
        
        print("="*40 + "\n")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        print("Conversation history cleared.")
    
    def save_conversation(self, filename: str = None):
        """Save conversation history to file.
        
        Args:
            filename: Optional filename (auto-generated if None)
        """
        import json
        from datetime import datetime
        from pathlib import Path
        
        if not self.conversation_history:
            print("No conversation to save.")
            return
        
        # Ensure output directory exists
        output_dir = Path("data/sample_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_session_{timestamp}.json"
        
        filepath = output_dir / filename
        
        # Prepare conversation data
        conversation_data = {
            "model_path": self.generator.model_path,
            "system_prompt": self.generator.default_system_prompt,
            "timestamp": datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"Conversation saved to: {filepath}")
    
    def process_command(self, user_input: str) -> bool:
        """Process special commands.
        
        Args:
            user_input: User input string
            
        Returns:
            True if command was processed, False otherwise
        """
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'q']:
            return True
        elif command == 'clear':
            self.clear_history()
        elif command == 'save':
            self.save_conversation()
        elif command == 'history':
            self.display_history()
        elif command == 'help':
            self.display_help()
        elif command == 'info':
            self.display_model_info()
        elif command.startswith('temp '):
            self.set_temperature(command)
        elif command.startswith('tokens '):
            self.set_max_tokens(command)
        else:
            return False
        
        return True
    
    def set_temperature(self, command: str):
        """Set generation temperature."""
        try:
            temp_str = command.split(' ', 1)[1]
            temperature = float(temp_str)
            if 0.1 <= temperature <= 2.0:
                self.temperature = temperature
                print(f"Temperature set to: {temperature}")
            else:
                print("Temperature must be between 0.1 and 2.0")
        except (ValueError, IndexError):
            print("Invalid temperature value. Use: temp <value>")
    
    def set_max_tokens(self, command: str):
        """Set maximum tokens."""
        try:
            tokens_str = command.split(' ', 1)[1]
            max_tokens = int(tokens_str)
            if 50 <= max_tokens <= 500:
                self.max_tokens = max_tokens
                print(f"Max tokens set to: {max_tokens}")
            else:
                print("Max tokens must be between 50 and 500")
        except (ValueError, IndexError):
            print("Invalid max tokens value. Use: tokens <value>")
    
    def display_model_info(self):
        """Display model information."""
        info = self.generator.get_model_info()
        print("\n" + "="*40)
        print("MODEL INFORMATION")
        print("="*40)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("="*40 + "\n")
    
    def start_chat(self, max_tokens: int = 200, temperature: float = 0.7):
        """Start interactive chat session.
        
        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.session_active = True
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"\nChat session started (max_tokens={max_tokens}, temperature={temperature})")
        print("You can start typing your questions now:\n")
        
        try:
            while self.session_active:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                if self.process_command(user_input):
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    continue
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                
                try:
                    # Convert conversation history to format expected by generator
                    context = []
                    for entry in self.conversation_history:
                        context.append({"role": "user", "content": entry["user"]})
                        context.append({"role": "assistant", "content": entry["assistant"]})
                    
                    response = self.generator.generate_with_context(
                        context, user_input, 
                        max_tokens=self.max_tokens, 
                        temperature=self.temperature
                    )
                    
                    print(response)
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        "user": user_input,
                        "assistant": response
                    })
                    
                except Exception as e:
                    print(f"Error: {e}")
                
                print()  # Add blank line for readability
        
        except KeyboardInterrupt:
            print("\n\nChat session interrupted by user.")
        
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        
        finally:
            self.end_chat()
    
    def end_chat(self):
        """End chat session."""
        self.session_active = False
        
        print("\n" + "="*60)
        print("CHAT SESSION ENDED")
        print("="*60)
        
        if self.conversation_history:
            print(f"Total exchanges: {len(self.conversation_history)}")
            
            # Ask if user wants to save
            try:
                save_choice = input("Save conversation? (y/n): ").lower().strip()
                if save_choice in ['y', 'yes']:
                    self.save_conversation()
            except:
                pass  # Handle any input errors gracefully
        
        print("Thank you for using the chat interface!")
    
    def quick_test(self, test_questions: List[str], max_tokens: int = 150, 
                   temperature: float = 0.7) -> Dict[str, str]:
        """Quick test with predefined questions.
        
        Args:
            test_questions: List of questions to test
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary mapping questions to responses
        """
        print("="*60)
        print("QUICK TEST MODE")
        print("="*60)
        
        results = {}
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 40)
            
            response = self.generator.generate_response(
                question, max_tokens=max_tokens, temperature=temperature
            )
            
            print(f"Response: {response}")
            results[question] = response
        
        print("\n" + "="*60)
        print("QUICK TEST COMPLETED")
        print("="*60)
        
        return results