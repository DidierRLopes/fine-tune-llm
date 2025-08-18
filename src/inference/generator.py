"""Text generation utilities for inference."""

from mlx_lm import load, generate
from typing import List, Dict, Any, Optional


class TextGenerator:
    """Handle text generation with fine-tuned models."""
    
    def __init__(self, model_path: str, system_prompt: str = None):
        """Initialize text generator.
        
        Args:
            model_path: Path to model directory
            system_prompt: Default system prompt
        """
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)
        
        self.default_system_prompt = (
            system_prompt or 
            "You are Didier, CEO of OpenBB. You write with clarity and impact, "
            "focusing on fintech, open source, AI, and the future of research workflows."
        )
        
        print(f"Text generator initialized with model: {model_path}")
    
    def format_prompt(self, user_message: str, system_prompt: str = None) -> str:
        """Format user message into Phi-3 chat format.
        
        Args:
            user_message: User's question or message
            system_prompt: System prompt (uses default if None)
            
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        
        formatted_prompt = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_message}<|end|>\n"
            f"<|assistant|>"
        )
        
        return formatted_prompt
    
    def generate_response(self, user_message: str, 
                         system_prompt: str = None,
                         max_tokens: int = 200, 
                         temperature: float = 0.7,
                         clean_response: bool = True) -> str:
        """Generate response to user message.
        
        Args:
            user_message: User's question or message
            system_prompt: System prompt (uses default if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            clean_response: Whether to clean the response
            
        Returns:
            Generated response text
        """
        # Format the prompt
        formatted_prompt = self.format_prompt(user_message, system_prompt)
        
        try:
            # Generate response
            response = generate(
                self.model, 
                self.tokenizer, 
                formatted_prompt, 
                max_tokens=max_tokens, 
                temp=temperature
            )
            
            if clean_response:
                response = self.clean_response(response)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {str(e)}"
    
    def clean_response(self, response: str) -> str:
        """Clean generated response by removing format tokens.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response text
        """
        # Extract only the assistant part
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        
        # Remove end tokens
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]
        
        return response.strip()
    
    def batch_generate(self, user_messages: List[str],
                      system_prompt: str = None,
                      max_tokens: int = 200,
                      temperature: float = 0.7) -> List[str]:
        """Generate responses to multiple messages.
        
        Args:
            user_messages: List of user messages
            system_prompt: System prompt (uses default if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for message in user_messages:
            response = self.generate_response(
                message, system_prompt, max_tokens, temperature
            )
            responses.append(response)
        
        return responses
    
    def generate_with_context(self, conversation_history: List[Dict[str, str]],
                            new_message: str,
                            system_prompt: str = None,
                            max_tokens: int = 200,
                            temperature: float = 0.7) -> str:
        """Generate response with conversation context.
        
        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            new_message: New user message
            system_prompt: System prompt (uses default if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        
        # Build conversation prompt
        conversation_prompt = f"<|system|>\n{system_prompt}<|end|>\n"
        
        # Add conversation history
        for msg in conversation_history:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                conversation_prompt += f"<|user|>\n{content}<|end|>\n"
            elif role == 'assistant':
                conversation_prompt += f"<|assistant|>\n{content}<|end|>\n"
        
        # Add new message
        conversation_prompt += f"<|user|>\n{new_message}<|end|>\n<|assistant|>"
        
        try:
            # Generate response
            response = generate(
                self.model,
                self.tokenizer,
                conversation_prompt,
                max_tokens=max_tokens,
                temp=temperature
            )
            
            return self.clean_response(response)
            
        except Exception as e:
            print(f"Error generating contextual response: {e}")
            return f"Error: Unable to generate response - {str(e)}"
    
    def save_generation_sample(self, prompts: List[str], responses: List[str], 
                              save_path: str = "data/sample_outputs/generation_samples.json"):
        """Save generation samples for analysis.
        
        Args:
            prompts: List of input prompts
            responses: List of generated responses
            save_path: Path to save samples
        """
        import json
        from pathlib import Path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for prompt, response in zip(prompts, responses):
            samples.append({
                "prompt": prompt,
                "response": response,
                "model_path": self.model_path
            })
        
        with open(save_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"Generation samples saved to: {save_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "default_system_prompt": self.default_system_prompt,
            "tokenizer_vocab_size": len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else "Unknown"
        }