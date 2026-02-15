"""
Google Gemini LLM Client Implementation
Primary LLM provider for Ask Your PDF
"""

import time
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from .base_llm import BaseLLM, LLMResponse, EmbeddingResponse

class GeminiClient(BaseLLM):
    """
    Google Gemini client implementation

    Features:
    - Text generation with gemini
    - Embeddings with model/embedding-001
    - Automatic retry on rate limits
    - Token counting
    """

    # Supported models
    SUPPORTED_MODELS = {
        "gemini-2.5-flash": {"max_input": 30720, "max_output": 2048},
        "gemini-3-flash-preview": {"max_input": 1048576, "max_output": 8192},
    }

    # Embedding model 
    EMBEDDING_MODEL = "models/embedding-001"
    EMBEDDING_DIMENSION = 768

    def _validate_config(self) -> None:
        """Validate Gemini-specific configuration"""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Gemini model: {self.model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {self.temperature}")
        
        # Check max_tokens against model limits
        model_limits = self.SUPPORTED_MODELS[self.model_name]
        max_output_limit = model_limits['max_output']
        if self.max_tokens > max_output_limit:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) exceeds model limit "
                f"({max_output_limit}) for {self.model_name}"
            )

    def _initialize_client(self) -> None:
        """Initialize Gemini SDK"""
        try:
            # Configure API key 
            genai.configure(api_key=self.api_key)

            # Create model instance for generation 
            self.client = genai.GenerativeModel(self.model_name)

            # generation config
            self.generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": self.extra_params.get("top_p", 0.95),
                "top_k": self.extra_params.get("top_k", 40)
            }

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")
        
    def generate(
            self, 
            prompt: str, 
            system_prompt: Optional[str] = None, 
            temperature: Optional[float] = None, 
            max_tokens: Optional[int] = None, 
            **kwargs
        ) -> LLMResponse:
        """
        Generate text using Gemini

        Args:
            prompt: User prompt
            system_prompt: System instruction (prepended to prompt)
            temperature: Overide max tokens
            **kwargs: Additional Gemini parameters

        Returns:
            LLMResponse with generated text
        """
        # Build final prompt 
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Overide generation config if needed
        config = self.generation_config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        # Generate with retry logic 
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 2)

        for attemp in range(max_retries):
            try:
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=config
                )

                # Extract text
                if not response.parts:
                    raise ValueError("Gemini returned empty response")
                
                text = response.text

                # Build standardize response 
                return LLMResponse(
                    text=text,
                    model=self.model_name,
                    token_used=self._estimate_tokens(full_prompt, text),
                    finish_reason=self._get_finish_reason(response),
                    metadata={
                        "prompt_feedback": response.prompt_feedback,
                        "safety_ratings": [
                            {
                                "category": rating.category.name,
                                "probability": rating.probability.name
                            }
                            for rating in response.candidates[0].safety_ratings
                        ] if response.candidates else []
                    }
                )
            
            except google_exceptions.ResourceExhausted as e:
                # Rate limit - retry with exponential backoff
                if attemp < max_retries - 1:
                    wait_time = retry_delay * (2 ** attemp)
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Gemini rate limit exceeded: {e}")
                
            except Exception as e:
                raise RuntimeError(f"Gemini generation failed: {e}")
    
    def get_embedding(
            self, 
            text: str, 
            **kwargs
        ) -> EmbeddingResponse:
        """
        Generate embedding using Gemnini embedding model

        Args:
            text: Text to embed

        Returns:
            EmbeddingResponse with 768-dimensional vector
        """
        try:
            # Use Gemini's embedding model 
            result = genai.embed_content(
                model=self.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )

            embedding = result["embedding"]

            return EmbeddingResponse(
                embedding=embedding,
                model=self.EMBEDDING_MODEL,
                dimension=self.EMBEDDING_DIMENSION,
                metadata={
                    "task_type": "retrieval_document",
                    "text_length": len(text)
                }
            )
        
        except Exception as e:
            raise RuntimeError(f"Gemini embedding generation failed: {e}")
        
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's tokenizer

        Args:
            text: Text to count

        Returns:
            Token count
        """
        try:
            # Gemini provides a count_tokens method
            result = self.client.count_tokens(text)
            return result.total_tokens
        
        except Exception as e:
            # Fallback
            return len(text) // 4
        
    def get_embedding_dimension(self) -> int:
        """Get Gemini embedding dimension"""
        return self.EMBEDDING_DIMENSION
    
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate total tokens used"""
        return self.count_tokens(prompt) + self.count_tokens(response)
    
    def _get_finish_reason(self, response) -> str:
        """Extract finish reason from Gemini response"""
        try:
            if response.candidates:
                return response.candidates[0].finish_reason.name
            return "UNKNOWN"
        except:
            return "UNKNWON"
        
    def validate_api_key(self) -> bool:
        """
        Validate Gemini API key by making a test request

        Returns:
            True if valid, False otherwise
        """
        if not super().validate_api_key():
            return False
        
        try:
            # Try to list models as validation
            genai.configure(api_key=self.api_key)
            list(genai.list_models())
            return True
        except:
            return False