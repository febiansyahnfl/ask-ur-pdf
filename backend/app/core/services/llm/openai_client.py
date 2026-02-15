"""
OpenAI LLM Client Implementation
Fallback LLM provider for Ask Your PDF
"""

import time
from typing import List, Optional, Dict, Any
import openai
from openai import OpenAI, OpenAIError

from .base_llm import BaseLLM, LLMResponse, EmbeddingResponse

class OpenAIClient(BaseLLM):
    """
    OpenAI client implementation

    Feature:
    - Text generatino with GPT-3.5
    - Embeddings with text-embedding-ada-002
    - Automatic retry on rate limits
    - Token counting with tiktoken
    """

    # Supported models 
    SUPPORTED_MODELS = {
        "gpt-3.5-turbo": {"max_tokens": 4096, "context_window": 16385},
        "gpt-3.5-turbo-16k": {"max_tokens": 16384, "context_window": 16385},
        "gpt-4": {"max_tokens": 8192, "context_window": 8192},
        "gpt-4-turbo": {"max_tokens": 4096, "context_window": 128000},
        "gpt-4o": {"max_tokens": 4096, "context_window": 128000},
    }

    # Embedding model 
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSION = 1536

    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported OpenAI model: {self.model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        # Check max_tokens 
        model_limits = self.SUPPORTED_MODELS[self.model_name]
        max_token_limit = model_limits['max_tokens']
        if self.max_tokens > max_token_limit:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) exceeds model limit "
                f"({max_token_limit}) for {self.model_name}"
            )
    
    def _initialize_client(self) -> None:
        """Initilize OpenAI SDK"""
        try:
            # Create OpenAI client
            self.client = OpenAI(api_key=self.api_key)

            # Store generation parameters
            self.generation_params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.extra_params.get("top_p", 1.0),
                "frequency_penalty": self.extra_params.get("frequency_penalty", 0.0),
                "presence_penalty": self.extra_params.get("presence_penalty", 0.0)
            }

        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI cleint: {e}")
    
    def generate(
            self, 
            prompt: str, 
            system_prompt: Optional[str] = None, 
            temperature: Optional[float] = None, 
            max_tokens: Optional[int] = None, 
            **kwargs
        ) -> LLMResponse:
        """
        Generate text using OpenAI Chat Completions

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse with generated text
        """
        # Build message
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Override params if needed 
        params = self.generation_params.copy()
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Add messages 
        params["messages"] = messages

        # Generate with retry logic 
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 2)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)

                # Extract text
                text = response.choices[0].message.content

                # Building standardized response 
                return LLMResponse(
                    text=text,
                    model=response.model,
                    token_used=response.usage.total_tokens,
                    finish_reason=response.choices[0].finish_reason,
                    metadata={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "system_fingerprint": response.system_fingerprint
                    }
                )
            
            except openai.RateLimitError as e:
                # Rate limit - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"OpenAI rate limit exceeded: {e}")
                
            except openai.APIError as e:
                raise RuntimeError(f"OpenAI API error: {e}")

            except Exception as e:
                raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def get_embedding(self, text: str, **kwargs) -> EmbeddingResponse:
        """
        Generate embedding using OpenAI embedding model

        Args:
            text: Text to embed

        Returns:
            EmbeddingResponse with 1536-dimensional vecctor
        """
        try:
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text
            )

            embedding = response.data[0].embedding

            return EmbeddingResponse(
                embedding=embedding,
                model=self.EMBEDDING_MODEL,
                dimension=self.EMBEDDING_DIMENSION,
                metadata={
                    "usage": response.usage.total_tokens,
                    "text_length": len(text)
                }
            )
        
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken


        Args:
            text: Text to count

        Returns: Token count
        """
        try:
            import tiktoken

            # Get encoding for model 
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fallback to to cl100k_base (GPT-3.5/GPT-4)
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except ImportError:
            return len(text) // 4

        except Exception as e:
            return len(text) // 4
    
    def get_embedding_dimension(self) -> int:
        """Get OpenAI embedding dimension"""
        return self.EMBEDDING_DIMENSION
    
    def validate_api_key(self) -> bool:
        """
        Validate OpenAI API key by making a test request

        Returns:
            True if valid, else otherwise
        """
        if not super().validate_api_key():
            return False

        try:
            # Try to list models as validation
            self.client.models.list()
            return True
        except:
            return False