"""
Abstract Base Class for LLM Clients
Defines the contract for all LLM implementation (Gemini, OpenAI)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standardized LLM Response format"""
    text: str
    model: str
    token_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EmbeddingResponse:
    """Standardized embedding response format"""
    embedding: List[float]
    model: str
    dimension: int
    metadata: Optional[Dict[str, Any]] = None

class BaseLLM(ABC):
    """
    Abstract base class for LLM clients.
    All LLM providers must implement these methods

    Design Pattern: Template Method Pattern
    - Common initialization in __init__
    - Provider-specific logic in abstract methods
    - Consistent interface for factory pattern
    """

    def __init__(
            self,
            api_key: str,
            model_name: str,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            **kwargs
    ):
        """
        Initialize base LLM client
        
        Args:
            :param api_key: API key for the provider
            :type model_name: Model identifier (e.g., 'gemini-pro')
            :param temperature: Sampling temperature
            :type max_tokens: Maximum tokens in response
            :param kwargs: Provider-specific parameters
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

        # Validate on initialization
        self._validate_config()

        # Initialize provider-specific client
        self._initialize_client()
        
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate provider-specific configuration
        Raises ValueError if config is invalid
        """
        pass

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the provider's SDK/client
        Should set self.cleint attribute
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion from prompt
        
        Args:
            :param prompt: User prompt/question
            :type system_prompt: Optional system instruction
            :param temperature: Overide default temperature
            :type max_tokens: Overide default max tokens
            :param kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            Exception: Provider-specific errors (API errors, rate limits)
        """
        pass

    @abstractmethod
    def get_embedding(
        self,
        text: str,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Generate embedding vector for text

        Args:
            text: Text to embed
            **kwargs: Provider-specific parameters

        Returns:
            EmbeddingResponse with embedding vector and metadata

        Raises:
            Exception: Provider-specific errors
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using provider's tokenizer

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for this provider/model

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """
        Check if API key is valid

        Returns:
            True if API key is valid, False otherwise 
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    def __str__(self) -> str:
        """String representation of LLM client"""
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self):
        """Detailed representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name},"
            f"temperature={self.temperature},"
            f"max_tokens={self.max_tokens})"
        )