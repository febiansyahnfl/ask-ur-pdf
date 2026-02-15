"""
LLM Factory Pattern
Creates LLM clients based on configuration with automatic fallback
"""

from typing import Dict, Optional, Type
from enum import Enum

from .base_llm import BaseLLM
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"

class LLMFactory:
    """
    Factory for creating LLM clients

    Features:
    - Dynamic provider switching based on config
    - Automatic fallback if primary provider fails
    - API key validation
    - Extensible for new providers

    Usage:
        factory = LLMFactory(config)
        llm = factory.create_llm()
    """

    # Registry of available providers
    _providers: Dict[LLMProvider, Type[BaseLLM]] = {
        LLMProvider.GEMINI: GeminiClient,
        LLMProvider.OPENAI: OpenAIClient,
    }

    def __init__(
            self,
            primary_provider: str,
            api_keys: Dict[str, str],
            model_config: Dict,
            fallback_provider: Optional[str] = None,
            auto_fallback: bool = True
        ):
        """
        Initialize LLM Factory

        Args:
            primary_provider: Primary LLM provider
            api_keys: Dict of API keys
            model_config: Model configuration fom YAML
            fallback_provider: Optional fallback provider
            auto_fallback: Automatically fallback on primary failure
        """
        self.primary_provider = LLMProvider(primary_provider.lower())
        self.fallback_provider = LLMProvider(fallback_provider.lower())
        self.api_keys = api_keys
        self.model_config = model_config
        self.auto_fallback = auto_fallback

        # Validate providers 
        self._validate_providers()

    def _validate_providers(self) -> None:
        """Validate that providers are suppoerted"""
        if self.primary_provider not in self._providers:
            raise ValueError(
                f"Unsupported primary provider: {self.primary_provider}. "
                f"Supported: {[p.value for p in self._providers.keys()]}"
            )
        
        if self.fallback_provider and self.fallback_provider not in self._providers:
            raise ValueError(
                f"Unsupported fallback provider: {self.fallback_provider}. "
                f"Supported: {[p.value for p in self._providers.keys()]}"
            )

    def create_llm(
            self,
            provider: Optional[str] = None,
            **override_params
    ) -> BaseLLM:
        """
        Create LLm client

        Args:
            provider: Override provider
            **override_params: Override model config parameters

        Returns:
            Initialized LLM client

        Raises:
            RuntimeError: If LLm creation fails
        """
        # Determine which provider to use 
        target_provider = LLMProvider(provider.lower()) if provider else self.primary_provider

        try:
            # Try to create primary LLM 
            llm = self._create_provider_client(target_provider, **override_params)

            # Validate API key 
            if not llm.validate_api_key():
                raise ValueError(f"Invalid API key for {target_provider.value}")
            
            return llm
        
        except Exception as e:
            # If auto_fallback enabled and we have a fallback provider 
            if self.auto_fallback and self.fallback_provider and target_provider != self.fallback_provider:
                print(f"Primary LLm ({target_provider.value}) failed: {e}")
                print(f"Falling back to {self.fallback_provider.value}...")

                try:
                    fallback_llm = self._create_provider_client(self.fallback_provider, **override_params)

                    if not fallback_llm.validate_api_key():
                        raise ValueError(f"Invalid API key for {self.fallback_provider.value}")
                    
                    print(f"Fallback LLM ({self.fallback_provider.value}) initialized successfully")
                    return fallback_llm
                
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Both primary ({target_provider.value}) and fallback ({self.fallback_provider.value}) LLMs failed. "
                        f"Primary error: {e}. Fallback error: {fallback_error}"
                    )
            else:
                raise RuntimeError(f"Failed to create LLM for {target_provider.value}: {e}")

    def _create_provider_client(
            self,
            provider: LLMProvider,
            **override_params
    ) -> BaseLLM:
        """
        Create client for specific provider

        Args:
            provider: LLM provider enum
            **override_params: Override config parameters

        Returns:
            Initialized LLM client
        """
        # Get API key 
        api_key = self.api_keys.get(provider.value)
        if not api_key:
            raise ValueError(f"Api key not found for {provider.value}")
        
        # Get model config
        provider_config = self.model_config.get("models", {}).get(provider.value, {})
        if not provider_config:
            raise ValueError(f"Model config not found for {provider.value}")
        
        # Build parameters 
        params = {
            "api_key": api_key,
            "model_name": provider_config.get("name"),
            "temperature": provider_config.get("temperature", 0.7),
            "max_tokens": provider_config.get("max_tokens", 2048),
            **{k: v for k, v in provider_config.items() if k not in ["name", "temperature", "max_tokens"]}
        }

        # Apply overrides 
        params.update(override_params)

        # Get client class and instantiate
        client_class = self._providers[provider]
        return client_class(**params)

    def get_available_providers(self) -> list[str]:
        """
        Get list of available providers with valid API keys

        Returns:
            List of provider names
        """
        available = []
        for provider in self._providers.keys():
            if provider.value in self.api_keys and self.api_keys[provider.value]:
                available.append(provider.value)
        return available

    def get_embedding_dimension(
            self, provider: Optional[str] = None
    ) -> int:
        """
        Get embedding dimension for a provider

        Args:
            provider: Provider name

        Returns:
            Embedding dimension
        """
        target_provider = LLMProvider(provider.lower()) if provider else self.primary_provider
        
        # Create temporary client to get dimension
        try:
            client = self._create_provider_client(target_provider)
            return client.get_embedding_dimension()
        except:
            # Fallback to knwon dimensions
            if target_provider == LLMProvider.GEMINI:
                return GeminiClient.EMBEDDING_DIMENSION
            elif target_provider == LLMProvider.OPENAI:
                return OpenAIClient.EMBEDDING_DIMENSION
            return 768 #Default

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        client_class: Type[BaseLLM]
    ) -> None:
        """
        Regiter a new LLM provider

        Args:
            provider_name: Provider identifier
            client_class: BaseLLM implementation class

        Example:
            LLMFactory.register_provider("provider", ProviderClient)
        """
        provider_enum = LLMProvider(provider_name.lower())
        cls._providers[provider_enum] = client_class

    def __str__(self) -> str:
        """String representation"""
        return (
            f"LLMFactory(primary={self.primary_provider.value}, "
            f"fallback={self.fallback_provider.value if self.fallback_provider else None})"
        )

# Convenience function for quick LLM creation
def create_llm_from_settings(settings) -> BaseLLM:
    """
    Create LLM client from application settings

    Args:
        settings: Application settings object with llm_provider, api_keys, and model_config

    Returns:
        Initialized LLM client

    Usage:
        from app.config.settings import settins
        llm = create_llm_from_settings(settings)
    """
    factory = LLMFactory(
        primary_provider=settings.llm_provider,
        api_keys={
            "gemini": settings.gemini_api_key,
            "openai": settings.openai_api_key,
        },
        model_config=settings.model_config.get("llm", {}),
        fallback_provider="openai" if settings.llm_provider == "gemini" else None,
        auto_fallback=True
    )

    return factory.create_llm()