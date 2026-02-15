"""
LLM Service Package
Multi-LLM support with factory pattern

Usage:
    from app.core.services.llm import create_llm_from_settings, LLMFactory

    # Option 1: Quick creation from settings
    llm = create_llm_from_settings(settings)

    OPtion 2: Factory with custom config
    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={"gemini": "...", "openai": "..."},
        model_config=model_config,
        fallback_provider="openai"
    )
    llm = factory.create_llm()

    # Use the LLM
    response = llm.generate("What is RAG?")
    print(response.text)

    embedding = llm.get_embedding("Document text")
    print(f"Embedding dimension: {len(embedding.embedding)}")
"""

from .base_llm import BaseLLM, LLMResponse, EmbeddingResponse
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .llm_factory import LLMFactory, LLMProvider, create_llm_from_settings

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMResponse",
    "EmbeddingResponse",

    # Implementations
    "GeminiClient",
    "OpenAIClient",

    # Factory 
    "LLMFactory",
    "LLMProvider",
    "create_llm_from_settings"
]