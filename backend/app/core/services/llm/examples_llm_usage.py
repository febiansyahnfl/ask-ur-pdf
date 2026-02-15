"""
LLM Services - Usage Examples & Testing Guide

This file demonstrates how to use the LLM factory pattern in the Ask Your PDF project.
"""

# =========================================
# EXAMPLE 1: Basic Usage with Factory
# =========================================

def example_basic_usage():
    """Basic LLM usage with factory pattern"""
    from app.core.services.llm import LLMFactory
    import os

    # Initialize factory
    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY")
        },
        model_config={
            "models": {
                "gemini": {
                    "name": "gemini-2.5-flash",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.95
                },
                "openai": {
                    "name": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            }
        },
        fallback_provider="openai",
        auto_fallback=True
    )

    # Create LLM client
    llm = factory.create_llm()
    print(f"Created LLM: {llm}")

    # Generate text
    response = llm.generate(
        prompt="Explain what RAG is in 2 sentences.",
        system_prompt="Your are a helpful AI assistant."
    )

    print(f"\n Response: {response.text}")
    print(f"Tokens used: {response.token_used}")
    print(f"Finish reason: {response.finish_reason}")

# =========================================
# EXAMPLE 2: Using Settings from config
# =========================================

def example_from_settings():
    """Create LLM from application settings"""
    from app.config.settings import settings
    from app.core.services.llm import create_llm_from_settings

    # One-liner to create LLM from settings
    llm = create_llm_from_settings(settings)

    # Use it 
    response = llm.generate("What is a vectore database?")
    print(response.text)

# =========================================
# EXAMPLE 3: Generate Embeddings
# =========================================

def example_embeddings():
    """Generate embeddings for text"""
    from app.core.services.llm import LLMFactory
    import os

    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={"gemini": os.getenv("GEMINI_API_KEY")},
        model_config={
            "models": {
                "gemini": {"name": "gemini-2.5-flash", "temperature": 0.7, "max_tokens": 2048}
            }
        }
    )

    llm = factory.create_llm()

    # Generate embedding 
    text = "FAISS is a library for efficient similarity search."
    embedding_response = llm.get_embedding(text)

    print(f"Embedding dimension: {embedding_response.dimension}")
    print(f"Embedding model: {embedding_response.model}")
    print(f"First 5 values: {embedding_response.embedding[:5]}")

# =========================================
# EXAMPLE 4: Override Parameters
# =========================================

def example_override_params():
    """Override default parameters for specific request"""
    from app.core.services.llm import create_llm_from_settings
    from app.config.settings import settings

    llm = create_llm_from_settings(settings)

    # User custom temperature and max_tokens
    response = llm.generate(
        prompt="Write a creative story opening.",
        temperature=0.9,
        max_tokens=500
    )

    print(response.text)

# =========================================
# EXAMPLE 5: Automatic Fallback
# =========================================

def example_automatic_fallback():
    """Demonstarte automatic fallback on primary failure"""
    from app.core.services.llm import LLMFactory
    import os

    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={
            "gemini": "invalid-key",
            "openai": os.getenv("OPENAI_API_KEY")
        },
        model_config={
            "models": {
                "gemini": {"name": "gemini-2.5-flash", "temperature": 0.7, "max_tokens": 2048},
                "openai": {"name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 2048}
            }
        },
        fallback_provider="openai",
        auto_fallback=True
    )

    # This will be automatically fallback to OpenAI
    llm = factory.create_llm()
    response = llm.generate("Hello, world!")
    print(response.text)

# =========================================
# EXAMPLE 6: Force Spesific Provider
# =========================================

def example_force_provider():
    """Force using a specific provider (bypass primary)"""
    from app.core.services.llm import LLMFactory
    import os

    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY")
        },
        model_config={
            "models": {
                "gemini": {"name": "gemini-2.5-flash", "temperature": 0.7, "max_tokens": 2048},
                "openai": {"name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 2048}
            }
        }
    )

    # Forec use OpenAI even though Gemini is primary 
    openai_llm = factory.create_llm(provider="openai")
    print(f"Using: {openai_llm}")

    response = openai_llm.generate("Test")
    print(response.text)

# =========================================
# EXAMPLE 7: Token Counting
# =========================================

def example_token_counting():
    """Count tokens in text"""
    from app.core.services.llm import create_llm_from_settings
    from app.config.settings import settings

    llm = create_llm_from_settings(settings)

    text = "This is a sample text for token counting. It has multiple sentences."
    token_count = llm.count_tokens(text)

    print(f"Text: {text}")
    print(f"Token count: {token_count}")

# =========================================
# EXAMPLE 8: Check Available Providers
# =========================================
def example_check_providers():
    """Check which providers are available"""
    from app.core.services.llm import LLMFactory
    import os

    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": None
        },
        model_config={
            "models": {
                "gemini": {"name": "gemini-2.5-flash", "temperature": 0.7, "max_tokens": 2048}
            }
        }
    )

    available = factory.get_available_providers()
    print(f"Available providers: {available}")

# =========================================
# EXAMPLE 9: Get Embedding Dimensions
# =========================================

def example_embedding_dimensions():
    """Get embedding dimensions for each provider"""
    from app.core.services.llm import LLMFactory
    import os

    factory = LLMFactory(
        primary_provider="gemini",
        api_keys={"gemini": os.getenv("GEMINI_API_KEY")},
        model_config={
            "models": {
                "gemini": {"name": "gemini-2.5-flash", "tempreature": 0.7, "max_tokens": 2048}
            }
        }
    )

    gemini_dim = factory.get_embedding_dimension("gemini")
    print(f"Gemini embedding dimension: {gemini_dim}")

# =========================================
# EXAMPLE 10: Error Handling
# =========================================

def example_error_handling():
    """Proper error handling"""
    from app.core.services.llm import LLMFactory

    try:
        factory = LLMFactory(
            primary_provider="gemini",
            api_keys={"gemini": "invalid-key"},
            model_config={
                "models": {
                    "gemini": {"name": "gemini-2.5-flash", "tempreature": 0.7, "max_tokens": 2048}
                }
            },
            auto_fallback=False
        )

        llm = factory.create_llm()

    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"LLM creation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# =========================================
# TESTING GUIDE
# =========================================

"""
TESTING CHECKLIST:

1. Test Gemini Client
    - Set GEMINI_API_KEY in .env
    - Run: python -c "from examples_llm_usgae import test_gemini; test_gemini()"

2. Test OpenAI Client
    - Set OPENAI_API_KEY in .env
    - Run: python -c "from examples_llm_usage import test_openai; test_openai()"

3. Test Factory Pattern
    - Run: python -c "from examples_llm_usage import test_factory; test_factory()"

4. Test Automatic Fallback
    - Temporarily break Gemini kye
    - Run: python -c "from examples_llm_usage import example_automatic_fallback; example_automatic_fallback()"

5. Test Embeddings
    - Run: python -c "from examples_llm_usage import example_embeddings; example_embeddings()

6. Integration with RAG
    - Create sample PDF
    - Generate embeddings
    - Store in FAISS
    - Query and retrieve
"""

def test_gemini():
    """Quick test for Gemini client"""
    import os
    from app.core.services.llm import GeminiClient

    client = GeminiClient(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=100
    )

    print("Testing Gemini Client...")
    response = client.generate("Say, 'Gemini is working!'")
    print(f"Response: {response.text}")

    embedding = client.get_embedding("Test text")
    print(f"Embedding dimension: {len(embedding.embedding)}")

def test_openai():
    """Quick test for OpenAI client"""
    import os
    from app.core.services.llm import OpenAIClient

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )

    print("Testing OpenAI Client...")
    response = client.generate("Say, 'OpenAI is working!'")
    print(f"Response: {response.text}")

    embedding = client.get_embedding("Test text")
    print(f"Embedding dimension: {len(embedding.embedding)}")

def test_factory():
    """Quick test for factory pattern"""
    import os
    from dotenv import load_dotenv
    from app.core.services.llm import LLMFactory

    load_dotenv()

    # Get API keys with validation
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key:
        print("⚠️  Warning: GEMINI_API_KEY not found in environment")
    if not openai_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    
    # Build api_keys dict explicitly
    api_keys_dict = {
        "gemini": gemini_key,
        "openai": openai_key
    }
    
    print(f"🔑 API Keys loaded: {list(api_keys_dict.keys())}")
    print(f"   Gemini key present: {bool(gemini_key)}")
    print(f"   OpenAI key present: {bool(openai_key)}")

    factory = LLMFactory(
        primary_provider="gemini",
        api_keys=api_keys_dict,
        model_config={
            "models": {
                "gemini": {"name": "gemini-2.5-flash", "temperature": 0.7, "max_tokens": 100},
                "openai": {"name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 100}
            }
        },
        fallback_provider="openai"
    )

    print("Testing Factory Pattern...")
    llm = factory.create_llm()
    print(f"created: {llm}")

    response = llm.generate("Say, 'Factory is working!'")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    print("=" * 80)
    print("LLM Service - Example & Testing")
    print("=" * 80)

    # Run basic example 
    print("\nRunning basic usage example...")
    example_basic_usage()

    print("\n" + "=" * 80)
    print("Example completed! Check other examples in this file.")
    print("=" * 80)