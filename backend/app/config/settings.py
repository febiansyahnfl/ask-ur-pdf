import os
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from .env"""

    # API KEYs
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # LLM Configuration 
    LLM_PROVIDER: str = "gemini"

    # Application
    APP_NAME: str = "Ask Your PDF"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Session
    SESSION_LIFETIME_HOURS: int = 2
    MAX_FILE_SIZE_MB: int = 10

    # Vector Store
    VECTOR_STORE_TYPE: str = "faiss"

    # Server
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:8501"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()

def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    config_path = Path(__file__).parent.parent.parent / "config" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Loads Configs
settings = get_settings()
model_config = load_yaml_config("model_config")
logging_config = load_yaml_config("logging_config")