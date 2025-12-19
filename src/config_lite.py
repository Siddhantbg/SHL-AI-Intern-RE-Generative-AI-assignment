"""
Lightweight configuration for memory-constrained environments like Render free tier.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class LiteSettings(BaseSettings):
    """Lightweight settings for reduced memory usage."""
    
    # API Configuration
    api_port: int = int(os.getenv("PORT", 8000))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "production")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    # LLM Configuration
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # Memory optimization settings
    max_assessments_in_memory: int = 100  # Limit assessments loaded
    use_simple_embeddings: bool = True  # Use lightweight embeddings
    lazy_load_models: bool = True  # Load models only when needed
    
    # Data directories (use /tmp for ephemeral storage on Render)
    data_dir: Path = Path("/tmp/shl_data")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_lite_settings() -> LiteSettings:
    """Get lightweight settings instance."""
    return LiteSettings()