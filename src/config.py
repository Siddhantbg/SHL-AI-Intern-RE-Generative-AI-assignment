"""Configuration management for the SHL Assessment Recommendation System."""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Application settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "info")
        
        # API settings
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_workers = int(os.getenv("API_WORKERS", "1"))
        
        # LLM service settings
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Database settings
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Scraping settings
        self.scraping_delay = int(os.getenv("SCRAPING_DELAY", "1"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # Embedding settings
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", "384"))
        
        # Evaluation settings
        self.evaluation_data_path = os.getenv("EVALUATION_DATA_PATH", "data/evaluation/")
        self.training_data_file = os.getenv("TRAINING_DATA_FILE", "train_queries.json")
        self.test_data_file = os.getenv("TEST_DATA_FILE", "test_queries.json")
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings


def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        settings.data_dir,
        settings.models_dir,
        settings.logs_dir,
        settings.data_dir / "scraped",
        settings.data_dir / "processed",
        settings.data_dir / "embeddings",
        settings.data_dir / "evaluation",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Create directories when run directly
    create_directories()
    print("Configuration loaded successfully!")
    print(f"Environment: {settings.environment}")
    print(f"API Host: {settings.api_host}:{settings.api_port}")
    print(f"Data directory: {settings.data_dir}")