"""Hugging Face Spaces deployment entry point."""

import os
import uvicorn
from src.api.main import app

# Set environment for HF Spaces
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("DEBUG", "false")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF Spaces default port
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )