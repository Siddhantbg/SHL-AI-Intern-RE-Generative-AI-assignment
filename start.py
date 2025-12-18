#!/usr/bin/env python3
"""Startup script for Render.com deployment."""

import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment (Render.com provides this)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the FastAPI application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )