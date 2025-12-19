#!/usr/bin/env python3
"""Startup script for Render.com deployment - Lightweight version."""

import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment (Render.com provides this)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the lightweight FastAPI application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )