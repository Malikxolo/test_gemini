"""
Configuration module for centralized environment variable management.
Loads all API keys and settings from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# RAG API
RAG_API_URL = os.getenv("RAG_API_URL")
BOT_API_KEY = os.getenv("BOT_API_KEY")

# Recall.ai
RECALLAI_API_KEY = os.getenv("RECALLAI_API_KEY")
RECALLAI_REGION = os.getenv("RECALLAI_REGION", "us-west-2")

# Webhook
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL")


# Validation
def validate_config():
    """Validate that required environment variables are set."""
    required = {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
        "RAG_API_URL": RAG_API_URL,
        "BOT_API_KEY": BOT_API_KEY,
        "RECALLAI_API_KEY": RECALLAI_API_KEY,
        "WEBHOOK_BASE_URL": WEBHOOK_BASE_URL,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return True


# Run validation on import
validate_config()
