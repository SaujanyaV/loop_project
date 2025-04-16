# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_HERE")
    # Add other keys if needed, e.g., for different models or services
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "gpt-4o") # Or "gpt-4o"
    ROUTING_MODEL_NAME: str = os.getenv("ROUTING_MODEL_NAME", "gpt-4o") # Cheaper model for routing
    FAQ_MODEL_NAME: str = os.getenv("FAQ_MODEL_NAME", "gpt-4o") # Cheaper model for FAQs

    # Basic check
    if not OPENAI_API_KEY or "YOUR_DEFAULT_KEY_HERE" in OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not found or using default. Please set it in your .env file.")

settings = Settings()

# You might want to add more robust validation here