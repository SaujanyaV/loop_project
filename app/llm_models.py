# app/llm_integrations.py
from langchain_openai import ChatOpenAI
from .config import settings

def get_llm(model_name: str):
    """Initializes and returns a ChatOpenAI model."""
    # Consider adding error handling for invalid keys or model names
    return ChatOpenAI(
        model=model_name,
        openai_api_key=settings.OPENAI_API_KEY,
    )

# Pre-initialize models if desired (can save startup time within nodes)
# Or initialize them within each node function if state/context specific params needed later
router_llm = get_llm(settings.ROUTING_MODEL_NAME)
agent1_llm = get_llm(settings.VISION_MODEL_NAME) # Vision model
agent2_llm = get_llm(settings.FAQ_MODEL_NAME)