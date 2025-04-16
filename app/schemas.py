from pydantic import BaseModel, Field
from typing import Literal, Optional

class RouterOutput(BaseModel):
    """Structured output for the router LLM."""
    decision: Literal["agent1", "agent2", "clarify"] = Field(
        ..., # Ellipsis means this field is required
        description="The final routing decision: 'agent1' for visual issues with images, 'agent2' for tenancy FAQs, or 'clarify' if unsure or needs more info."
    )
    clarification_message: Optional[str] = Field(
        default=None, # Default to None
        description="The clarification message to ask the user. ONLY provide this text if the decision is 'clarify'. Otherwise, leave it null or omit it."
    )