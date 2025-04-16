from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages # Correct import path
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our graph using a message list.

    Attributes:
        messages: The list of messages, managed by add_messages.
                  Includes HumanMessage with text/images and AIMessages.
        agent_decision: The decision made by the router ('agent1', 'agent2', 'clarify').
        error: Stores any error message encountered during processing.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    agent_decision: Optional[str]
    error: Optional[str]
    # input_query, input_images_base64, response, clarification_message are removed
    # as they are now handled within 'messages'.