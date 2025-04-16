import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .graph_state import GraphState
# Import only the needed node functions
from .graph_nodes import (
    route_request,
    execute_agent1,
    execute_agent2,
    # pass_through_clarification # No longer needed
)

logger = logging.getLogger(__name__)

# --- Conditional Edge Logic (remains the same structure) ---

def decide_next_node(state: GraphState) -> str:
    """Determines the next node based on the router's decision."""
    decision = state.get("agent_decision")
    logger.info(f"Conditional Edge: Routing based on decision '{decision}'")
    if decision == "agent1":
        return "issue_detector"
    elif decision == "agent2":
        return "faq_agent"
    elif decision == "clarify":
        # Router added clarification AIMessage, graph can end.
        return "clarify" # This key name MUST match the one used in add_conditional_edges
    else:
        logger.warning(f"Invalid agent_decision '{decision}' in state during edge decision. Ending graph.")
        # Decide on a fallback - maybe END or route to a specific error node if you add one
        return END # Or perhaps "clarify" if you want the router's fallback message to be the output

# --- Build the Graph ---

def build_graph():
    memory = MemorySaver()
    """Builds and compiles the LangGraph StateGraph."""
    logger.info("Building graph...")
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", route_request)
    workflow.add_node("issue_detector", execute_agent1)
    workflow.add_node("faq_agent", execute_agent2)
    # No separate "clarifier" node definition needed

    # Set entry point
    workflow.set_entry_point("router")
    logger.info("Entry point set to 'router'.")

    # Add conditional edges from the router
    workflow.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "issue_detector": "issue_detector",
            "faq_agent": "faq_agent",
            # If router decided 'clarify', it added the AIMessage, so we end.
            "clarify": END, # Route directly to END for the 'clarify' decision
            # Add handling for the END fallback from decide_next_node if needed
            END: END
        }
    )
    logger.info("Conditional edges added from 'router'.")

    # Add edges from agent nodes to END
    workflow.add_edge("issue_detector", END)
    workflow.add_edge("faq_agent", END)
    logger.info("Edges added from agent nodes to END.")

    # Compile the graph
    app_graph = workflow.compile(checkpointer=memory)
    logger.info("Graph compiled successfully.")
    return app_graph