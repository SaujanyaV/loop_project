import logging
from typing import Dict, Any, List, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from .schemas import RouterOutput
from .graph_state import GraphState
from .llm_models import router_llm, agent1_llm, agent2_llm

logger = logging.getLogger(__name__)

# Helper function to extract text from message content
def get_text_from_message(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
    return "" # Return empty string if no text found

# Helper function to check for images in message content
def has_images_in_message(message: BaseMessage) -> bool:
    if isinstance(message.content, list):
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False

async def route_request(state: GraphState) -> Dict[str, Any]:
    """
    Determines which agent should handle the request using an LLM with
    structured output, CONSIDERING conversation history.
    Updates 'agent_decision' and potentially adds a
    clarification 'AIMessage' to the state's 'messages'.
    """
    logger.info("--- Routing Request (Structured Output with History) ---")
    messages = state["messages"]
    if not messages:
        logger.error("Router called with empty messages state.")
        return {
            "agent_decision": "clarify",
            "messages": [AIMessage(content="Something went wrong, no user message found.")],
            "error": "Routing failed: No messages in state."
        }

    # --- Consider History ---
    # Combine message content for the prompt. Be mindful of token limits for long histories.
    # Option 1: Simple concatenation (good for short histories)
    conversation_history_text = "\n".join(
        [f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {get_text_from_message(msg)}" for msg in messages]
    )
    # Option 2: Last few messages (safer for token limits)
    # num_messages_for_context = 5 # Adjust as needed
    # recent_messages = messages[-num_messages_for_context:]
    # conversation_history_text = "\n".join(
    #     [f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {get_text_from_message(msg)}" for msg in recent_messages]
    # )

    # Check for images in the *last* user message specifically for agent1 routing override
    last_message = messages[-1]
    last_query_text = get_text_from_message(last_message) # Keep for override logic if needed
    has_images_in_last_message = has_images_in_message(last_message)

    logger.info(f"Routing based on History (last part): '...{conversation_history_text[-200:]}', Has Images in Last Msg: {has_images_in_last_message}")

    structured_llm = router_llm.with_structured_output(RouterOutput)

    # --- Updated Prompt ---
    prompt_text = f"""You are an expert router for a real estate assistant chatbot. Classify the user's *latest* request based on the conversation history provided.

Available Agents:
1.  **Agent 1 (Issue Detector)**: Handles visual inspection. Requires images *in the latest message*. Choose 'agent1' if images are present in the last message AND the query concerns a visual aspect.
2.  **Agent 2 (Tenancy FAQ)**: Handles text-only tenancy questions (laws, agreements, etc.). Choose 'agent2' if the latest query is about these topics and no relevant images are present in the last message.
3.  **Clarification**: Handles simple conversation, greetings, remembers previous context, or asks for more details. Choose 'clarify' if unsure, if more info is needed (e.g., location), for simple chat (like remembering a name based on history), or if the request doesn't fit Agent 1 or 2. Be friendly and act as an assistant.

Conversation History:
--- START HISTORY ---
{conversation_history_text}
--- END HISTORY ---

Images provided in the *latest* user message: {"Yes" if has_images_in_last_message else "No"}

Based on the *latest* user request within the context of the history, respond using the 'RouterOutput' structure:
{{
  "decision": "'agent1' | 'agent2' | 'clarify'",
  "clarification_message": "string | null" # ONLY provide if decision is 'clarify' AND you need to ask the user something OR provide a conversational reply. If you remember something (like a name) just make the decision 'clarify' and set this message to the answer.
}}"""

    try:
        # Pass only the system prompt to the router LLM
        response_object: RouterOutput = await structured_llm.ainvoke([SystemMessage(content=prompt_text)])

        decision = response_object.decision
        clarification = response_object.clarification_message

        logger.info(f"Router LLM structured decision: '{decision}', Clarification provided: {clarification is not None}")

        # --- Override Logic (using last_query_text and has_images_in_last_message) ---
        final_decision = decision
        # Check if images are present in the last message specifically for overrides
        if has_images_in_last_message and decision == "agent2":
            # Your existing FAQ keyword logic can stay if needed
            faq_keywords = ["lease", "rent", "landlord", "tenant", "deposit", "agreement", "eviction", "notice", "law", "rights"]
            if not any(keyword in last_query_text.lower() for keyword in faq_keywords):
                 logger.warning(f"OVERRIDE: LLM chose '{decision}' but images are present in last msg and query lacks strong FAQ keywords. Switching to 'agent1'.")
                 final_decision = "agent1"
                 clarification = None # Agent 1 doesn't need a clarification message from the router
            else:
                 logger.info("LLM chose 'agent2' despite image in last msg, query text seems FAQ-related. Proceeding.")
        elif not has_images_in_last_message and decision == "agent1":
             logger.warning(f"OVERRIDE: LLM chose '{decision}' but no images provided in last msg. Switching to 'clarify'.")
             final_decision = "clarify"
             # Ensure clarification exists or provide a default
             clarification = clarification or "It looks like you might need help with a visual issue, but you didn't provide an image in your last message. Could you upload one? Or is your question about something else?"
        elif final_decision == "clarify" and not clarification:
             # If the LLM decided to clarify but didn't provide text (e.g., it handled a simple chat internally),
             # you might need a default, or trust it doesn't need one if it's just acknowledging something.
             # Let's add a default *just in case* it was an error state.
             logger.warning("LLM chose 'clarify' but didn't provide a message. Adding default asking for more info.")
             clarification = "How can I help you further? Please ask a question about a property issue (with an image if needed) or a tenancy topic."
             # **** IMPORTANT: If the user asked "What's my name?", the LLM *should* have put the name in the clarification field based on the updated prompt. ****
             # If it didn't, the prompt might need further tweaking.

        # --- End Override Logic ---

        logger.info(f"Final Routing Decision: {final_decision}")

        update_dict: Dict[str, Any] = {"agent_decision": final_decision}
        if final_decision == "clarify" and clarification: # Only add message if clarification text exists
            # Add the clarification/response as an AIMessage to the history
            update_dict["messages"] = [AIMessage(content=clarification)]
            logger.info(f"Clarification/Response AIMessage added to state: '{clarification}'")

        return update_dict

    except Exception as e:
        logger.error(f"Error during structured routing LLM call: {e}", exc_info=True)
        # Fallback: clarify and add error message to state
        clarification_msg = "Sorry, I encountered an issue routing your request. Could you please rephrase or specify if it's about a visual issue (with image) or a tenancy question?"
        return {
            "agent_decision": "clarify",
            "messages": [AIMessage(content=clarification_msg)],
            "error": f"Routing failed: {e}"
        }

async def execute_agent1(state: GraphState) -> Dict[str, Any]:
    """
    Handles image-based issue detection using a vision model.
    Appends an AIMessage with the analysis to the state's 'messages'.
    """
    logger.info("--- Executing Agent 1 (Issue Detector) ---")
    # The user message (with images) is the last one in the state
    user_message = state["messages"][-1]
    # History is everything before the last message
    history = state["messages"][:-1]

    system_prompt = """You are an expert real estate issue detection assistant. Analyze the provided image(s) and the user's query text from the latest message to identify potential property problems (e.g., water damage, mold, cracks).

Provide a clear analysis:
1.  **Identify** visible issues.
2.  **Explain** potential causes.
3.  **Suggest** troubleshooting or professional help.
4.  Ask **clarifying questions** if needed.
Be concise and helpful. Base your analysis on the latest user message and images."""

    # Combine history (if any) and the latest user message for the LLM call
    messages_to_llm = history + [user_message] if history else [user_message]

    try:
        # Pass the System Prompt AND the message list (including images) to the LLM
        ai_response: AIMessage = await agent1_llm.ainvoke([
            SystemMessage(content=system_prompt),
            *messages_to_llm # Unpack the list of messages
        ])
        logger.info("Agent 1 LLM call successful.")
        # Return the AI's response message to be appended to the state
        return {"messages": [ai_response]}
    except Exception as e:
        logger.error(f"Error during Agent 1 LLM call: {e}", exc_info=True)
        # Return an error message as an AIMessage
        error_ai_msg = AIMessage(content="Sorry, I encountered an error analyzing the image(s).")
        return {"messages": [error_ai_msg], "error": f"Agent 1 failed: {e}"}


async def execute_agent2(state: GraphState) -> Dict[str, Any]:
    """
    Handles text-based tenancy FAQs.
    Appends an AIMessage with the answer to the state's 'messages'.
    """
    logger.info("--- Executing Agent 2 (Tenancy FAQ) ---")
    user_message = state["messages"][-1]
    history = state["messages"][:-1]
    # Extract just the text for the system prompt context if needed,
    # but the main query is in the user_message passed to invoke.
    query_text = get_text_from_message(user_message)

    system_prompt = f"""You are a helpful assistant knowledgeable about general real estate tenancy topics (laws, agreements, rent, deposits, eviction etc.). Answer the user's question from their latest message.

Provide clear, concise information based on common practices.
**Important**: If the question needs specific legal advice or depends on local regulations, state your answer is general and advise the user to mention their location or consult a local expert/organization. Do not invent local laws.

Focus on the latest user query text: '{query_text[:200]}...'""" # Optional: include truncated query for context

    messages_to_llm = history + [user_message] if history else [user_message]

    try:
        # Pass System Prompt and the message list to the LLM
        ai_response: AIMessage = await agent2_llm.ainvoke([
            SystemMessage(content=system_prompt),
            *messages_to_llm # Unpack the list of messages
        ])
        logger.info("Agent 2 LLM call successful.")
        return {"messages": [ai_response]}
    except Exception as e:
        logger.error(f"Error during Agent 2 LLM call: {e}", exc_info=True)
        error_ai_msg = AIMessage(content="Sorry, I encountered an error answering your question.")
        return {"messages": [error_ai_msg], "error": f"Agent 2 failed: {e}"}
    