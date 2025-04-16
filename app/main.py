# main.py

import base64
import logging
from typing import List, Optional, Dict, Any
import uuid # Import uuid

# Import message types
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .graph_state import GraphState
from .graph_builder import build_graph
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: str # Make session_id mandatory for tracking conversations
    # Images will be handled separately by FastAPI's File/UploadFile

class ChatResponse(BaseModel):
    response: str
    session_id: str # Return the session_id used

class ClearResponse(BaseModel):
    message: str
    session_id: Optional[str] = None # Optionally confirm which session *was* active


# --- FastAPI App Setup ---
app = FastAPI(
    title="Multi-Agent Real Estate Assistant",
    description="A POC chatbot with image and text capabilities using LangGraph.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Graph (as before) ---
try:
    if not settings.OPENAI_API_KEY or "YOUR_DEFAULT_KEY_HERE" in settings.OPENAI_API_KEY:
         logger.warning("OpenAI API Key is missing or default. Graph functionality will likely fail.")
    # Assuming build_graph() correctly sets up MemorySaver internally
    app_graph = build_graph()
    logger.info("Real LangGraph graph loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to load or compile the graph on startup: {e}", exc_info=True)
    app_graph = None # Ensure graph is None if loading fails


# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    session_id: str = Form(...), # Make session_id mandatory
    query: str = Form(...),
    images: List[UploadFile] = File(default=[], description="Optional list of images."),
):
    """
    Handles a chat request for a specific session_id.
    Requires the client to manage and send the session_id.
    """
    if app_graph is None:
        logger.error("Chat endpoint called but graph is not loaded.")
        raise HTTPException(status_code=503, detail="Chatbot Service Unavailable: Core component failed to load.") # 503 might be more appropriate

    if not session_id:
        # This shouldn't happen if Form(...) is used, but belt-and-suspenders
        logger.error("Chat request received without a session_id.")
        raise HTTPException(status_code=400, detail="session_id is required.")

    logger.info(f"Received chat request for session_id: {session_id}")

    # --- Prepare message content (Text + Images) ---
    input_message_content: List[Dict[str, Any]] = [{"type": "text", "text": query}]
    processed_image_count = 0
    image_processing_errors = [] # Keep track of image errors

    if images:
        logger.info(f"Received {len(images)} image(s) for session {session_id}. Processing...")
        for i, image in enumerate(images):
            if not image.content_type or not image.content_type.startswith("image/"):
                logger.warning(f"Invalid file type: {image.filename} ({image.content_type}) for session {session_id}. Skipping.")
                image_processing_errors.append(f"Skipped invalid file: {image.filename}")
                # No need to close already processed ones here, handled in finally
                # await image.close() # Close invalid file immediately
                continue # Skip to next image

            try:
                image_bytes = await image.read()
                if not image_bytes:
                    logger.warning(f"Empty image file received: {image.filename} for session {session_id}. Skipping.")
                    image_processing_errors.append(f"Skipped empty file: {image.filename}")
                    continue

                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                mime_type = image.content_type
                input_message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                })
                processed_image_count += 1
                logger.debug(f"Successfully encoded image: {image.filename} for session {session_id}") # Debug level maybe
            except Exception as e:
                logger.error(f"Error processing image file '{image.filename}' for session {session_id}: {e}", exc_info=True)
                image_processing_errors.append(f"Error processing file: {image.filename}")
            finally:
                # Ensure all files opened in this request are closed, even if skipped
                # This assumes UploadFile needs explicit closing. Check FastAPI docs if unsure.
                await image.close()


        logger.info(f"Processed {processed_image_count} valid images out of {len(images)} received for session {session_id}.")
        if image_processing_errors:
             logger.warning(f"Image processing issues for session {session_id}: {'; '.join(image_processing_errors)}")
             # Optionally prepend a warning to the user's query or handle differently
    else:
        logger.info(f"No images received for session {session_id}.")

    # --- Prepare Initial State & Config ---
    # The state only needs the *current* message. LangGraph's checkpointer
    # will load previous messages for the given thread_id automatically.
    initial_state: GraphState = {
        "messages": [HumanMessage(content=input_message_content)],
        # Initialize other state fields if necessary (agent_decision, error)
        "agent_decision": None,
        "error": None,
    }

    # Configure the graph invocation to use the specific session_id as thread_id
    config = {"configurable": {"thread_id": session_id}}

    try:
        logger.info(f"Invoking graph for session_id: {session_id}, query: '{query[:50]}...' ({processed_image_count} images)")

        # Use ainvoke for async operation
        # LangGraph's MemorySaver checkpointer will automatically load the history
        # for the given thread_id and merge it with the input 'messages'.
        final_state = await app_graph.ainvoke(initial_state, config=config)

        if final_state is None:
             logger.error(f"Graph invocation for session {session_id} yielded no final state.")
             raise HTTPException(status_code=500, detail="Chatbot failed to produce a result.")


        # --- Handle Graph Response (Processing the FINAL state) ---
        response_text = ""
        error_message = final_state.get("error")
        all_messages = final_state.get("messages", [])
        last_message: Optional[BaseMessage] = all_messages[-1] if all_messages else None

        if error_message:
            logger.error(f"Graph execution finished with error for session {session_id}: {error_message}")
            # You might want to prioritize the error message or combine it
            response_text = f"An error occurred: {error_message}" # Simple error reporting

        # Extract content from the last message if it's an AIMessage and no error took precedence
        if not response_text and last_message and isinstance(last_message, AIMessage):
             if isinstance(last_message.content, str):
                 response_text = last_message.content
             elif isinstance(last_message.content, list): # Handle potential multi-content AI messages if needed
                 text_parts = [part["text"] for part in last_message.content if isinstance(part, dict) and part.get("type") == "text"]
                 response_text = "\n".join(text_parts) if text_parts else "Received a non-standard response."
                 if not text_parts:
                     logger.warning(f"Last AI message for session {session_id} has complex content but no text part: {last_message.content}")
             else:
                 logger.warning(f"Last AI message content for session {session_id} is not string or list: {type(last_message.content)}")
                 response_text = "Received an unexpected response format from the assistant."

        elif not response_text: # If no error message and no usable AI message at the end
             logger.error(f"Graph execution for session {session_id} finished but no usable response or error found. Final state messages: {all_messages}")
             # Avoid sending back the user's own message if it's the last one
             if last_message and isinstance(last_message, HumanMessage):
                  response_text = "I received your message, but I'm unable to provide a further response at this time."
             else:
                  response_text = "Chatbot failed to generate a valid response." # Generic fallback
             # Consider raising HTTPException here if this state is truly unexpected
             # raise HTTPException(status_code=500, detail="Chatbot failed to generate a valid response.")

        logger.info(f"Graph invocation successful for session {session_id}. Returning response.")
        # Ensure response_text is a string before returning
        if not isinstance(response_text, str):
             logger.error(f"Final response_text is not a string for session {session_id}: {type(response_text)}. State: {final_state}")
             response_text = "Internal error: Chatbot generated invalid response format." # Final fallback
             # Consider raising HTTPException
             # raise HTTPException(status_code=500, detail="Internal error: Chatbot generated invalid response format.")


        return ChatResponse(response=response_text, session_id=session_id)

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions cleanly
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error during graph invocation or response handling for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred.")


@app.post("/clear", response_model=ClearResponse)
async def clear_session(session_id: Optional[str] = Form(None)):
    """
    Signals that the context for a given session ID should be considered cleared.
    The actual clearing happens when the client starts sending requests
    with a NEW session_id. This endpoint primarily serves as a signal/confirmation.
    It does NOT modify the backend memory state directly.
    """
    if session_id:
        logger.info(f"Clear request received for session_id: {session_id}. Client should generate a new ID for future requests.")
        message = f"Session {session_id} context clear signaled. Please use a new session_id for the next chat message to start fresh."
        return ClearResponse(message=message, session_id=session_id)
    else:
        logger.info("Clear request received without a specific session_id. Client should generate a new ID for future requests.")
        message = "Context clear signaled. Please use a new session_id for the next chat message to start fresh."
        return ClearResponse(message=message)

# --- Main Execution Block (as before) ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    if not settings.OPENAI_API_KEY or "YOUR_DEFAULT_KEY_HERE" in settings.OPENAI_API_KEY:
         print("\n!!! WARNING: OpenAI API Key is missing or default. !!!\n")
    # Use reload=True only for development
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)