from typing import Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from langgraph_agentflow.single_step.agent_types import MessagesState


def create_tool_checker(
    llm: Optional[BaseChatModel] = None, retry_handler: Optional[Callable] = None
) -> Callable:
    """
    Create a tool checker node that verifies tool outputs.

    Args:
        llm: Language model to generate helpful error messages
        retry_handler: Optional custom handler to manage tool failures

    Returns:
        A function that checks tool execution results
    """

    def check_tool_outputs(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]

        # Check if we have tool outputs to verify
        if not hasattr(last_message, "tool_outputs") or not last_message.tool_outputs:
            logger.warning("--- No Tool Outputs Found to Check ---")
            return {"has_error": False}

        logger.info("--- Checking Tool Outputs ---")

        # Look for errors in tool outputs
        errors = []
        for i, output in enumerate(last_message.tool_outputs):
            if isinstance(output, dict) and output.get("error"):
                errors.append(f"Tool #{i+1}: {output.get('error')}")
            elif output is None or (isinstance(output, str) and not output.strip()):
                errors.append(f"Tool #{i+1}: Empty or null output")

        if errors:
            error_message = "\n".join(errors)
            logger.error(f"Tool execution errors detected:\n{error_message}")

            # If we have an LLM, use it to generate a helpful error message
            if llm:
                error_response = generate_llm_error_response(llm, state, errors)
                logger.info(f"Generated LLM error response: {error_response}")
                return {
                    "has_error": True,
                    "error_message": error_message,
                    "llm_response": error_response,
                }

            # If we have a retry handler, use it
            if retry_handler:
                logger.info("--- Attempting Tool Execution Recovery ---")
                return retry_handler(state, errors)

            return {"has_error": True, "error_message": error_message}

        logger.info("--- Tool Outputs Verified Successfully ---")
        return {"has_error": False}

    return check_tool_outputs


def generate_llm_error_response(
    llm: BaseChatModel, state: MessagesState, errors: List[str]
) -> HumanMessage:
    """
    Generate a helpful error message using the LLM.

    Args:
        llm: Language model to use for error analysis
        state: Current state with messages
        errors: List of error messages from tools

    Returns:
        A HumanMessage with helpful error context
    """
    original_message = (
        state["messages"][-2].content
        if len(state["messages"]) > 1
        else "Unknown request"
    )
    tool_calls = (
        state["messages"][-1].tool_calls
        if hasattr(state["messages"][-1], "tool_calls")
        else []
    )

    tools_used = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tools_used.append(
                f"Tool: {tool_call.get('name', 'Unknown')} with arguments: {tool_call.get('args', 'Unknown')}"
            )

    tools_info = (
        "\n".join(tools_used) if tools_used else "No tool information available"
    )
    errors_text = "\n".join(errors)

    prompt = SystemMessage(
        content=f"""
    You are an AI assistant tasked with helping the user understand and recover from tool execution errors.
    Analyze the original request, the tools that were called, and the errors that occurred.
    Provide a human-friendly explanation of what went wrong and suggest how to proceed.
    Keep your response concise but helpful.
    """
    )

    user_message = HumanMessage(
        content=f"""
    Original request: {original_message}
    
    Tools attempted:
    {tools_info}
    
    Errors encountered:
    {errors_text}
    
    Please provide a helpful message explaining what went wrong and how to proceed.
    """
    )

    response = llm.invoke([prompt, user_message])
    return {"llm_response": response.content.strip()}


def default_retry_decision(state: MessagesState, errors: List[str]) -> Dict:
    """
    Default retry decision logic when tool execution fails.

    Args:
        state: Current state with messages
        errors: List of error messages from tools

    Returns:
        Decision dict with has_error flag and retry info
    """
    return {
        "has_error": True,
        "error_message": "\n".join(errors),
        "should_retry": False,  # Default is not to retry
    }
