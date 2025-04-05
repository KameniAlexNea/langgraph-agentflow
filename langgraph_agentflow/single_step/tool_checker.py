from typing import Callable, Dict, List, Optional

from loguru import logger

from langgraph_agentflow.single_step.agent_types import MessagesState


def create_tool_checker(retry_handler: Optional[Callable] = None) -> Callable:
    """
    Create a tool checker node that verifies tool outputs.

    Args:
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
            
            # If we have a retry handler, use it
            if retry_handler:
                logger.info("--- Attempting Tool Execution Recovery ---")
                return retry_handler(state, errors)
                
            return {"has_error": True, "error_message": error_message}
        
        logger.info("--- Tool Outputs Verified Successfully ---")
        return {"has_error": False}
    
    return check_tool_outputs


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
        "should_retry": False  # Default is not to retry
    }