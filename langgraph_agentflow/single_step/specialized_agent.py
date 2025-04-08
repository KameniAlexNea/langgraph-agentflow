from typing import Callable, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from loguru import logger

from langgraph_agentflow.single_step.agent_types import MessagesState


def create_specialized_agent(
    name: str,
    llm: BaseChatModel,
    tools: List = None,
) -> Callable:
    """
    Create a specialized agent function for a specific domain.

    Args:
        name: Name of the agent
        llm: Language model to use
        tools: List of tools the agent can use

    Returns:
        A function that processes messages using the agent
    """
    # Bind tools to the LLM if provided
    agent_llm = llm.bind_tools(tools=tools) if tools else llm

    def call_agent(state: MessagesState):
        messages = state["messages"]
        logger.info(f"--- Calling {name.capitalize()} Agent ---")
        
        # If there's an LLM-generated error response, add it to the messages
        if "llm_response" in state:
            logger.info("--- Adding LLM Error Response to Messages ---")
            messages = messages + [state["llm_response"]]
        
        response = agent_llm.invoke(messages)
        return {"messages": [response]}

    return call_agent
