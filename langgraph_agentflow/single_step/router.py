from typing import Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from loguru import logger

from langgraph_agentflow.single_step.agent_types import (
    DEFAULT_ROUTER_PROMPT,
    MessagesState,
)


def create_router_agent(
    llm: BaseChatModel,
    agent_descriptions: Dict[str, str],
    router_prompt: str = DEFAULT_ROUTER_PROMPT,
) -> Callable:
    """
    Create a router agent function that decides which specialist agent to call.

    Args:
        llm: Language model to use for routing
        agent_descriptions: Dictionary mapping agent names to their descriptions
        router_prompt: Template for the router prompt

    Returns:
        A function that routes messages to the appropriate agent
    """
    # Format the agent descriptions for the prompt
    descriptions_text = "\n".join(
        [f"- '{name}_agent': {desc}" for name, desc in agent_descriptions.items()]
    )
    agent_keywords = ", ".join([f"'{name}'" for name in agent_descriptions.keys()])

    def route_request(state: MessagesState):
        messages = state["messages"]
        user_query = messages[-1].content

        # Format the router prompt with agent info
        formatted_prompt = router_prompt.format(
            query=user_query,
            agent_descriptions=descriptions_text,
            agent_keywords=agent_keywords,
        )

        router_messages = [HumanMessage(content=formatted_prompt)]
        logger.info("--- Calling Router Agent ---")
        response = llm.invoke(router_messages)
        logger.info(f"Router Decision: {response.content}")

        return {
            "messages": [],
            "route": response.content.strip().lower(),
        }

    return route_request
