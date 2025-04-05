from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str] = None
    has_error: bool = False
    error_message: Optional[str] = None
    should_retry: bool = False


# Default router prompt template
DEFAULT_ROUTER_PROMPT = """You are an expert request router. Your task is to analyze the user's latest query and determine which specialized agent is best suited to handle it.
The available agents are:
{agent_descriptions}

Based *only* on the user's last message, respond with *only* one of the following exact keywords: {agent_keywords}.

User Query:
{query}
"""
