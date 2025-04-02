from typing import Any, Dict, Generator

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger


def stream_agent_responses(
    graph: CompiledStateGraph, user_input: str, config: Dict = None
) -> Generator[Any, None, None]:
    """Stream responses from the agent graph."""
    if config is None:
        config = {"configurable": {"thread_id": "user-thread-1"}}

    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in graph.stream(inputs, config, stream_mode="values"):
            yield event
    except Exception as e:
        logger.error(f"\nAn error occurred during graph execution: {e}")
