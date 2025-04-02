from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from langgraph_agentflow.single_step.agent_types import DEFAULT_ROUTER_PROMPT
from langgraph_agentflow.single_step.graph_builder import build_agent_graph
from langgraph_agentflow.single_step.visualization import visualize_graph


def create_hierarchical_agent(
    llm: BaseChatModel,
    agent_configs: Optional[List[Dict[str, Any]]] = None,
    router_prompt: str = DEFAULT_ROUTER_PROMPT,
    visualize: bool = True,
) -> Tuple[CompiledStateGraph, Dict]:
    """
    Create a complete hierarchical agent system with one function call.

    Args:
        llm: Language model to use (will create default if None)
        agent_configs: List of configuration dictionaries for each specialized agent
        router_prompt: Custom router prompt
        visualize: Whether to visualize the graph

    Returns:
        Tuple of (graph, config)
    """
    # Create default agent configs if not provided
    if agent_configs is None:
        agent_configs = [
            {
                "name": "general",
                "description": "Handles general conversation and queries not fitting other categories.",
                "tools": [],
            }
        ]

    # Build the graph
    graph, _ = build_agent_graph(llm, agent_configs, router_prompt)

    # Visualize if requested
    if visualize:
        visualize_graph(graph)

    # Create default config
    config = {"configurable": {"thread_id": "user-thread-1"}}

    # Return all the components needed for interaction
    return (graph, config)
