"""
AgentFlow is a Python library that automates the orchestration of
multi-step agent workflows by integrating intelligent planning,
routing, and execution of specialized operations.
"""

__version__ = "0.1.0"

from langgraph_agentflow.multi_step import (
    create_multi_step_agent,
    invoke_multi_step_agent,
    stream_multi_step_agent,
)

# Import key components for easier access
from langgraph_agentflow.single_step import (
    build_agent_graph,
    create_hierarchical_agent,
    stream_agent_responses,
    visualize_graph,
)

__all__ = [
    # single step
    "create_hierarchical_agent",
    "build_agent_graph",
    "stream_agent_responses",
    # visualization
    "visualize_graph",
    # multi step
    "create_multi_step_agent",
    "invoke_multi_step_agent",
    "stream_multi_step_agent",
]
