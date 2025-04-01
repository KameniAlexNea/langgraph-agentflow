"""
AgentFlow is a Python library that automates the orchestration of 
multi-step agent workflows by integrating intelligent planning, 
routing, and execution of specialized operations.
"""

__version__ = "0.1.0"

# Import key components for easier access
from langgraph_agentflow.single_step import (
    create_hierarchical_agent,
    build_agent_graph,
    run_interactive_loop,
    stream_agent_responses,
    visualize_graph,
)

from langgraph_agentflow.multi_step import (
    create_multi_step_agent,
    invoke_multi_step_agent,
    stream_multi_step_agent,
)

__all__ = [
    "create_hierarchical_agent",
    "build_agent_graph",
    "run_interactive_loop",
    "stream_agent_responses",
    "visualize_graph",
    "create_multi_step_agent",
    "invoke_multi_step_agent",
    "stream_multi_step_agent",
]
