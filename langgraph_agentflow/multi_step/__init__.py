from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_agentflow.multi_step.agents import (
    create_agent_function,
    create_execute_step_function,
    create_plan_request_function,
    create_process_step_result_function,
)
from langgraph_agentflow.multi_step.graph import build_multi_step_graph
from langgraph_agentflow.multi_step.prompts import (
    DEFAULT_EXECUTOR_ROUTER_PROMPT,
    DEFAULT_PLANNER_PROMPT,
    DEFAULT_ROUTER_PROMPT,
    DEFAULT_SYNTHESIS_PROMPT,
)


def create_multi_step_agent(
    llm: BaseChatModel,
    agent_tools: List[Dict],
    custom_prompts: Optional[Dict[str, str]] = None,
):
    """
    Create a multi-step agent that can handle complex tasks by breaking them down.

    Args:
        llm: The language model to use for all agents
        agent_tools: List of agent definitions, each containing:
            - name: Name of the agent
            - tools: List of tools the agent can use (optional)
            - description: Description of the agent's capabilities
        custom_prompts: Optional dictionary of custom prompts to override defaults
            - planner: Custom planner prompt
            - executor: Custom executor prompt
            - router: Custom router prompt
            - synthesis: Custom synthesis prompt

    Returns:
        A compiled LangGraph that can be invoked with user queries
    """

    # Initialize custom prompts
    prompts = {
        "planner": DEFAULT_PLANNER_PROMPT,
        "executor": DEFAULT_EXECUTOR_ROUTER_PROMPT,
        "router": DEFAULT_ROUTER_PROMPT,
        "synthesis": DEFAULT_SYNTHESIS_PROMPT,
    }

    if custom_prompts:
        prompts.update(custom_prompts)

    # Extract agent descriptions and create agent functions
    agent_descriptions = {}
    specialized_agents = {}
    tool_nodes = {}

    for agent_def in agent_tools:
        name = agent_def["name"]
        desc = agent_def.get("description", f"{name.capitalize()} agent")
        tools = agent_def.get("tools", [])

        agent_descriptions[name] = desc
        specialized_agents[name] = create_agent_function(llm, tools, name)

        if tools:
            tool_nodes[f"{name}_tools"] = ToolNode(tools)

    # Create core agent functions
    plan_fn = create_plan_request_function(llm, agent_descriptions, prompts["planner"])
    execute_fn = create_execute_step_function(
        llm, agent_descriptions, prompts["executor"]
    )
    process_fn = create_process_step_result_function(llm, prompts["synthesis"])

    # Build and compile the graph
    graph = build_multi_step_graph(
        plan_fn,
        execute_fn,
        process_fn,
        specialized_agents,
        tool_nodes,
    )

    return graph


def invoke_multi_step_agent(
    graph: CompiledStateGraph, query: str, config: Optional[Dict] = None
):
    """
    Invoke a multi-step agent with a user query.

    Args:
        graph: Compiled multi-step agent graph
        query: User query as a string
        config: Optional configuration parameters

    Returns:
        Final response from the agent
    """
    if config is None:
        config = {"configurable": {"thread_id": "user-thread-1"}}

    inputs = {"messages": [HumanMessage(content=query)]}
    return graph.invoke(inputs, config)


def stream_multi_step_agent(graph, query: str, config: Optional[Dict] = None):
    """
    Stream updates from a multi-step agent with a user query.

    Args:
        graph: Compiled multi-step agent graph
        query: User query as a string
        config: Optional configuration parameters

    Yields:
        Updates from the agent execution
    """
    if config is None:
        config = {"configurable": {"thread_id": "user-thread-1"}}

    inputs = {"messages": [HumanMessage(content=query)]}
    for event in graph.stream(inputs, config, stream_mode="values"):
        yield event
