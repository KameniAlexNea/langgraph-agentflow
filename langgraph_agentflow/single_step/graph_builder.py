from typing import Any, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from langgraph_agentflow.single_step.agent_types import (
    DEFAULT_ROUTER_PROMPT,
    MessagesState,
)
from langgraph_agentflow.single_step.router import create_router_agent
from langgraph_agentflow.single_step.specialized_agent import create_specialized_agent
from langgraph_agentflow.single_step.tool_checker import create_tool_checker, default_retry_decision


def build_agent_graph(
    llm: BaseChatModel,
    agent_configs: List[Dict[str, Any]],
    router_prompt: str = DEFAULT_ROUTER_PROMPT,
    tool_checker_enabled: bool = True,
) -> Tuple[CompiledStateGraph, MemorySaver]:
    """
    Build a complete agent graph with router and specialized agents.

    Args:
        llm: Base language model
        agent_configs: List of agent configuration dictionaries. Each must contain 'name' and 'description',
                      and may optionally contain 'tools'.
        router_prompt: Custom router prompt template
        tool_checker_enabled: Whether to enable tool execution validation

    Returns:
        Tuple of (compiled_graph, memory)
    """
    # Extract agent descriptions for the router
    agent_descriptions = {
        config["name"]: config.get(
            "description", f"Handles {config['name']}-related queries"
        )
        for config in agent_configs
    }

    # Create the router agent
    router_agent = create_router_agent(llm, agent_descriptions, router_prompt)

    # Create specialized agents and tool nodes
    specialized_agents = {}
    tool_nodes = {}
    tool_checker_nodes = {}

    for config in agent_configs:
        name = config["name"]
        tools = config.get("tools", [])
        specialized_agents[name] = create_specialized_agent(name, llm, tools)
        if tools:
            tool_nodes[f"{name}_tools"] = ToolNode(tools)
            if tool_checker_enabled:
                tool_checker_nodes[f"{name}_tool_checker"] = create_tool_checker(default_retry_decision)

    # Build the graph
    workflow = StateGraph(MessagesState)

    # Add the router node
    workflow.add_node("router", router_agent)

    # Add specialized agent nodes
    for name, agent_func in specialized_agents.items():
        workflow.add_node(f"{name}_agent", agent_func)

    # Add tool nodes
    for name, tool_node in tool_nodes.items():
        workflow.add_node(name, tool_node)
    
    # Add tool checker nodes if enabled
    if tool_checker_enabled:
        for name, checker_node in tool_checker_nodes.items():
            workflow.add_node(name, checker_node)

    # Define entry point
    workflow.add_edge(START, "router")

    # Router decision logic
    def decide_next_node(state: MessagesState):
        route = state["route"]
        if route is not None:
            for config in agent_configs:
                name = config["name"]
                if name in route:
                    return f"{name}_agent"
            # Default to the first agent if no match
            return f"{agent_configs[0]['name']}_agent"
        return END

    # Add conditional edges from router to agents
    conditional_targets = {
        f"{config['name']}_agent": f"{config['name']}_agent" for config in agent_configs
    }
    conditional_targets[END] = END
    workflow.add_conditional_edges("router", decide_next_node, conditional_targets)

    # Tool routing logic
    def route_tools(state: MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            logger.info(
                f"--- Routing to Tools: {last_message.tool_calls[0]['name']} ---"
            )
            return "call_tools"
        logger.warning("--- No Tool Call Detected by Agent ---")
        return END

    # Tool checker decision logic
    def tool_checker_decision(state: MessagesState) -> str:
        if state.get("has_error", False):
            logger.warning("--- Tool Execution Error Detected ---")
            # Could implement retry logic here in the future
            return "back_to_agent"
        logger.info("--- Tool Execution Successful ---")
        return "back_to_agent"

    # Connect agents to their tools with tool checker
    for config in agent_configs:
        name = config["name"]
        if f"{name}_tools" in tool_nodes:
            # Connect agent to tools
            workflow.add_conditional_edges(
                f"{name}_agent", route_tools, {"call_tools": f"{name}_tools", END: END}
            )
            
            if tool_checker_enabled:
                # Connect tools to tool checker
                workflow.add_edge(f"{name}_tools", f"{name}_tool_checker")
                
                # Connect tool checker back to agent
                workflow.add_conditional_edges(
                    f"{name}_tool_checker", 
                    tool_checker_decision,
                    {"back_to_agent": f"{name}_agent"}
                )
            else:
                # Direct connection without checker
                workflow.add_edge(f"{name}_tools", f"{name}_agent")
        else:
            workflow.add_edge(f"{name}_agent", END)

    # Compile the graph
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph, memory