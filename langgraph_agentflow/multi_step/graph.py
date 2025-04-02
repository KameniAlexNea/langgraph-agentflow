from typing import Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from langgraph_agentflow.multi_step.state import MultiStepAgentState


def create_decision_functions():
    """Create the decision functions used in the graph's conditional edges."""

    def decide_plan_or_route(state: MultiStepAgentState):
        if state.get("plan") and len(state["plan"]) > 0:
            return "execute_plan"
        elif state.get("route"):
            return "route_simple"
        else:
            logger.error("Error: Planner did not produce plan or simple route.")
            return END

    def route_to_specialist(state: MultiStepAgentState):
        route = state.get("route")
        if route:
            return route.strip().lower()
        logger.error("Error: No route found after executor.")
        return END

    def route_tools(state: MultiStepAgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            logger.info(
                f"--- Routing to Tools: {last_message.tool_calls[0]['name']} ---"
            )
            return "call_tools"
        logger.warning("--- No Tool Call Detected by Agent ---")
        return END

    def decide_continue_or_synthesize(state: MultiStepAgentState):
        if state.get("plan") is None:
            return END
        else:
            return "continue_plan"

    return {
        "decide_plan_or_route": decide_plan_or_route,
        "route_to_specialist": route_to_specialist,
        "route_tools": route_tools,
        "decide_continue_or_synthesize": decide_continue_or_synthesize,
    }


def build_multi_step_graph(
    plan_function,
    execute_function,
    process_function,
    specialized_agents: Dict[str, callable],
    tool_nodes: Dict[str, ToolNode],
):
    """Build and compile the LangGraph for multi-step agent."""

    # Create the workflow graph
    workflow = StateGraph(MultiStepAgentState)

    # Add core nodes
    workflow.add_node("planner", plan_function)
    workflow.add_node("executor", execute_function)
    workflow.add_node("step_processor", process_function)

    # Add specialized agent nodes and tool nodes
    for name, agent_fn in specialized_agents.items():
        workflow.add_node(name, agent_fn)

    for name, tool_node in tool_nodes.items():
        workflow.add_node(name, tool_node)

    # Get decision functions
    decisions = create_decision_functions()

    # Add edges
    workflow.add_edge(START, "planner")

    # Conditional edge after planner
    workflow.add_conditional_edges(
        "planner",
        decisions["decide_plan_or_route"],
        {
            "execute_plan": "executor",
            "route_simple": "executor",
            END: END,
        },
    )

    # Executor routes to specialists
    specialist_routes = {name: name for name in specialized_agents.keys()}
    specialist_routes[END] = END
    workflow.add_conditional_edges(
        "executor",
        decisions["route_to_specialist"],
        specialist_routes,
    )

    # Connect specialists to tools and then to step processor
    for agent_name in specialized_agents.keys():
        tool_name = f"{agent_name}_tools"
        if tool_name in tool_nodes:
            workflow.add_conditional_edges(
                agent_name,
                decisions["route_tools"],
                {"call_tools": tool_name, END: "step_processor"},
            )
            workflow.add_edge(tool_name, "step_processor")
        else:
            workflow.add_edge(agent_name, "step_processor")

    # Plan continuation loop
    workflow.add_conditional_edges(
        "step_processor",
        decisions["decide_continue_or_synthesize"],
        {
            "continue_plan": "executor",
            END: END,
        },
    )

    # Compile the graph
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)

    return compiled_graph
