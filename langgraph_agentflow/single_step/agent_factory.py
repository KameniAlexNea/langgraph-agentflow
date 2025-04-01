from typing import Annotated, Dict, List, Callable, Optional, Any, Union
import inspect

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from loguru import logger


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str = None


# Default router prompt template
DEFAULT_ROUTER_PROMPT = """You are an expert request router. Your task is to analyze the user's latest query and determine which specialized agent is best suited to handle it.
The available agents are:
{agent_descriptions}

Based *only* on the user's last message, respond with *only* one of the following exact keywords: {agent_keywords}.

User Query:
{query}
"""


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
        response = agent_llm.invoke(messages)
        return {"messages": [response]}

    return call_agent


def build_agent_graph(
    llm: BaseChatModel,
    agent_configs: Dict[str, Dict[str, Any]],
    router_prompt: str = DEFAULT_ROUTER_PROMPT,
) -> tuple:
    """
    Build a complete agent graph with router and specialized agents.

    Args:
        llm: Base language model
        agent_configs: Dictionary mapping agent names to their configurations
                       (each containing 'description' and optional 'tools')
        router_prompt: Custom router prompt template

    Returns:
        Tuple of (compiled_graph, memory)
    """
    # Extract agent descriptions for the router
    agent_descriptions = {
        name: config.get("description", f"Handles {name}-related queries")
        for name, config in agent_configs.items()
    }

    # Create the router agent
    router_agent = create_router_agent(llm, agent_descriptions, router_prompt)

    # Create specialized agents and tool nodes
    specialized_agents = {}
    tool_nodes = {}

    for name, config in agent_configs.items():
        tools = config.get("tools", [])
        specialized_agents[name] = create_specialized_agent(name, llm, tools)
        if tools:
            tool_nodes[f"{name}_tools"] = ToolNode(tools)

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

    # Define entry point
    workflow.add_edge(START, "router")

    # Router decision logic
    def decide_next_node(state: MessagesState):
        route = state["route"]
        if route is not None:
            for name in agent_configs.keys():
                if name in route:
                    return f"{name}_agent"
            # Default to the first agent if no match
            return f"{list(agent_configs.keys())[0]}_agent"
        return END

    # Add conditional edges from router to agents
    conditional_targets = {
        f"{name}_agent": f"{name}_agent" for name in agent_configs.keys()
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

    # Connect agents to their tools
    for name in agent_configs.keys():
        if f"{name}_tools" in tool_nodes:
            workflow.add_conditional_edges(
                f"{name}_agent", route_tools, {"call_tools": f"{name}_tools", END: END}
            )
            workflow.add_edge(f"{name}_tools", f"{name}_agent")
        else:
            workflow.add_edge(f"{name}_agent", END)

    # Compile the graph
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph, memory


def visualize_graph(graph: CompiledStateGraph):
    """Try to visualize the graph if possible."""
    try:
        from IPython.display import Image, display  # type: ignore
    except ImportError:
        logger.warning("IPython is not available. Cannot display graph visualization.")
        return
    try:
        img_data = graph.get_graph().draw_png()
        display(Image(img_data))
    except Exception as e:
        logger.error(f"Graph visualization failed (requires graphviz): {e}")


def stream_agent_responses(
    graph: CompiledStateGraph, user_input: str, config: Dict = None
):
    """Stream responses from the agent graph."""
    if config is None:
        config = {"configurable": {"thread_id": "user-thread-1"}}

    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in graph.stream(inputs, config, stream_mode="values"):
            yield event
    except Exception as e:
        logger.error(f"\nAn error occurred during graph execution: {e}")


def run_interactive_loop(graph: CompiledStateGraph, config: Dict = None):
    """Run an interactive chat loop with the agent graph."""
    if config is None:
        config = {"configurable": {"thread_id": "user-thread-1"}}

    logger.info("Starting hierarchical chatbot. Type 'quit', 'exit', or 'q' to stop.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                logger.info("Goodbye!")
                break
            else:
                stream_agent_responses(graph, user_input, config)
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the input loop: {e}")
            break

    logger.info("Exiting the chatbot.")


def create_hierarchical_agent(
    llm: BaseChatModel,
    agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    router_prompt: str = DEFAULT_ROUTER_PROMPT,
    visualize: bool = True,
):
    """
    Create a complete hierarchical agent system with one function call.

    Args:
        llm: Language model to use (will create default if None)
        agent_configs: Configuration for each specialized agent
        router_prompt: Custom router prompt
        visualize: Whether to visualize the graph

    Returns:
        Tuple of (graph, config, stream_function, interactive_loop_function)
    """
    # Create default agent configs if not provided
    if agent_configs is None:
        agent_configs = {
            "general": {
                "description": "Handles general conversation and queries not fitting other categories.",
                "tools": [],
            }
        }

    # Build the graph
    graph, memory = build_agent_graph(llm, agent_configs, router_prompt)

    # Visualize if requested
    if visualize:
        visualize_graph(graph)

    # Create default config
    config = {"configurable": {"thread_id": "user-thread-1"}}

    # Return all the components needed for interaction
    return (
        graph,
        config,
        lambda user_input: stream_agent_responses(graph, user_input, config),
        lambda: run_interactive_loop(graph, config),
    )
