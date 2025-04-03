"""
Single step agent example using LangGraph and Ollama LLM
This example demonstrates how to create a single-step agent using the LangGraph framework and the Ollama LLM. The agent is designed to handle queries related to news, sector performance, and ticker information.
It showcases how to build the agent graph, configure it, and interact with it to get responses.

pip install git+ssh://git@github.com/Nganga-AI/tumkwe-invest.git
"""

from langchain_ollama import ChatOllama  # type: ignore
from tumkwe_invest.news import TOOL_DESCRIPTION as NEWS_TOOL_DESCRIPTION
from tumkwe_invest.news import tools as news_tools
from tumkwe_invest.sector import TOOL_DESCRIPTION as SECTOR_TOOL_DESCRIPTION
from tumkwe_invest.sector import tools as sector_tools
from tumkwe_invest.ticker import TOOL_DESCRIPTION as TICKER_TOOL_DESCRIPTION
from tumkwe_invest.ticker import tools as ticker_tools

from langgraph_agentflow.single_step import build_agent_graph, stream_agent_responses

# Initialize LLM
llm = ChatOllama(model="llama3.3", temperature=0.7)

# Create the agent configuration
agent_config = [
    {
        "name": "news",
        "tools": news_tools,
        "description": NEWS_TOOL_DESCRIPTION,
    },
    {
        "name": "sector",
        "tools": sector_tools,
        "description": SECTOR_TOOL_DESCRIPTION,
    },
    {
        "name": "ticker",
        "tools": ticker_tools,
        "description": TICKER_TOOL_DESCRIPTION,
    },
    {
        "name": "general",
        "description": "Handles general information and queries not specific to other domains",
    },
]

# Use the agent
graph, _ = build_agent_graph(llm, agent_config)
config = {"configurable": {"thread_id": "user-thread-1"}}

# Example usage
if __name__ == "__main__":
    # Option 1: Stream a single response
    print("Getting a complete response:")
    query = "What's the recent performance of Apple stock and how does it relate to tech sector news?"
    for step in stream_agent_responses(graph, query, config):
        print("-" * 50)
        message = step["messages"][-1]
        message.pretty_print()
        print("-" * 50)

    print("\n\nStarting interactive loop (press Ctrl+C to exit):")
