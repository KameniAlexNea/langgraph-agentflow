"""
Multi-step agent example using LangGraph and Ollama
This example demonstrates how to create a multi-step agent using the LangGraph framework and the Ollama LLM. The agent is designed to handle queries related to news, sector performance, and ticker information.
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

from langgraph_agentflow.multi_step import (
    create_multi_step_agent,
    stream_multi_step_agent,
)

# Initialize LLM
llm = ChatOllama(model="llama3.3", temperature=0.7)

# Create the multi-step agent
agent = create_multi_step_agent(
    llm=llm,
    agent_tools=[
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
    ],
)

# Use the agent
if __name__ == "__main__":
    # Option 1: Get a complete response
    print("Getting a complete response:")
    query = "What's the recent performance of Apple stock and how does it relate to tech sector news?"

    print("\n\nStreaming the step-by-step execution:")
    # Option 2: Stream the steps
    for step in stream_multi_step_agent(agent, query):
        # This will print out intermediate steps and reasoning
        print("-" * 50)
        message = step["messages"][-1]
        message.pretty_print()
        print("-" * 50)
