from langchain_core.tools import Tool
from langchain_ollama import ChatOllama  # type: ignore
from langgraph_agentflow.single_step import create_hierarchical_agent

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# Define your tools (simple examples)
news_tools = [
    Tool(
        name="search_news",
        description="Search for recent news about a company or topic",
        func=lambda x: f"News results for {x}: Latest financial news...",
    )
]

sector_tools = [
    Tool(
        name="sector_analysis",
        description="Get analysis for a market sector",
        func=lambda x: f"Sector analysis for {x}: The sector is performing well...",
    )
]

ticker_tools = [
    Tool(
        name="get_stock_price",
        description="Get the current stock price for a ticker symbol",
        func=lambda x: f"Stock price for {x}: $150.42, up 1.2% today",
    )
]

# Create the agent configuration
agent_config = [
    {
        "name": "news",
        "tools": news_tools,
        "description": "Retrieves and analyzes news about companies and markets",
    },
    {
        "name": "sector",
        "tools": sector_tools,
        "description": "Analyzes sector performance and trends",
    },
    {
        "name": "ticker",
        "tools": ticker_tools,
        "description": "Retrieves and analyzes stock ticker information",
    },
    {
        "name": "general",
        "description": "Handles general information and queries not specific to other domains",
    },
]

# Use the agent
graph, config, output_stream, loop = create_hierarchical_agent(llm, agent_config)

# Example usage
if __name__ == "__main__":
    # Option 1: Stream a single response
    print("Streaming a single response:")
    for step in loop():
        print("-" * 50)
        message = step["messages"][-1]
        message.pretty_print()
        print("-" * 50)
    
    print("\n\nStarting interactive loop (press Ctrl+C to exit):")
    # Option 2: Run an interactive loop
    try:
        loop()
    except KeyboardInterrupt:
        print("\nExited by user.")
