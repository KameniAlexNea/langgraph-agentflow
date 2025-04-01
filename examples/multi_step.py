from langchain_core.tools import Tool
from langchain_ollama import ChatOllama  # type: ignore
from langgraph_agentflow.multi_step import create_multi_step_agent, invoke_multi_step_agent, stream_multi_step_agent

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

# Create the multi-step agent
agent = create_multi_step_agent(
    llm=llm,
    agent_tools=[
        {
            "name": "news",
            "tools": news_tools,
            "description": "Retrieves and analyzes news about companies and markets"
        },
        {
            "name": "sector",
            "tools": sector_tools,
            "description": "Analyzes sector performance and trends"
        },
        {
            "name": "ticker",
            "tools": ticker_tools,
            "description": "Retrieves and analyzes stock ticker information"
        },
        {
            "name": "general",
            "description": "Handles general information and queries not specific to other domains"
        }
    ],
    custom_prompts={
        # Optional: override default prompts
        "planner": """You are an expert planner specialized in financial analysis tasks.
Your task is to analyze the user's query and determine if it requires multiple steps.
Available agents: {agent_descriptions}
User Query: {query}
If it's a simple query, respond with: SIMPLE: [agent_name]
If it requires multiple steps, respond with: PLAN: followed by numbered steps.
"""
    }
)

# Use the agent
if __name__ == "__main__":
    # Option 1: Get a complete response
    print("Getting a complete response:")
    query = "What's the recent performance of Apple stock and how does it relate to tech sector news?"
    response = invoke_multi_step_agent(agent, query)
    print("Final response:", response)
    
    print("\n\nStreaming the step-by-step execution:")
    # Option 2: Stream the steps
    for step in stream_multi_step_agent(agent, query):
        # This will print out intermediate steps and reasoning
        print("-" * 50)
        message = step["messages"][-1]
        message.pretty_print()
        print("-" * 50)