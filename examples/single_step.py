from langchain_ollama import ChatOllama # type: ignore
from langgraph_agentflow.single_step import create_hierarchical_agent

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# Define your tools and agent capabilities
news_tools = [...]
sector_tools = [...]
ticker_tools = [...]

# Create the multi-step agent
agent = [
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
graph, config, output_stream, loop = create_hierarchical_agent(llm, agent)
output_stream("")
