from langchain_ollama import ChatOllama # type: ignore
from langgraph_agentflow.multi_step import create_multi_step_agent, invoke_multi_step_agent

# Initialize LLM
llm = ChatOllama(model="llama3.3", temperature=0.7)

# Define your tools and agent capabilities
news_tools = [...]
sector_tools = [...]
ticker_tools = [...]

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
        "planner": "Your custom planner prompt...",
    }
)

# Use the agent
response = invoke_multi_step_agent(agent, "What's the recent performance of Apple stock and how does it relate to tech sector news?")