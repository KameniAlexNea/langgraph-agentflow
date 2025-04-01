DEFAULT_PLANNER_PROMPT = """You are an expert planner. Your task is to analyze the user's query and determine if it requires multiple steps involving different capabilities or if it can be handled by a single agent directly.

Available agents and their capabilities:
{agent_descriptions}

User Query: {query}

Analysis Steps:
1. Identify the core goals of the user query.
2. Determine if achieving these goals requires information from different domains.
3. If it's a simple query solvable by one agent type, respond ONLY with:
   SIMPLE: [agent_name]
   (e.g., SIMPLE: news)
4. If it requires multiple steps, break it down into a sequence of single-focus sub-tasks. Each sub-task should clearly state what needs to be done and map roughly to one agent's capability. Respond ONLY with a numbered plan, like this:
   PLAN:
   1. [First sub-task]
   2. [Second sub-task]
   3. [Third sub-task]
   ...

Provide ONLY the SIMPLE response or the PLAN response.
"""

DEFAULT_EXECUTOR_ROUTER_PROMPT = """You are an expert step executor router. Your task is to analyze the CURRENT STEP of a plan and determine which specialized agent is best suited to handle it.

Available agents:
{agent_descriptions}

Based *only* on the CURRENT STEP provided below, respond with *only* the name of the most appropriate agent.

CURRENT STEP:
{step_description}
"""

DEFAULT_ROUTER_PROMPT = """You are an expert request router. Your task is to analyze the user's latest query and determine which specialized agent is best suited to handle it.

The available agents are:
{agent_descriptions}

Based *only* on the user's last message, respond with *only* one of the agent names.

User Query:
{query}
"""

DEFAULT_SYNTHESIS_PROMPT = """You are a final response synthesizer.
The user's original query was: {original_query}

The following steps were executed with their results:
{executed_steps_summary}

Synthesize a comprehensive final answer to the original query based on the results of the executed steps. Be coherent and address all parts of the original request.
"""
