from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from loguru import logger

from langgraph_agentflow.multi_step.prompts import (
    DEFAULT_EXECUTOR_ROUTER_PROMPT,
    DEFAULT_PLANNER_PROMPT,
    DEFAULT_ROUTER_PROMPT,
    DEFAULT_SYNTHESIS_PROMPT,
)
from langgraph_agentflow.multi_step.state import MultiStepAgentState


def create_plan_request_function(
    llm: BaseChatModel,
    agent_descriptions: Dict[str, str],
    planner_prompt: str = DEFAULT_PLANNER_PROMPT,
):
    """Creates a planning function that decides whether to route directly or create a multi-step plan."""

    def plan_request(state: MultiStepAgentState):
        logger.info("--- Calling Planner Agent ---")
        query = state["messages"][-1].content

        # Format agent descriptions for prompt
        agent_desc_text = "\n".join(
            [f"- '{name}': {desc}" for name, desc in agent_descriptions.items()]
        )

        prompt = planner_prompt.format(
            query=query,
            agent_descriptions=agent_desc_text,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"Planner Output:\n{response.content}")

        if response.content.startswith("SIMPLE:"):
            route = response.content.split(":", 1)[1].strip().lower()
            return {"route": route, "plan": None, "original_query": query}
        elif response.content.startswith("PLAN:"):
            plan_str = response.content.split("PLAN:", 1)[1].strip()
            plan_steps = [
                step.strip()
                for step in plan_str.split("\n")
                if step.strip() and step.strip()[0].isdigit()
            ]
            return {
                "plan": plan_steps,
                "current_step_index": 0,
                "executed_steps": [],
                "route": None,
                "original_query": query,
            }
        else:
            # Fallback if planner output is malformed
            logger.error("Planner failed to provide valid output, routing to general.")
            return {"route": "general", "plan": None, "original_query": query}

    return plan_request


def create_execute_step_function(
    llm: BaseChatModel,
    agent_descriptions: Dict[str, str],
    executor_prompt: str = DEFAULT_EXECUTOR_ROUTER_PROMPT,
):
    """Creates a function that executes a single step in the plan."""

    def execute_step(state: MultiStepAgentState):
        logger.info("--- Calling Executor Agent ---")
        plan = state["plan"]
        step_index = state["current_step_index"]

        if plan is None or step_index >= len(plan):
            logger.warning(
                "Error: Execute step called with no plan or index out of bounds."
            )
            return {"route": "general"}

        current_step_description = plan[step_index]
        logger.info(f"Executing Step {step_index + 1}: {current_step_description}")

        # Format agent descriptions for prompt
        agent_desc_text = "\n".join(
            [f"- '{name}': {desc}" for name, desc in agent_descriptions.items()]
        )

        prompt = executor_prompt.format(
            step_description=current_step_description,
            agent_descriptions=agent_desc_text,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        route = response.content.strip().lower()
        logger.info(
            f"Executor Routing Decision: {route} for step '{current_step_description}'"
        )

        step_message = HumanMessage(
            content=f"Focus on this task: {current_step_description}"
        )

        return {
            "route": route,
            "messages": [step_message],
        }

    return execute_step


def create_process_step_result_function(
    llm: BaseChatModel, synthesis_prompt: str = DEFAULT_SYNTHESIS_PROMPT
):
    """Creates a function that processes the result of a step and decides next actions."""

    def process_step_result(state: MultiStepAgentState):
        logger.info("--- Processing Step Result ---")
        last_message = state["messages"][-1]
        step_index = state["current_step_index"]
        plan = state["plan"]
        executed_steps = state["executed_steps"] or []

        step_summary = (
            f"Step {step_index + 1} ({plan[step_index]}): {last_message.content}"
        )
        executed_steps.append(step_summary)

        next_step_index = step_index + 1

        if next_step_index < len(plan):
            logger.info(
                f"Finished step {step_index + 1}. Moving to step {next_step_index + 1}."
            )
            return {
                "executed_steps": executed_steps,
                "current_step_index": next_step_index,
                "route": None,
            }
        else:
            logger.info("Plan finished. Synthesizing final answer.")
            summary_str = "\n".join(executed_steps)
            synthesis_prompt_filled = synthesis_prompt.format(
                original_query=state["original_query"],
                executed_steps_summary=summary_str,
            )

            final_response = llm.invoke([HumanMessage(content=synthesis_prompt_filled)])
            logger.info("--- Final Synthesized Response ---")
            return {
                "executed_steps": executed_steps,
                "messages": [final_response],
                "plan": None,
            }

    return process_step_result


def create_route_request_function(
    llm: BaseChatModel,
    agent_descriptions: Dict[str, str],
    router_prompt: str = DEFAULT_ROUTER_PROMPT,
):
    """Creates a routing function that decides which specialist agent to call."""

    def route_request(state: MultiStepAgentState):
        messages = state["messages"]
        user_query = messages[-1].content

        # Format agent descriptions for prompt
        agent_desc_text = "\n".join(
            [f"- '{name}': {desc}" for name, desc in agent_descriptions.items()]
        )

        router_messages = [
            HumanMessage(
                content=router_prompt.format(
                    query=user_query,
                    agent_descriptions=agent_desc_text,
                )
            ),
        ]

        logger.info("--- Calling Router Agent ---")
        response = llm.invoke(router_messages)
        logger.info(f"Router Decision: {response.content}")

        return {
            "messages": [],
            "route": response.content.strip().lower(),
        }

    return route_request


def create_agent_function(
    llm: BaseChatModel, tools: Optional[List[BaseTool]] = None, name: str = "generic"
):
    """Creates a specialized agent function with optional tools."""

    # Bind tools to LLM if provided
    agent_llm = llm.bind_tools(tools) if tools else llm

    def call_agent(state: MultiStepAgentState):
        messages = state["messages"]
        logger.info(f"--- Calling {name.capitalize()} Agent ---")
        response = agent_llm.invoke(messages)
        return {"messages": [response]}

    return call_agent
