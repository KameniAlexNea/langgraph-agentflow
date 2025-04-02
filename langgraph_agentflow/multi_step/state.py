from typing import Annotated, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class MultiStepAgentState(TypedDict):
    """State for a multi-step agent workflow."""

    messages: Annotated[list, add_messages]
    route: Optional[str]  # For specialized agent routing within a step
    plan: Optional[List[str]]  # Stores the decomposed steps of the plan
    executed_steps: Optional[List[str]]  # Stores summaries of executed steps/results
    current_step_index: int  # Tracks which step we are on
    original_query: Optional[
        str
    ]  # Stores the initial complex query for final synthesis
