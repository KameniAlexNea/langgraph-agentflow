import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool

from langgraph_agentflow.multi_step.agents import (
    create_agent_function,
    create_execute_step_function,
    create_plan_request_function,
    create_process_step_result_function,
)


class MockChatModel:
    """Mock chat model for testing."""

    response_content: str

    def __init__(self, response_content="mock response"):
        self.response_content = response_content

    def _generate(self, messages, stop=None, run_id=None, **kwargs):
        return MagicMock()

    def invoke(self, messages, **kwargs):
        if "plan" in messages[0].content.lower():
            return AIMessage(content="PLAN:\n1. First step\n2. Second step")
        if "current step" in messages[0].content.lower():
            return AIMessage(content="general")
        return AIMessage(content=self.response_content)

    def _llm_type(self):
        return "mock_chat_model"

    @property
    def _identifying_params(self):
        return {"model_name": "mock_model"}

    def bind_tools(self, tools=None):
        return self


class TestMultiStepAgents(unittest.TestCase):
    """Test cases for the multi-step agents module."""

    def setUp(self):
        self.llm = MockChatModel()
        self.agent_descriptions = {
            "general": "Handles general queries",
            "specialized": "Handles specialized tasks",
        }
        self.mock_tool = Tool(
            name="test_tool", func=lambda x: f"Result: {x}", description="Test tool"
        )

    def test_create_plan_request_function(self):
        """Test creating a planning function."""
        planner = create_plan_request_function(self.llm, self.agent_descriptions)
        self.assertTrue(callable(planner))

        # Test planning with a complex query
        state = {"messages": [HumanMessage(content="This is a complex query")]}
        result = planner(state)

        # Verify plan structure
        self.assertIn("plan", result)
        self.assertIsNotNone(result["plan"])
        self.assertEqual(len(result["plan"]), 2)  # Two steps from our mock
        self.assertIn("original_query", result)

    def test_create_execute_step_function(self):
        """Test creating a step execution function."""
        executor = create_execute_step_function(self.llm, self.agent_descriptions)
        self.assertTrue(callable(executor))

        # Test executing a step
        state = {"plan": ["Step 1", "Step 2"], "current_step_index": 0}
        result = executor(state)

        # Verify execution result
        self.assertIn("route", result)
        self.assertEqual(
            result["route"], "plan:\n1. first step\n2. second step"
        )  # From our mock
        self.assertIn("messages", result)

    def test_create_process_step_result_function(self):
        """Test creating a step result processing function."""
        processor = create_process_step_result_function(self.llm)
        self.assertTrue(callable(processor))

        # Test processing a step result - not the last step
        state = {
            "messages": [AIMessage(content="Step result")],
            "current_step_index": 0,
            "plan": ["Step 1", "Step 2"],
            "executed_steps": [],
        }
        result = processor(state)

        # Verify it moves to the next step
        self.assertEqual(result["current_step_index"], 1)
        self.assertEqual(len(result["executed_steps"]), 1)

        # Test processing the last step
        state = {
            "messages": [AIMessage(content="Final step result")],
            "current_step_index": 1,
            "plan": ["Step 1", "Step 2"],
            "executed_steps": ["Step 1 result"],
            "original_query": "Original query",
        }
        result = processor(state)

        # Verify it synthesizes a final answer
        self.assertIn("messages", result)
        self.assertEqual(result["plan"], None)  # Plan is cleared when complete

    def test_create_agent_function(self):
        """Test creating a specialized agent function."""
        agent = create_agent_function(self.llm, [self.mock_tool], "test")
        self.assertTrue(callable(agent))

        # Test the agent's response
        state = {"messages": [HumanMessage(content="Test query")]}
        result = agent(state)

        # Verify agent response
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].content, "mock response")
