import unittest
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool

from langgraph_agentflow.single_step.specialized_agent import create_specialized_agent


class MockChatModel(BaseChatModel):
    """Mock chat model for testing."""

    def _generate(self, messages, stop=None, run_id=None, **kwargs):
        return MagicMock()

    def invoke(self, messages, **kwargs):
        return AIMessage(content="This is a response from the specialized agent")

    def _llm_type(self):
        return "mock_chat_model"

    @property
    def _identifying_params(self):
        return {"model_name": "mock_model"}

    def bind_tools(self, tools=None):
        return self


class TestSpecializedAgent(unittest.TestCase):
    """Test cases for the specialized agent module."""

    def setUp(self):
        self.llm = MockChatModel()
        self.mock_tool = Tool(
            name="test_tool", func=lambda x: f"Result: {x}", description="Test tool"
        )

    def test_create_specialized_agent(self):
        """Test creating a specialized agent."""
        agent_func = create_specialized_agent("test", self.llm, [self.mock_tool])
        self.assertTrue(callable(agent_func))

    def test_specialized_agent_execution(self):
        """Test that the specialized agent correctly processes messages."""
        agent_func = create_specialized_agent("test", self.llm)

        # Create a test state with a message
        state = {"messages": [HumanMessage(content="What is the capital of France?")]}

        # Get the agent's response
        result = agent_func(state)

        # Assert the structure of the result
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(
            result["messages"][0].content,
            "This is a response from the specialized agent",
        )
