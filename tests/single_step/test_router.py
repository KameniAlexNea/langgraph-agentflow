import unittest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langgraph_agentflow.single_step.router import create_router_agent


class MockChatModel(BaseChatModel):
    """Mock chat model for testing."""

    def _generate(self, messages, stop=None, run_id=None, **kwargs):
        # Always return "general" as the routing decision
        return AIMessage(content="general")

    def invoke(self, messages, stop=None, run_id=None, **kwargs):
        # Always return "general" as the routing decision
        return AIMessage(content="general")

    def _llm_type(self):
        return "mock_chat_model"

    @property
    def _identifying_params(self):
        return {"model_name": "mock_model"}


class TestRouter(unittest.TestCase):
    """Test cases for the router module."""

    def setUp(self):
        self.llm = MockChatModel()
        self.agent_descriptions = {
            "general": "Handles general queries",
            "specialized": "Handles specialized tasks",
        }

    def test_create_router_agent(self):
        """Test creating a router agent."""
        router_func = create_router_agent(self.llm, self.agent_descriptions)
        self.assertTrue(callable(router_func))

    def test_router_agent_routing(self):
        """Test that the router agent correctly routes messages."""
        router_func = create_router_agent(self.llm, self.agent_descriptions)

        # Create a test state with a message
        state = {"messages": [AIMessage(content="Hello, how can you help me?")]}

        # Get the routing result
        result = router_func(state)

        # Assert the structure of the result
        self.assertIn("messages", result)
        self.assertIn("route", result)
        self.assertEqual(result["route"], "general")
        self.assertEqual(result["messages"], [])
