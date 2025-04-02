import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel

from langgraph_agentflow.single_step.agent_factory import create_hierarchical_agent


class MockChatModel(BaseChatModel):
    """Mock chat model for testing."""

    def _generate(self, messages, stop=None, run_id=None, **kwargs):
        return MagicMock()

    def _llm_type(self):
        return "mock_chat_model"

    @property
    def _identifying_params(self):
        return {"model_name": "mock_model"}

    def bind_tools(self, tools=None):
        return self


class TestAgentFactory(unittest.TestCase):
    """Test cases for the agent factory module."""

    def setUp(self):
        self.llm = MockChatModel()
        self.agent_configs = [
            {
                "name": "test",
                "description": "Test agent for testing",
                "tools": [],
            }
        ]

    @patch("langgraph_agentflow.single_step.agent_factory.build_agent_graph")
    @patch("langgraph_agentflow.single_step.agent_factory.visualize_graph")
    def test_create_hierarchical_agent(self, mock_visualize, mock_build_graph):
        """Test creation of hierarchical agent."""
        # Set up mocks
        mock_graph = MagicMock()
        mock_memory = MagicMock()
        mock_build_graph.return_value = (mock_graph, mock_memory)

        # Call the function
        graph, config = create_hierarchical_agent(
            self.llm, self.agent_configs, visualize=True
        )

        # Assertions
        mock_build_graph.assert_called_once()
        mock_visualize.assert_called_once_with(mock_graph)
        self.assertEqual(graph, mock_graph)
        self.assertIsInstance(config, dict)
        self.assertIn("configurable", config)
