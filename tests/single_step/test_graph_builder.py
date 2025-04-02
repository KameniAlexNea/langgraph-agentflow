import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool

from langgraph_agentflow.single_step.graph_builder import build_agent_graph


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


class TestGraphBuilder(unittest.TestCase):
    """Test cases for the graph builder module."""

    def setUp(self):
        self.llm = MockChatModel()
        self.mock_tool = Tool(
            name="test_tool", func=lambda x: f"Result: {x}", description="Test tool"
        )
        self.agent_configs = [
            {
                "name": "general",
                "description": "General agent",
                "tools": [],
            },
            {
                "name": "specialized",
                "description": "Specialized agent",
                "tools": [self.mock_tool],
            },
        ]

    @patch("langgraph_agentflow.single_step.graph_builder.StateGraph")
    @patch("langgraph_agentflow.single_step.graph_builder.MemorySaver")
    @patch("langgraph_agentflow.single_step.graph_builder.ToolNode")
    def test_build_agent_graph(
        self, mock_tool_node, mock_memory_saver, mock_state_graph
    ):
        """Test building an agent graph with router and specialized agents."""
        # Set up mocks
        mock_graph_instance = MagicMock()
        mock_state_graph.return_value = mock_graph_instance
        mock_graph_instance.compile.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()

        # Call the function
        graph, memory = build_agent_graph(self.llm, self.agent_configs)

        # Assertions
        self.assertIsNotNone(graph)
        self.assertIsNotNone(memory)
        mock_graph_instance.add_node.assert_called()
        mock_graph_instance.add_edge.assert_called()
        mock_graph_instance.add_conditional_edges.assert_called()
        mock_graph_instance.compile.assert_called_once()
