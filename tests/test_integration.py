import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool

from langgraph_agentflow import (
    create_hierarchical_agent,
    create_multi_step_agent,
    invoke_multi_step_agent,
)


class MockChatModel(BaseChatModel):
    """Mock chat model for testing."""

    def _generate(self, messages, stop=None, run_id=None, **kwargs):
        return MagicMock()

    def invoke(self, messages, **kwargs):
        # Different responses based on context
        if any("plan" in str(m.content).lower() for m in messages):
            return AIMessage(content="PLAN:\n1. First step\n2. Second step")
        if any("current step" in str(m.content).lower() for m in messages):
            return AIMessage(content="general")
        return AIMessage(content="This is a response from the mock model")

    def _llm_type(self):
        return "mock_chat_model"

    @property
    def _identifying_params(self):
        return {"model_name": "mock_model"}

    def bind_tools(self, tools=None):
        return self


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        self.llm = MockChatModel()
        self.mock_tool = Tool(
            name="test_tool", func=lambda x: f"Result: {x}", description="Test tool"
        )
        self.agent_configs = [
            {
                "name": "general",
                "description": "Handles general queries",
                "tools": [],
            },
            {
                "name": "specialized",
                "description": "Handles specialized tasks",
                "tools": [self.mock_tool],
            },
        ]

    @patch("langgraph_agentflow.single_step.agent_factory.visualize_graph")
    def test_hierarchical_agent_creation(self, mock_visualize):
        """Test creating a hierarchical agent."""
        # Disable visualization for testing
        mock_visualize.return_value = None

        # Create the agent
        graph, config = create_hierarchical_agent(
            self.llm, self.agent_configs, visualize=False
        )

        # Verify the agent was created
        self.assertIsNotNone(graph)
        self.assertIsInstance(config, dict)

    @patch("langgraph_agentflow.multi_step.build_multi_step_graph")
    def test_multi_step_agent_creation(self, mock_build_graph):
        """Test creating a multi-step agent."""
        # Mock the build graph function
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        # Create the agent
        agent = create_multi_step_agent(llm=self.llm, agent_tools=self.agent_configs)

        # Verify the agent was created
        self.assertEqual(agent, mock_graph)
        mock_build_graph.assert_called_once()

    @patch("langgraph_agentflow.multi_step.graph.StateGraph")
    @patch("langgraph_agentflow.multi_step.invoke_multi_step_agent")
    def test_multi_step_agent_invocation(self, mock_invoke, mock_state_graph):
        """Test invoking a multi-step agent."""
        # Mock response
        mock_response = {"messages": [AIMessage(content="Final response")]}
        mock_invoke.return_value = mock_response

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = (
            mock_invoke  # lambda *args, **kwargs: print("Mocked invoke call")
        )
        mock_state_graph.return_value.compile.return_value = mock_graph

        agent = create_multi_step_agent(llm=self.llm, agent_tools=self.agent_configs)

        result = invoke_multi_step_agent(agent, "Test query")

        print("mock_invoke call count:", mock_invoke.call_count)

        self.assertEqual(result, mock_response)
        mock_invoke.assert_called_once()
