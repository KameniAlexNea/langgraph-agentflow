import unittest
from unittest.mock import MagicMock, patch

from langgraph.prebuilt import ToolNode

from langgraph_agentflow.multi_step.graph import (
    build_multi_step_graph,
    create_decision_functions,
)


class TestMultiStepGraph(unittest.TestCase):
    """Test cases for the multi-step graph module."""

    def setUp(self):
        self.plan_function = MagicMock()
        self.execute_function = MagicMock()
        self.process_function = MagicMock()
        self.specialized_agents = {"general": MagicMock(), "specialized": MagicMock()}
        self.tool_nodes = {"general_tools": ToolNode([])}

    def test_create_decision_functions(self):
        """Test creating decision functions for the graph."""
        decision_fns = create_decision_functions()

        self.assertIn("decide_plan_or_route", decision_fns)
        self.assertIn("route_to_specialist", decision_fns)
        self.assertIn("route_tools", decision_fns)
        self.assertIn("decide_continue_or_synthesize", decision_fns)

        # Test the decision functions with sample states
        plan_decision_fn = decision_fns["decide_plan_or_route"]

        # Test with a plan
        state_with_plan = {"plan": ["Step 1"]}
        self.assertEqual(plan_decision_fn(state_with_plan), "execute_plan")

        # Test with a route
        state_with_route = {"plan": [], "route": "general"}
        self.assertEqual(plan_decision_fn(state_with_route), "route_simple")

    @patch("langgraph_agentflow.multi_step.graph.StateGraph")
    @patch("langgraph_agentflow.multi_step.graph.MemorySaver")
    def test_build_multi_step_graph(self, mock_memory_saver, mock_state_graph):
        """Test building a multi-step graph."""
        # Set up mocks
        mock_graph_instance = MagicMock()
        mock_state_graph.return_value = mock_graph_instance
        mock_graph_instance.compile.return_value = MagicMock()

        # Call the function
        graph = build_multi_step_graph(
            self.plan_function,
            self.execute_function,
            self.process_function,
            self.specialized_agents,
            self.tool_nodes,
        )

        # Assertions
        self.assertIsNotNone(graph)
        # Verify graph construction
        mock_graph_instance.add_node.assert_called()
        mock_graph_instance.add_edge.assert_called()
        mock_graph_instance.add_conditional_edges.assert_called()
        mock_graph_instance.compile.assert_called_once()
