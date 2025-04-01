import unittest
from unittest.mock import MagicMock, patch

from langgraph_agentflow.single_step import create_hierarchical_agent


class TestSingleStepAgent(unittest.TestCase):
    def setUp(self):
        # Create a mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = MagicMock(content="general")
        
        # Define test agent configs
        self.agent_configs = [
            {
                "name": "general",
                "description": "Handles general conversation and queries.",
                "tools": [],
            }
        ]

    def test_create_hierarchical_agent(self):
        # Set visualize=False to avoid needing graphviz in tests
        graph, config, stream_fn, loop_fn = create_hierarchical_agent(
            self.mock_llm, self.agent_configs, visualize=False
        )
        
        # Test that the functions and graph are created correctly
        self.assertIsNotNone(graph)
        self.assertTrue(callable(stream_fn))
        self.assertTrue(callable(loop_fn))


if __name__ == "__main__":
    unittest.main()
