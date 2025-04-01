import unittest
from unittest.mock import MagicMock, patch

from langgraph_agentflow.multi_step import create_multi_step_agent


class TestMultiStepAgent(unittest.TestCase):
    def setUp(self):
        # Create a mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = MagicMock(content="SIMPLE: general")
        
        # Define test agent configs
        self.agent_tools = [
            {
                "name": "general",
                "description": "Handles general conversation and queries.",
                "tools": [],
            }
        ]

    def test_create_multi_step_agent(self):
        graph = create_multi_step_agent(self.mock_llm, self.agent_tools)
        
        # Test that the graph is created correctly
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
