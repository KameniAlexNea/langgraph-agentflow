import unittest

from langgraph_agentflow.multi_step.state import MultiStepAgentState


class TestMultiStepState(unittest.TestCase):
    """Test cases for the MultiStepAgentState structure."""

    def test_state_structure(self):
        """Test that the MultiStepAgentState has the expected structure."""
        # Check that all expected keys are in the TypedDict
        expected_keys = {
            "messages",
            "route",
            "plan",
            "executed_steps",
            "current_step_index",
            "original_query",
        }

        # Get the actual keys from the TypedDict
        # Note: TypedDict.__annotations__ gives the type hints
        actual_keys = set(MultiStepAgentState.__annotations__.keys())

        # Verify all expected keys are present
        self.assertTrue(
            expected_keys.issubset(actual_keys),
            f"Missing keys: {expected_keys - actual_keys}",
        )
