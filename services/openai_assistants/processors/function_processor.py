"""
Processes function calls and converts them to frontend actions.
"""

import json
import logging
from typing import List, Dict, Any, Set, Optional
from ..utils.constants import FUNCTION_TO_ACTION_MAPPING

logger = logging.getLogger(__name__)


class FunctionProcessor:
    """Handles conversion of function calls to frontend actions."""

    def __init__(self):
        self.action_mapping = FUNCTION_TO_ACTION_MAPPING
        self.processed_signatures: Set[str] = set()

    def process_function_calls_to_actions(
        self, function_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Convert function calls to frontend actions with deduplication.

        Args:
            function_calls: List of function call results

        Returns:
            List of action dictionaries for frontend
        """
        actions = []

        # Reset processed signatures for a new batch
        self.processed_signatures.clear()

        for func_call in function_calls:
            action = self._process_single_function_call(func_call)
            if action:
                actions.append(action)

        logger.info(f"🔧 Total actions: {[a['type'] for a in actions]}")
        return actions

    def _process_single_function_call(
        self, func_call: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single function call into an action.

        Args:
            func_call: Function call dictionary

        Returns:
            Action dictionary or None if duplicate/error
        """
        function_name = func_call.get("function")
        result = func_call.get("result", {})
        arguments = func_call.get("arguments", {})

        logger.info(f"🔧 Processing: {function_name}")

        # Check for duplicate
        function_signature = self._create_function_signature(function_name, arguments)

        if function_signature in self.processed_signatures:
            logger.info(f"🔄 Skipping duplicate function call: {function_name}")
            return None

        self.processed_signatures.add(function_signature)

        # Handle errors
        if result.get("error"):
            return {
                "type": "error",
                "data": result,
                "function_called": function_name,
            }

        # Get action type from mapping
        action_type = self.action_mapping.get(function_name, f"show_{function_name}")

        # Create the action
        action = {
            "type": action_type,
            "data": result,
            "function_called": function_name,
        }

        logger.info(f"🔧 Created action: {action_type}")
        return action

    def _create_function_signature(self, function_name: str, arguments: Dict) -> str:
        """
        Create unique signature for function call.

        Args:
            function_name: Name of the function
            arguments: Function arguments

        Returns:
            Unique signature string
        """
        return f"{function_name}:{json.dumps(arguments, sort_keys=True)}"

    def reset_processed_signatures(self):
        """Reset the processed signatures set."""
        self.processed_signatures.clear()

    def get_action_type(self, function_name: str) -> str:
        """
        Get the action type for a function name.

        Args:
            function_name: Function name to map

        Returns:
            Corresponding action type
        """
        return self.action_mapping.get(function_name, f"show_{function_name}")

    def is_valid_function(self, function_name: str) -> bool:
        """
        Check if a function name is valid/known.

        Args:
            function_name: Function name to check

        Returns:
            True if function is known
        """
        return function_name in self.action_mapping
