"""
Controller for learning and feedback endpoints.
Handles learning-related HTTP requests and delegates to business services.
"""

from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.fin_learning_service import FinLearningService


class LearningController(BaseController):
    """Controller for learning and feedback operations"""

    def __init__(self):
        super().__init__()
        self.fin_learning_service = FinLearningService()

    def learn_from_feedback(self) -> Tuple[Dict[str, Any], int]:
        """
        Learn from user feedback to improve Fin's capabilities

        Expected JSON input:
        {
            "dataset": {
                "helpful_conversations": [
                    {
                        "id": 123,
                        "messages": [...],
                        "user_id": "user_123"
                    },
                    ...
                ],
                "unhelpful_conversations": [...],
                "tool_usage": {
                    "forecast_cash_flow": {
                        "total": 15,
                        "success": 12,
                        "contexts": ["What if I spend $5000 on marketing?", ...],
                        "parameters": [{"amount": 5000, "category": "marketing"}, ...]
                    },
                    ...
                }
            }
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Extract dataset
            dataset = data.get("dataset", {})

            # Validate dataset is not empty
            if not dataset or not any(dataset.values()):
                raise ValueError("Dataset cannot be empty")

            # Log request details
            helpful_count = len(dataset.get("helpful_conversations", []))
            unhelpful_count = len(dataset.get("unhelpful_conversations", []))
            tool_usage_count = len(dataset.get("tool_usage", {}))

            self.logger.info(f"Processing learning dataset:")
            self.logger.info(f"  - Helpful conversations: {helpful_count}")
            self.logger.info(f"  - Unhelpful conversations: {unhelpful_count}")
            self.logger.info(f"  - Tool usage entries: {tool_usage_count}")

            # Delegate to service
            result = self.fin_learning_service.process_learning_dataset(dataset)

            # Log result
            self.logger.info(
                f"Learning process completed with status: {result.get('status')}"
            )

            return self.success_response(result)

        return self.handle_request(_handle)
