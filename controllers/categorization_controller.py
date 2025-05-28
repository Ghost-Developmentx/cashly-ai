from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.categorize_service import CategorizationService


class CategorizationController(BaseController):
    """Controller for transaction categorization operations"""

    def __init__(self):
        super().__init__()
        self.categorization_service = CategorizationService()

    def categorize_transaction(self) -> Tuple[Dict[str, Any], int]:
        """
        Categorize a single transaction

        Expected JSON input:
        {
            "description": "AMAZON PAYMENT",
            "amount": -45.67,
            "date": "2025-03-10"
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["description", "amount"])

            # Extract parameters
            description = data.get("description", "")
            amount = data.get("amount", 0)
            date = data.get("date")

            # Log request details
            self.logger.info(f"Categorizing transaction: {description}, ${amount}")

            # Delegate to service
            result = self.categorization_service.categorize_transaction(
                description=description, amount=amount, date=date
            )

            # Log result
            self.logger.info(
                f"Categorization result: {result['category']} "
                f"(confidence: {result['confidence']:.2f})"
            )

            return self.success_response(result)

        return self.handle_request(_handle)
