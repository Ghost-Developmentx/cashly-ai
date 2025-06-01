"""
Controller for transaction categorization endpoints.
Now uses async categorization service.
"""

import asyncio
from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from app.services import AsyncCategorizationService


class CategorizationController(BaseController):
    """Controller for categorization operations"""

    def __init__(self):
        super().__init__()
        self.categorization_service = AsyncCategorizationService()

    def categorize_transaction(self) -> Tuple[Dict[str, Any], int]:
        """
        Categorize a single transaction.

        Expected JSON input:
        {
            "description": "Starbucks Coffee",
            "amount": -5.75,
            "merchant": "Starbucks"  # Optional
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["description", "amount"])

            description = data.get("description")
            amount = float(data.get("amount", 0))
            merchant = data.get("merchant")

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.categorization_service.categorize_transaction(
                        description, amount, merchant
                    )
                )
            finally:
                loop.close()

            return self.success_response(result)

        return self.handle_request(_handle)

    def categorize_batch(self) -> Tuple[Dict[str, Any], int]:
        """
        Categorize multiple transactions.

        Expected JSON input:
        {
            "transactions": [
                {
                    "description": "Walmart",
                    "amount": -45.32,
                    "merchant": "Walmart"
                },
                ...
            ]
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["transactions"])

            transactions = data.get("transactions", [])

            if not transactions:
                return self.error_response("No transactions provided", 400)

            self.logger.info(f"Categorizing {len(transactions)} transactions")

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    self.categorization_service.categorize_batch(transactions)
                )
            finally:
                loop.close()

            response = {"categorized_count": len(results), "results": results}

            return self.success_response(response)

        return self.handle_request(_handle)

    def learn_from_feedback(self) -> Tuple[Dict[str, Any], int]:
        """
        Learn from user categorization feedback.

        Expected JSON input:
        {
            "description": "Coffee Shop XYZ",
            "amount": -3.50,
            "correct_category": "Food & Dining"
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(
                data, ["description", "amount", "correct_category"]
            )

            description = data.get("description")
            amount = float(data.get("amount", 0))
            correct_category = data.get("correct_category")

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                success = loop.run_until_complete(
                    self.categorization_service.learn_from_feedback(
                        description, amount, correct_category
                    )
                )
            finally:
                loop.close()

            if success:
                return self.success_response(
                    {"message": "Successfully learned from feedback", "success": True}
                )
            else:
                return self.error_response("Failed to learn from feedback", 500)

        return self.handle_request(_handle)
