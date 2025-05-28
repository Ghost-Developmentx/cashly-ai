"""
Controller for anomaly detection endpoints.
Handles anomaly detection HTTP requests and delegates to business services.
"""

from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.anomaly_service import AnomalyService


class AnomalyController(BaseController):
    """Controller for anomaly detection operations"""

    def __init__(self):
        super().__init__()
        self.anomaly_service = AnomalyService()

    def detect_anomalies(self) -> Tuple[Dict[str, Any], int]:
        """
        Detect anomalous transactions

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [
                {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
                {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
                ...
            ],
            "threshold": -0.5  # Optional anomaly score threshold
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "transactions"])

            # Extract parameters
            user_id = data.get("user_id")
            transactions = data.get("transactions", [])
            threshold = data.get("threshold")

            # Validate transactions data
            if not transactions:
                raise ValueError("Transactions list cannot be empty")

            # Validate threshold if provided
            if threshold is not None and not isinstance(threshold, (int, float)):
                raise ValueError("Threshold must be a number")

            # Log request details
            self.logger.info(f"Detecting anomalies for user {user_id}")
            if threshold is not None:
                self.logger.info(f"Using custom threshold: {threshold}")

            # Delegate to service
            result = self.anomaly_service.detect_anomalies(
                user_id=user_id, transactions=transactions, threshold=threshold
            )

            # Log result
            anomaly_count = len(result.get("anomalies", []))
            self.logger.info(
                f"Anomaly detection completed with {anomaly_count} anomalies found"
            )

            return self.success_response(result)

        return self.handle_request(_handle)
