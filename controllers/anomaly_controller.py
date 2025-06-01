"""
Controller for anomaly detection endpoints.
Now uses async anomaly service.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from app.services import AsyncAnomalyService


class AnomalyController(BaseController):
    """Controller for anomaly detection operations"""

    def __init__(self):
        super().__init__()
        self.anomaly_service = AsyncAnomalyService()

    def detect_anomalies(self) -> Tuple[Dict[str, Any], int]:
        """
        Detect anomalous transactions.

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [...],
            "threshold": 2.5  # Optional, standard deviations
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "transactions"])

            user_id = data.get("user_id")
            transactions = data.get("transactions", [])
            threshold = data.get("threshold")

            # Validate threshold if provided
            if threshold is not None:
                try:
                    threshold = float(threshold)
                    if threshold <= 0 or threshold > 5:
                        return self.error_response(
                            "Threshold must be between 0 and 5", 400
                        )
                except ValueError:
                    return self.error_response("Invalid threshold value", 400)

            self.logger.info(
                f"Detecting anomalies for user {user_id} with "
                f"{len(transactions)} transactions"
            )

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.anomaly_service.detect_anomalies(
                        user_id, transactions, threshold
                    )
                )
            finally:
                loop.close()

            if "error" in result:
                self.logger.error(f"Anomaly detection failed: {result['error']}")
                return self.error_response(result["error"], 400)

            anomaly_count = len(result.get("anomalies", []))
            self.logger.info(f"Detected {anomaly_count} anomalies for user {user_id}")

            return self.success_response(result)

        return self.handle_request(_handle)

    def get_anomaly_summary(self) -> Tuple[Dict[str, Any], int]:
        """
        Get summary of anomalies without full details.

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [...]
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "transactions"])

            user_id = data.get("user_id")
            transactions = data.get("transactions", [])

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.anomaly_service.detect_anomalies(user_id, transactions)
                )
            finally:
                loop.close()

            if "error" in result:
                return self.error_response(result["error"], 400)

            # Return only summary
            summary = result.get("summary", {})
            summary["user_id"] = user_id
            summary["analysis_date"] = datetime.now().isoformat()

            return self.success_response(summary)

        return self.handle_request(_handle)
