"""
Base controller class with common functionality for all controllers.
"""

import json
import logging
from typing import Dict, Any, Tuple, Callable
from flask import request
from datetime import datetime, date


class BaseController:
    """Base controller with common functionality"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_request_data() -> Dict[str, Any]:
        """
        Get and validate request JSON data

        Returns:
            Parsed JSON data from request

        Raises:
            ValueError: If request data is invalid
        """
        if not request.is_json:
            raise ValueError("Request must be JSON")

        data = request.get_json()
        if not data:
            raise ValueError("Request body cannot be empty")

        return data

    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: list):
        """
        Validate that required fields are present in data

        Args:
            data: Request data dictionary
            required_fields: List of required field names

        Raises:
            ValueError: If any required field is missing
        """
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    @staticmethod
    def success_response(
        data: Any, status_code: int = 200
    ) -> Tuple[Dict[str, Any], int]:
        """
        Create a success response

        Args:
            data: Response data
            status_code: HTTP status code (default: 200)

        Returns:
            Tuple of (response_dict, status_code)
        """
        return data, status_code

    @staticmethod
    def error_response(
        message: str, status_code: int = 400
    ) -> Tuple[Dict[str, Any], int]:
        """
        Create an error response

        Args:
            message: Error message
            status_code: HTTP status code (default: 400)

        Returns:
            Tuple of (response_dict, status_code)
        """
        return {
            "error": message,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
        }, status_code

    def handle_request(self, handler: Callable) -> Tuple[Dict[str, Any], int]:
        """
        Handle request with error handling

        Args:
            handler: Request handler function

        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            return handler()
        except ValueError as e:
            self.logger.warning(f"Validation error: {str(e)}")
            return self.error_response(str(e), 400)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return self.error_response("An unexpected error occurred", 500)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""

    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)
