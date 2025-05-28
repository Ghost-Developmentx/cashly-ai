"""
Base controller class for Cashly API endpoints.
Provides common functionality for all controllers.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from flask import jsonify, request
from datetime import datetime
import json


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)


class BaseController:
    """Base controller providing common functionality for all API endpoints"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle_request(
        self, handler_func, *args, **kwargs
    ) -> Tuple[Dict[str, Any], int]:
        """
        Common request handling with error management and logging

        Args:
            handler_func: The actual handler function to execute
            *args, **kwargs: Arguments to pass to the handler

        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            # Log incoming request
            self.logger.info(f"Processing request: {handler_func.__name__}")

            # Execute the handler
            result = handler_func(*args, **kwargs)

            # Handle different return types
            if isinstance(result, tuple):
                response_data, status_code = result
            else:
                response_data, status_code = result, 200

            # Ensure response is JSON serializable
            response_json = json.loads(json.dumps(response_data, cls=DateTimeEncoder))

            self.logger.info(f"Request completed successfully: {handler_func.__name__}")
            return response_json, status_code

        except ValueError as e:
            self.logger.warning(
                f"Validation error in {handler_func.__name__}: {str(e)}"
            )
            return self._error_response(f"Invalid input: {str(e)}", 400), 400

        except Exception as e:
            self.logger.error(f"Error in {handler_func.__name__}: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")

            return (
                self._error_response(
                    "An internal error occurred. Please try again.", 500
                ),
                500,
            )

    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
        """
        Validate that required fields are present in the request data

        Args:
            data: Request data dictionary
            required_fields: List of required field names

        Raises:
            ValueError: If any required fields are missing
        """
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    @staticmethod
    def get_request_data() -> Dict[str, Any]:
        """
        Get and validate JSON request data

        Returns:
            Dictionary containing request data

        Raises:
            ValueError: If request data is invalid
        """
        if not request.is_json:
            raise ValueError("Request must contain valid JSON")

        data = request.get_json()
        if not data:
            raise ValueError("Request body cannot be empty")

        return data

    @staticmethod
    def _error_response(message: str, status_code: int) -> Dict[str, Any]:
        """
        Create a standardized error response

        Args:
            message: Error message
            status_code: HTTP status code

        Returns:
            Error response dictionary
        """
        return {
            "error": message,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code,
        }

    @staticmethod
    def success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standardized success response

        Args:
            data: Response data
            message: Optional success message

        Returns:
            Success response dictionary
        """
        response = {"success": True, "timestamp": datetime.now().isoformat()}

        if message:
            response["message"] = message

        # Handle different data types
        if isinstance(data, dict):
            response.update(data)
        else:
            response["data"] = data

        return response
