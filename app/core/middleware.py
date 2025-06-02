"""
Custom middleware for FastAPI.
Replaces Flask middleware with FastAPI equivalents.
"""

import time
import logging
from uuid import uuid4
from typing import Callable
from fastapi.responses import JSONResponse
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing and logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid4())

        # Log request start
        logger.info(
            f"Request {request_id} started: {request.method} {request.url.path}"
        )

        # Time the request
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        # Log request completion
        logger.info(
            f"Request {request_id} completed in {process_time:.3f}s "
            f"with status {response.status_code}"
        )

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                content={"detail": "Internal server error"},
                status_code=500,
            )
