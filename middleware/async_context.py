"""
Async context middleware for Flask.
Manages async operations using a centralized event loop.
"""

import logging
from flask import Flask, g
from .async_manager import async_manager

logger = logging.getLogger(__name__)


class AsyncContextMiddleware:
    """
    Middleware to provide async context for Flask requests.
    Uses a centralized event loop manager instead of per-request loops.
    """

    def __init__(self, app: Flask):
        self.app = app
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up request handlers."""

        @self.app.before_request
        def before_request():
            """Store async manager in Flask g object."""
            g.async_manager = async_manager
            logger.debug("Async context established for request")

        @self.app.teardown_appcontext
        def teardown_appcontext(exception):
            """Clean up request context."""
            if exception:
                logger.error(f"Request ended with exception: {exception}")

            # Remove async manager from g
            g.pop("async_manager", None)
