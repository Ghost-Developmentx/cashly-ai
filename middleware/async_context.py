"""
Middleware to ensure proper async context for each request.
"""

import asyncio
import logging
from flask import g

logger = logging.getLogger(__name__)


class AsyncContextMiddleware:
    def __init__(self, app):
        self.app = app
        app.before_request(self.before_request)
        app.teardown_appcontext(self.teardown_appcontext)

    def before_request(self):
        """Ensure each request has its own event loop."""
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            g._async_loop_reused = True
        except RuntimeError:
            # Create a new event loop for this request
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            g._async_loop = loop
            g._async_loop_reused = False

    def teardown_appcontext(self, exception):
        """Clean up request-specific resources."""
        # Only close the loop if we created it
        if hasattr(g, "_async_loop") and not g._async_loop_reused:
            loop = g._async_loop

            # Close any pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Run until all tasks are cancelled
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            # Close the loop
            loop.close()
