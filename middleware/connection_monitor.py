import psutil
import os
import logging
from flask import Flask

logger = logging.getLogger(__name__)


class ConnectionMonitorMiddleware:
    """Middleware to monitor connection usage."""

    def __init__(self, app: Flask):
        self.app = app
        self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize the middleware with the Flask app."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_request(self.teardown_request)

    def before_request(self):
        """Called before each request."""
        self._log_stats("Before request")

    def after_request(self, response):
        """Called after each request."""
        self._log_stats("After request")
        return response

    def teardown_request(self, exception=None):
        """Called at the end of each request."""
        if exception:
            self._log_stats(f"Request teardown (exception: {type(exception).__name__})")

    @staticmethod
    def _log_stats(context: str):
        """Log connection statistics."""
        try:
            proc = psutil.Process(os.getpid())
            conns = proc.net_connections(kind="inet")

            # Categorize connections
            established = sum(1 for c in conns if c.status == "ESTABLISHED")
            listening = sum(1 for c in conns if c.status == "LISTEN")
            other = len(conns) - established - listening

            logger.info(
                f"📊 {context} | FDs: {proc.num_fds()} | "
                f"Connections: EST={established}, LISTEN={listening}, OTHER={other}"
            )
        except Exception as e:
            logger.error(f"Connection monitoring error: {e}")
