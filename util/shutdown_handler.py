"""
Graceful shutdown handler for Flask application.
Ensures all resources are properly cleaned up.
"""

import signal
import logging
import sys
from typing import Callable

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handles graceful shutdown of the application."""

    def __init__(self):
        self._shutting_down = False
        self._cleanup_handlers = []
        self._original_handlers = {}

    def register(self, cleanup_func: Callable):
        """Register a cleanup function to be called on shutdown."""
        self._cleanup_handlers.append(cleanup_func)

    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        # Store original handlers
        self._original_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, self._signal_handler
        )
        self._original_handlers[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, self._signal_handler
        )

        logger.info("✅ Installed graceful shutdown handlers")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self._shutting_down:
            logger.warning("⚠️ Force shutdown requested")
            sys.exit(1)

        self._shutting_down = True
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"🛑 Received {signal_name}, shutting down gracefully...")

        # Run cleanup handlers
        for handler in reversed(self._cleanup_handlers):
            try:
                logger.info(f"Running cleanup handler: {handler.__name__}")
                handler()
            except Exception as e:
                logger.error(f"Cleanup handler {handler.__name__} failed: {e}")

        # Restore original handlers
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

        logger.info("✅ Graceful shutdown complete")
        sys.exit(0)


# Global instance
shutdown_handler = GracefulShutdown()
