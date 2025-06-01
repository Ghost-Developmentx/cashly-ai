"""
Centralized async event loop manager for Flask.
Ensures proper lifecycle management of async resources.
"""

import asyncio
import threading
import logging
from typing import Optional, Any, Callable, TypeVar
from concurrent.futures import Future
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncManager:
    """
    Manages a single background event loop for the entire Flask application.
    This ensures all async operations use the same loop and connections remain valid.
    """

    _instance: Optional["AsyncManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._thread: Optional[threading.Thread] = None
            self._started = threading.Event()
            self._stopping = False
            self._initialize()

    def _initialize(self):
        """Initialize the background event loop."""
        if self._loop is not None:
            return

        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        # Wait for the loop to start
        if not self._started.wait(timeout=5):
            raise RuntimeError("Failed to start async event loop")

        logger.info("✅ Async event loop initialized")

    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Signal that we're ready
            self._started.set()

            # Run forever until stopped
            self._loop.run_forever()

        except Exception as e:
            logger.error(f"Event loop crashed: {e}", exc_info=True)
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    def run_async(self, coro: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Run an async function from sync context.

        Args:
            coro: Async function to run
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the async function
        """
        if self._loop is None or self._stopping:
            raise RuntimeError("Async manager is not running")

        # Create future to get result
        future = Future()

        async def _wrapper():
            try:
                result = await coro(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        # Schedule coroutine on the event loop
        asyncio.run_coroutine_threadsafe(_wrapper(), self._loop)

        # Wait for result (with timeout)
        return future.result(timeout=300)  # 5 minute timeout

    def cleanup(self):
        """Clean up the event loop and resources."""
        if self._stopping:
            return

        self._stopping = True
        logger.info("🛑 Shutting down async manager...")

        if self._loop and not self._loop.is_closed():
            # Schedule cleanup on the event loop
            async def _cleanup():
                from db.singleton_registry import registry

                await registry.cleanup_all()

            future = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)

            try:
                future.result(timeout=10)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            # Stop the event loop
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for the thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("✅ Async manager shutdown complete")


# Global instance
async_manager = AsyncManager()


def run_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to run async functions in Flask routes.

    Usage:
        @run_async
        async def my_async_function():
            return await some_async_operation()

        # In Flask route:
        result = my_async_function()
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return async_manager.run_async(func, *args, **kwargs)

    return wrapper
