import logging
import asyncio
import weakref
from typing import Dict, Any, Optional, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SingletonRegistry:
    """
    Process-wide registry for singleton instances.
    Tracks event loop association and recreates instances when needed.
    """

    _instances: Dict[str, Any] = {}
    _loop_refs: Dict[str, weakref.ref] = {}
    _factories: Dict[str, Callable] = {}
    _locks: Dict[str, asyncio.Lock] = {}

    @classmethod
    def get_lock(cls, key: str) -> asyncio.Lock:
        """Get or create a lock for a specific key."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        lock_key = f"{key}_{loop_id}"

        if lock_key not in cls._locks:
            cls._locks[lock_key] = asyncio.Lock()
        return cls._locks[lock_key]

    @classmethod
    async def get_or_create(
        cls, key: str, factory: Callable[..., T], *args, **kwargs
    ) -> T:
        """
        Get or create a singleton instance.

        Args:
            key: Unique identifier for the singleton
            factory: Callable that creates the instance
            *args, **kwargs: Arguments for the factory

        Returns:
            The singleton instance
        """
        async with cls.get_lock(key):
            # Get the current event loop
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error(f"No running event loop for singleton {key}")
                raise

            # Check if we have an instance for this loop
            if key in cls._instances:
                instance = cls._instances[key]
                loop_ref = cls._loop_refs.get(key)

                # Check if the loop is still the same and alive
                if loop_ref and loop_ref() is current_loop:
                    # Verify instance is still valid
                    if hasattr(instance, "is_valid"):
                        if await instance.is_valid():
                            logger.debug(f"✅ Reusing singleton: {key}")
                            return instance
                        else:
                            logger.warning(f"♻️ Singleton {key} is invalid")
                    else:
                        # No validity check, assume it's good
                        return instance
                else:
                    logger.warning(f"🔄 Event loop changed for {key}, recreating...")

                # Clean up old instance
                await cls._cleanup_instance(key)

            # Store factory for potential recreation
            cls._factories[key] = factory

            # Create a new instance
            logger.info(f"🏗️ Creating new singleton: {key}")
            instance = await factory(*args, **kwargs)

            # Store instance and loop reference
            cls._instances[key] = instance
            cls._loop_refs[key] = weakref.ref(current_loop)

            return instance

    @classmethod
    async def _cleanup_instance(cls, key: str):
        """Clean up a singleton instance."""
        if key in cls._instances:
            instance = cls._instances[key]

            # Call the cleanup method if available
            if hasattr(instance, "close"):
                try:
                    logger.debug(f"🧹 Cleaning up {key}")
                    await instance.close()
                except Exception as e:
                    logger.error(f"Error closing {key}: {e}")

            # Remove from registry
            del cls._instances[key]
            if key in cls._loop_refs:
                del cls._loop_refs[key]

    @classmethod
    async def cleanup_all(cls):
        """Clean up all singleton instances."""
        logger.info("🧹 Cleaning up all singletons...")

        # Create a list of keys to avoid modification during iteration
        keys = list(cls._instances.keys())

        for key in keys:
            await cls._cleanup_instance(key)

        cls._instances.clear()
        cls._loop_refs.clear()
        cls._locks.clear()
        logger.info("✅ All singletons cleaned up")

    @classmethod
    def get_instance_sync(cls, key: str) -> Optional[Any]:
        """Get an instance without async (for debugging only)."""
        return cls._instances.get(key)

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_instances": len(cls._instances),
            "instances": list(cls._instances.keys()),
            "has_factories": list(cls._factories.keys()),
        }


# Global registry instance
registry = SingletonRegistry()
