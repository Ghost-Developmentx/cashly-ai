"""
Process-wide singleton registry to ensure true singletons across module imports.
"""

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SingletonRegistry:
    """
    Process-wide registry for singleton instances.
    Uses weak references to allow garbage collection when needed.
    """

    _instances: Dict[str, Any] = {}
    _locks: Dict[str, asyncio.Lock] = {}

    @classmethod
    def get_lock(cls, key: str) -> asyncio.Lock:
        """Get or create a lock for a specific key."""
        if key not in cls._locks:
            cls._locks[key] = asyncio.Lock()
        return cls._locks[key]

    @classmethod
    async def get_or_create(cls, key: str, factory, *args, **kwargs):
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
            # Check if we have a valid instance
            if key in cls._instances:
                instance = cls._instances[key]

                # Verify the instance is still valid
                if hasattr(instance, "is_valid"):
                    if await instance.is_valid():
                        logger.debug(f"✅ Reusing singleton: {key}")
                        return instance
                    else:
                        logger.warning(f"♻️ Singleton {key} is invalid, recreating...")
                        await cls._cleanup_instance(key)
                else:
                    return instance

            # Create new instance
            logger.info(f"🏗️ Creating new singleton: {key}")
            instance = await factory(*args, **kwargs)
            cls._instances[key] = instance
            return instance

    @classmethod
    async def _cleanup_instance(cls, key: str):
        """Clean up a singleton instance."""
        if key in cls._instances:
            instance = cls._instances[key]
            if hasattr(instance, "close"):
                try:
                    await instance.close()
                except Exception as e:
                    logger.error(f"Error closing {key}: {e}")
            del cls._instances[key]

    @classmethod
    async def cleanup_all(cls):
        """Clean up all singleton instances."""
        logger.info("🧹 Cleaning up all singletons...")
        keys = list(cls._instances.keys())
        for key in keys:
            await cls._cleanup_instance(key)
        cls._instances.clear()
        cls._locks.clear()


# Global registry instance
registry = SingletonRegistry()
