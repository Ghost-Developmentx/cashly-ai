"""
Database initialization module with async support.
"""

import asyncio
import atexit
import logging
import signal
from typing import Optional

from config.database import DatabaseConfig
from db.async_db import AsyncDatabaseConnection, AsyncDatabaseConfig
from db.singleton_registry import registry
from db.migrations.m001_create_conversation_embeddings import upgrade

logger = logging.getLogger(__name__)


async def get_async_db_connection() -> AsyncDatabaseConnection:
    """
    Get the global asynchronous database connection instance.
    Uses a process-wide singleton registry to ensure true singleton behavior.
    """

    async def create_connection():
        config = AsyncDatabaseConfig.from_env()
        return AsyncDatabaseConnection(config)

    return await registry.get_or_create("async_db_connection", create_connection)


def cleanup_handler(signum=None, frame=None):
    """Handle cleanup on shutdown."""
    logger.info("🛑 Shutdown signal received, cleaning up...")

    # Run async cleanup in a new event loop if needed
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(registry.cleanup_all())
    logger.info("✅ Cleanup complete")


# Register cleanup handlers
atexit.register(cleanup_handler)
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)


class AsyncDatabaseInitializer:
    """Handles asynchronous initialization of a database."""

    def __init__(self, config: Optional[AsyncDatabaseConfig] = None):
        self.config = config or AsyncDatabaseConfig.from_env()
        self.connection: Optional[AsyncDatabaseConnection] = None

    async def initialize(self) -> bool:
        """Initialize a database with all required tables and extensions."""
        try:
            # Use the singleton connection
            self.connection = await get_async_db_connection()

            # Test connection
            if not await self.connection.test_connection():
                logger.error("Database connection failed")
                return False

            # Set up pgvector extension
            pool = await self.connection.get_asyncpg_pool()
            from db.async_db.vector_ops import AsyncVectorOperations

            vector_ops = AsyncVectorOperations(pool)

            if not await vector_ops.create_vector_extension():
                logger.error("Failed to create vector extension")
                return False

            # Run migrations (still sync for now)
            logger.info("Running database migrations...")
            from db.connection import DatabaseConnection

            sync_config = DatabaseConfig.from_env()
            sync_conn = DatabaseConnection(sync_config)
            upgrade(sync_conn)
            sync_conn.close()

            # Create vector index
            await vector_ops.create_vector_index(
                "conversation_embeddings", "embedding", index_type="ivfflat", lists=100
            )

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
