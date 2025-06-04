"""
Database initialization module with async support.
"""

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.db.async_db.connection import AsyncDatabaseConnection
from app.db.singleton_registry import registry

logger = logging.getLogger(__name__)


async def get_async_db_connection() -> AsyncDatabaseConnection:
    """
    Get the global asynchronous database connection instance.
    Uses a process-wide singleton registry to ensure true singleton behavior.
    """

    async def create_connection():
        settings = get_settings()
        return AsyncDatabaseConnection(settings)

    return await registry.get_or_create("async_db_connection", create_connection)


class AsyncDatabaseInitializer:
    """Handles asynchronous initialization of a database."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
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
            from app.db.async_db.vector_ops import AsyncVectorOperations

            vector_ops = AsyncVectorOperations(pool)

            if not await vector_ops.create_vector_extension():
                logger.error("Failed to create vector extension")
                return False

            # Run migrations (still sync for now)
            logger.info("Running database migrations...")
            from app.db.connection import DatabaseConnection

            sync_conn = DatabaseConnection.from_settings(self.settings)
            from app.db.migrations.m001_create_conversation_embeddings import upgrade
            upgrade(sync_conn)
            sync_conn.close()

            # Create vector index
            await vector_ops.create_vector_index(
                "conversation_embeddings", "embedding", index_type="ivfflat", lists=100
            )

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            return False
