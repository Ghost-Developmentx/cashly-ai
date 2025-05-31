"""
Database initialization module with async support.
"""

import logging
from typing import Optional

from config.database import DatabaseConfig
from db.async_db import AsyncDatabaseConnection, AsyncDatabaseConfig
from db.migrations.m001_create_conversation_embeddings import upgrade

logger = logging.getLogger(__name__)


class AsyncDatabaseInitializer:
    """
    Handles asynchronous initialization of a database.

    This class is responsible for preparing the database environment,
    including testing the connection, setting up required extensions,
    managing migrations, and creating indexes. It leverages both
    asynchronous and synchronous operations to ensure proper database
    setup.

    Attributes
    ----------
    config : Optional[AsyncDatabaseConfig]
        Configuration details for the database connection.
    connection : AsyncDatabaseConnection
        Asynchronous database connection instance.
    """

    def __init__(self, config: Optional[AsyncDatabaseConfig] = None):
        self.config = config or AsyncDatabaseConfig.from_env()
        self.connection: Optional[AsyncDatabaseConnection] = (
            None  # ✅ Delay instantiation
        )

    async def initialize(self) -> bool:
        """
        Initialize a database with all required tables and extensions.

        Returns:
            True if successful, False otherwise
        """
        try:
            # ✅ Use shared async DB connection
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

        finally:
            # ✅ Do not close the shared pool here
            pass


# Singleton instance
_async_db_connection: Optional[AsyncDatabaseConnection] = None


async def get_async_db_connection() -> AsyncDatabaseConnection:
    """
    Get the global asynchronous database connection instance.

    Returns
    -------
    AsyncDatabaseConnection
        The global asynchronous database connection instance.
    """
    global _async_db_connection
    if _async_db_connection is None:
        config = AsyncDatabaseConfig.from_env()
        _async_db_connection = AsyncDatabaseConnection(config)
    return _async_db_connection


def get_db_connection():
    """
    Establish a synchronous database connection (deprecated).

    Returns
    -------
    DatabaseConnection
        An instance of `DatabaseConnection`.
    """
    logger.warning(
        "get_db_connection() is deprecated. Use get_async_db_connection() instead."
    )
    from db.connection import DatabaseConnection

    config = DatabaseConfig.from_env()
    return DatabaseConnection(config)
