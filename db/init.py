"""
Database initialization module with async support.
"""

import logging
import asyncio
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
        self.connection = AsyncDatabaseConnection(self.config)

    async def initialize(self) -> bool:
        """
        Initialize database with all required tables and extensions.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Test connection
            if not await self.connection.test_connection():
                logger.error("Database connection failed")
                return False

            # Setup pgvector extension
            pool = await self.connection.get_asyncpg_pool()
            from db.async_db.vector_ops import AsyncVectorOperations

            vector_ops = AsyncVectorOperations(pool)

            if not await vector_ops.create_vector_extension():
                logger.error("Failed to create vector extension")
                return False

            # Run migrations (still sync for now)
            logger.info("Running database migrations...")
            # Note: Migrations would need to be converted to async
            # For now, we'll use sync connection for migrations
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
            await self.connection.close()


# Singleton instance
_async_db_connection: Optional[AsyncDatabaseConnection] = None


async def get_async_db_connection() -> AsyncDatabaseConnection:
    """
    Get the global asynchronous database connection instance.

    This function initializes and retrieves a singleton instance of an asynchronous database connection.
    If the connection is not yet created, it will use configuration data loaded from the environment to
    initialize the connection. Subsequent calls to this function will return the same instance.

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
    Establish a synchronous database connection.

    This function retrieves database configuration using environment variables
    and establishes a connection to the database. Note that this function
    is deprecated and should be replaced by `get_async_db_connection()` for
    asynchronous database operations.

    Warnings
    --------
    This function is deprecated. Use `get_async_db_connection()` instead.

    Returns
    -------
    DatabaseConnection
        An instance of `DatabaseConnection` representing the established
        database connection.
    """
    logger.warning(
        "get_db_connection() is deprecated. " "Use get_async_db_connection() instead."
    )
    from db.connection import DatabaseConnection
    from config.database import DatabaseConfig

    config = DatabaseConfig.from_env()
    return DatabaseConnection(config)
