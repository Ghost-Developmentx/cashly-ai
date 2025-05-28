"""
Database initialization module.
"""

import logging
from typing import Optional

from config.database import DatabaseConfig
from db.connection import DatabaseConnection
from db.migrations.m001_create_conversation_embeddings import upgrade

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and setup."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()
        self.connection = DatabaseConnection(self.config)

    def initialize(self) -> bool:
        """
        Initialize a database with all required tables and extensions.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Test connection
            if not self.connection.test_connection():
                logger.error("Database connection failed")
                return False

            # Run migrations
            logger.info("Running database migrations...")
            upgrade(self.connection)

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
        finally:
            self.connection.close()


# Singleton instance
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """Get or create a database connection instance."""
    global _db_connection
    if _db_connection is None:
        config = DatabaseConfig.from_env()
        _db_connection = DatabaseConnection(config)
    return _db_connection
