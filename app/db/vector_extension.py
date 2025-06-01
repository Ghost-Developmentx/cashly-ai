"""
PostgreSQL pgvector extension management.
"""

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class VectorExtensionManager:
    """Manages pgvector extension setup and operations."""

    @staticmethod
    def setup_extension(session: Session) -> bool:
        """
        Set up pgvector extension in the database.

        Args:
            session: Database session

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if extension already exists
            result = session.execute(
                text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            )
            count = result.scalar()

            if count > 0:
                logger.info("✅ pgvector extension is already installed")
                return True

            logger.info("Creating pgvector extension...")
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
            return True

        except Exception as e:
            logger.error(f"❌ Error setting up pgvector extension: {e}")

            # Check what extensions are available
            try:
                result = session.execute(
                    text("SELECT * FROM pg_available_extensions WHERE name = 'vector'")
                )
                available = result.fetchall()
                if available:
                    logger.info(
                        f"ℹ️ pgvector is listed in pg_available_extensions: {available}"
                    )
                else:
                    logger.error("❌ pgvector is NOT listed in pg_available_extensions")
            except Exception as e2:
                logger.error(f"❌ Could not check available extensions: {e2}")

            session.rollback()
            return False

    @staticmethod
    def get_vector_dimensions(session: Session) -> Optional[int]:
        """
        Get supported vector dimensions.

        Args:
            session: Database session

        Returns:
            Vector dimension limit or None
        """
        try:
            session.execute(text("SELECT '1'::vector(1536)"))
            return 1536
        except Exception as e:
            logger.error(f"Error checking vector dimensions: {e}")
            return None
