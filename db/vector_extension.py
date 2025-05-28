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
            # Create an extension if not exists
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()

            # Verify the extension is installed
            result = session.execute(
                text("SELECT COUNT(*) FROM pg_extension " "WHERE extname = 'vector'")
            )
            count = result.scalar()

            if count > 0:
                logger.info("pgvector extension is installed and ready")
                return True
            else:
                logger.error("pgvector extension installation failed")
                return False

        except Exception as e:
            logger.error(f"Error setting up pgvector extension: {e}")
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
