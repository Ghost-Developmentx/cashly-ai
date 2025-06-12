"""
Database connection management with connection pooling.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import Settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connections with pooling."""

    def __init__(self, settings: Settings):
        self.config = settings
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "DatabaseConnection":
        return cls(settings)

    @property
    def engine(self) -> Engine:
        """Get or create a database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.DB_POOL_SIZE,
                max_overflow=self.config.DB_MAX_OVERFLOW,
                pool_pre_ping=True,
                echo=False,
            )
            logger.info("Database engine created successfully")
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine, expire_on_commit=False
            )
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")
