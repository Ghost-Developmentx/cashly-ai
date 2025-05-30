"""
Async database connection management.
Provides connection pooling and session management.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)

from config import AsyncDatabaseConfig

logger = logging.getLogger(__name__)


class AsyncDatabaseConnection:
    """Manages async database connections with pooling."""

    def __init__(self, config: Optional[AsyncDatabaseConfig] = None):
        self.config = config or AsyncDatabaseConfig.from_env()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._asyncpg_pool: Optional[asyncpg.Pool] = None

    @property
    async def engine(self) -> AsyncEngine:
        """Get or create async database engine."""
        if self._engine is None:
            self._engine = create_async_engine(
                self.config.async_connection_string,
                echo=False,
                pool_pre_ping=True,
                pool_size=self.config.min_pool_size,
                max_overflow=self.config.max_pool_size - self.config.min_pool_size,
                pool_recycle=3600,  # Recycle connections after 1 hour
                connect_args={
                    "server_settings": {"jit": "off"},
                    "timeout": self.config.command_timeout,
                    "command_timeout": self.config.command_timeout,
                },
            )
            logger.info("Created async database engine")
        return self._engine

    @property
    async def session_factory(self) -> async_sessionmaker:
        """Get or create an async session factory."""
        if self._session_factory is None:
            engine = await self.engine
            self._session_factory = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
        return self._session_factory

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions.

        Usage:
            async with db.get_session() as session:
                result = await session.execute(query)
        """
        factory = await self.session_factory
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def get_asyncpg_pool(self) -> asyncpg.Pool:
        """Get or create asyncpg connection pool for raw queries."""
        if self._asyncpg_pool is None:
            self._asyncpg_pool = await asyncpg.create_pool(
                self.config.asyncpg_url, **self.config.get_pool_kwargs()
            )
            logger.info("Created asyncpg connection pool")
        return self._asyncpg_pool

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def close(self):
        """Close all database connections."""
        if self._asyncpg_pool:
            await self._asyncpg_pool.close()
            logger.info("Closed asyncpg pool")

        if self._engine:
            await self._engine.dispose()
            logger.info("Disposed async engine")

        self._engine = None
        self._session_factory = None
        self._asyncpg_pool = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
