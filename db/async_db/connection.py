"""
Async database connection management.
Provides connection pooling and session management.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)

from .config import AsyncDatabaseConfig

logger = logging.getLogger(__name__)


class AsyncDatabaseConnection:
    """
    Manages asynchronous database connections and operations.

    This class provides functionality to work with asynchronous database connections,
    including the creation and management of an async engine, session factory, and
    connection pool. It supports executing queries, leveraging asyncpg for raw query
    operations, and ensures proper cleanup of resources.

    Attributes
    ----------
    config : AsyncDatabaseConfig
        Configuration object for database connection settings.
    _engine : Optional[AsyncEngine]
        Internal async database engine instance, lazily created.
    _session_factory : Optional[async_sessionmaker]
        Internal async session factory instance, lazily created.
    _asyncpg_pool : Optional[asyncpg.Pool]
        Internal asyncpg connection pool for raw queries, lazily created.
    """

    def __init__(self, config: Optional[AsyncDatabaseConfig] = None):
        self.config = config or AsyncDatabaseConfig.from_env()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._asyncpg_pool: Optional[asyncpg.Pool] = None
        self._loop_id: Optional[int] = None
        self._creation_time = asyncio.get_event_loop().time()

    async def is_valid(self) -> bool:
        """Check if this connection is valid for the current event loop."""
        current_loop = asyncio.get_running_loop()
        current_loop_id = id(current_loop)

        # Check if we're on a different event loop
        if self._loop_id is not None and self._loop_id != current_loop_id:
            logger.warning(
                f"🔄 Event loop changed from {self._loop_id} to {current_loop_id}"
            )
            return False

        # Test the connection
        if self._asyncpg_pool:
            try:
                async with self._asyncpg_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return True
            except Exception as e:
                logger.error(f"Connection test failed: {e}")
                return False

        return True

    @property
    async def engine(self) -> AsyncEngine:
        """Get or create an async database engine."""
        current_loop_id = id(asyncio.get_running_loop())

        if self._engine is None or self._loop_id != current_loop_id:
            if self._engine:
                await self._engine.dispose()

            logger.debug("🛠 Creating async SQLAlchemy engine...")
            self._engine = create_async_engine(
                self.config.async_connection_string,
                echo=False,
                pool_pre_ping=True,
                pool_size=self.config.min_pool_size,
                max_overflow=self.config.max_pool_size - self.config.min_pool_size,
                pool_recycle=3600,
                connect_args={
                    "server_settings": {"jit": "off"},
                    "timeout": self.config.command_timeout,
                    "command_timeout": self.config.command_timeout,
                },
            )
            self._loop_id = current_loop_id
            logger.info("✅ Created async SQLAlchemy engine")
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
        """Get or create asyncpg pool for the current event loop."""
        current_loop_id = id(asyncio.get_running_loop())

        if self._asyncpg_pool is None or self._loop_id != current_loop_id:
            if self._asyncpg_pool:
                await self._asyncpg_pool.close()

            logger.info(f"🚰 Creating asyncpg pool for loop {current_loop_id}")
            self._asyncpg_pool = await asyncpg.create_pool(
                self.config.asyncpg_url, **self.config.get_pool_kwargs()
            )
            self._loop_id = current_loop_id

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
        logger.info("🔒 Closing database connections...")

        if self._asyncpg_pool:
            try:
                await self._asyncpg_pool.close()
                logger.info("🔒 Closed asyncpg pool")
            except Exception as e:
                logger.error(f"Error closing asyncpg pool: {e}")
            self._asyncpg_pool = None

        if self._engine:
            try:
                await self._engine.dispose()
                logger.info("🔒 Disposed SQLAlchemy async engine")
            except Exception as e:
                logger.error(f"Error disposing engine: {e}")
            self._engine = None

        self._session_factory = None
        self._loop_id = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
