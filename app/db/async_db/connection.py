"""
Async database connection management.
Provides connection pooling and session management.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from sqlalchemy import text
import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from ...core.config import Settings

logger = logging.getLogger(__name__)


class AsyncDatabaseConnection:
    """
    Manages asynchronous database connections and operations.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or Settings()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._asyncpg_pool: Optional[asyncpg.Pool] = None
        self._loop_id: Optional[int] = None
        self._creation_time = asyncio.get_event_loop().time()
        self._closing = False

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
        if self._asyncpg_pool and not self._closing:
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
                self.config.async_database_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=self.config.async_db_min_pool_size,
                max_overflow=self.config.async_db_max_pool_size
                - self.config.async_db_min_pool_size,
                pool_recycle=3600,
                connect_args={
                    "server_settings": {"jit": "off"},
                    "timeout": self.config.async_db_command_timeout,
                    "command_timeout": self.config.async_db_command_timeout,
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
                await self._close_asyncpg_pool()

            logger.info(f"🚰 Creating asyncpg pool for loop {current_loop_id}")
            self._asyncpg_pool = await asyncpg.create_pool(
                dsn=self.config.asyncpg_dsn,
                min_size=self.config.async_db_min_pool_size,
                max_size=self.config.async_db_max_pool_size,
                timeout=self.config.async_db_command_timeout,
            )

            self._loop_id = current_loop_id

        return self._asyncpg_pool

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def _close_asyncpg_pool(self):
        """Close asyncpg pool with proper cleanup."""
        if self._asyncpg_pool:
            try:
                # Close all connections gracefully
                await self._asyncpg_pool.close()
                # Wait for connections to close
                await asyncio.sleep(0.1)
                logger.info("🔒 Closed asyncpg pool")
            except Exception as e:
                logger.error(f"Error closing asyncpg pool: {e}")
            finally:
                self._asyncpg_pool = None

    async def close(self):
        """Close all database connections."""
        if self._closing:
            return

        self._closing = True
        logger.info("🔒 Closing database connections...")

        # Close asyncpg pool first
        await self._close_asyncpg_pool()

        # Then close SQLAlchemy engine
        if self._engine:
            try:
                await self._engine.dispose()
                logger.info("🔒 Disposed SQLAlchemy async engine")
            except Exception as e:
                logger.error(f"Error disposing engine: {e}")
            finally:
                self._engine = None

        self._session_factory = None
        self._loop_id = None
        self._closing = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @classmethod
    def from_settings(cls, settings: Settings) -> "AsyncDatabaseConnection":
        return cls(settings)
