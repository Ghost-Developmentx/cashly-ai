"""
Async database configuration.
Manages connection settings and pool configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AsyncDatabaseConfig:
    """Async database configuration settings."""

    host: str
    port: int
    database: str
    user: str
    password: str

    # Async-specific settings
    min_pool_size: int = 10
    max_pool_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0

    # Performance settings
    statement_cache_size: int = 1024
    command_timeout: Optional[float] = 10.0

    @classmethod
    def from_env(cls) -> "AsyncDatabaseConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "cashly_ai_vectors"),
            user=os.getenv("POSTGRES_USER", "cashly_ai"),
            password=os.getenv("POSTGRES_PASSWORD", "qwerT12321"),
            min_pool_size=int(os.getenv("ASYNC_DB_MIN_POOL_SIZE", "10")),
            max_pool_size=int(os.getenv("ASYNC_DB_MAX_POOL_SIZE", "20")),
            max_queries=int(os.getenv("ASYNC_DB_MAX_QUERIES", "50000")),
            max_inactive_connection_lifetime=float(
                os.getenv("ASYNC_DB_MAX_INACTIVE_LIFETIME", "300.0")
            ),
            statement_cache_size=int(
                os.getenv("ASYNC_DB_STATEMENT_CACHE_SIZE", "1024")
            ),
            command_timeout=float(os.getenv("ASYNC_DB_COMMAND_TIMEOUT", "10.0")),
        )

    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    @property
    def asyncpg_url(self) -> str:
        """Get asyncpg-specific connection URL."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    def get_pool_kwargs(self) -> dict:
        """Get asyncpg pool configuration."""
        return {
            "min_size": self.min_pool_size,
            "max_size": self.max_pool_size,
            "max_queries": self.max_queries,
            "max_inactive_connection_lifetime": self.max_inactive_connection_lifetime,
            "command_timeout": self.command_timeout,
        }
