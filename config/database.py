"""
Database configuration for PostgreSQL with pgvector support.
"""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    host: str
    port: int
    database: str
    user: str
    password: str
    pool_size: int = 5
    max_overflow: int = 10

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "cashly_ai_vectors"),
            user=os.getenv("POSTGRES_USER", "cashly_ai"),
            password=os.getenv("POSTGRES_PASSWORD", "qwerT12321"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        )

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
