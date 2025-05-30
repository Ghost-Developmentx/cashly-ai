"""
Database configuration for PostgreSQL with pgvector support.
"""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """
    Configuration for connecting to a PostgreSQL database.

    This class provides configuration settings necessary for establishing a connection
    to a PostgreSQL database. It includes the host address, port, database name,
    user credentials, and connection pool settings. The default values for optional
    parameters can be overridden by environment variables.

    Attributes
    ----------
    host : str
        The hostname or IP address of the PostgreSQL server.
    port : int
        The port number on which the PostgreSQL server is listening.
    database : str
        The name of the PostgreSQL database.
    user : str
        The username to authenticate with the PostgreSQL server.
    password : str
        The password associated with the `user`.
    pool_size : int
        The number of connections to maintain in the pool. Defaults to 5.
    max_overflow : int
        The number of connections allowed above `pool_size` when the pool is full.
        Defaults to 10.
    """

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
