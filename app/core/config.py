"""
Centralized configuration using Pydantic Settings.
Replaces: config/database.py, config/openai.py
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Dict, Any
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""

    # API Configuration
    api_v1_prefix: str = "/api/v1"
    project_name: str = "Cashly AI Service"
    version: str = "2.0.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # CORS
    backend_cors_origins: list[str] = ["*"]

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_max_tokens: int = 8191
    openai_request_timeout: int = 60
    openai_max_retries: int = 3

    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cashly_ai_vectors"
    postgres_user: str = "cashly_ai"
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")

    # Connection Pool Settings
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # Async Database Settings
    async_db_min_pool_size: int = 10
    async_db_max_pool_size: int = 20
    async_db_command_timeout: float = 10.0

    # Redis Configuration (Future)
    redis_url: Optional[str] = None

    # Rails API Configuration
    rails_api_url: Optional[str] = Field(None, env="RAILS_API_URL")
    internal_api_key: str = "your-secure-internal-api-key"

    # Assistant IDs
    transaction_assistant_id: Optional[str] = None
    account_assistant_id: Optional[str] = None
    invoice_assistant_id: Optional[str] = None
    forecasting_assistant_id: Optional[str] = None
    budget_assistant_id: Optional[str] = None

    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        """Construct async database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
