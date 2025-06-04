# app/core/config.py
"""
Centralized configuration using Pydantic Settings.
Replaces: config/database.py, config/openai.py
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
    )

    # ML Model Configuration
    model_dir: str = Field(
        default="data/trained_models",
        validation_alias="MODEL_DIR"
    )
    training_data_dir: str = Field(
        default="data/training_data",
        validation_alias="TRAINING_DATA_DIR"
    )
    use_advanced_features: bool = Field(
        default=False,
        validation_alias="USE_ADVANCED_FEATURES"
    )
    model_cache_ttl: int = Field(
        default=3600,
        validation_alias="MODEL_CACHE_TTL"
    )
    enable_ml_forecasting: bool = Field(
        default=True,
        validation_alias="ENABLE_ML_FORECASTING"
    )
    ml_min_training_samples: int = Field(
        default=30,
        validation_alias="ML_MIN_TRAINING_SAMPLES"
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="sk-test", validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_max_tokens: int = 8191
    openai_request_timeout: int = 60
    openai_max_retries: int = 3
    openai_organization: str = Field(default="", validation_alias="OPENAI_ORGANIZATION")
    assistant_timeout: int = Field(
        default=30, validation_alias="ASSISTANT_TIMEOUT", ge=1
    )
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES", ge=0)
    insights_assistant_id: Optional[str] = Field(
        default=None, validation_alias="INSIGHTS_ASSISTANT_ID"
    )
    bank_connection_assistant_id: Optional[str] = Field(
        default=None, validation_alias="BANK_CONNECTION_ASSISTANT_ID"
    )
    payment_processing_assistant_id: Optional[str] = Field(
        default=None, validation_alias="PAYMENT_PROCESSING_ASSISTANT_ID"
    )

    # Database Configuration
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="cashly_ai_vectors", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="cashly_ai", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="password", validation_alias="POSTGRES_PASSWORD")

    # Connection Pool Settings
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # Async Database Settings - Just simple fields, no circular properties!
    async_db_min_pool_size: int = Field(default=10, validation_alias="ASYNC_DB_MIN_POOL_SIZE")
    async_db_max_pool_size: int = Field(default=20, validation_alias="ASYNC_DB_MAX_POOL_SIZE")
    async_db_command_timeout: float = Field(default=10.0, validation_alias="ASYNC_DB_COMMAND_TIMEOUT")
    async_db_max_queries: int = Field(default=50000, validation_alias="ASYNC_DB_MAX_QUERIES")
    async_db_max_inactive_lifetime: float = Field(default=300.0, validation_alias="ASYNC_DB_MAX_INACTIVE_LIFETIME")
    async_db_statement_cache_size: int = Field(default=1024, validation_alias="ASYNC_DB_STATEMENT_CACHE_SIZE")

    # Testing flag
    testing: bool = Field(default=False, validation_alias="TESTING")

    # Redis Configuration (Future)
    redis_url: Optional[str] = None

    # Rails API Configuration
    rails_api_url: Optional[str] = Field(default=None, validation_alias="RAILS_API_URL")
    internal_api_key: str = "your-secure-internal-api-key"

    # Assistant IDs
    transaction_assistant_id: Optional[str] = None
    account_assistant_id: Optional[str] = None
    invoice_assistant_id: Optional[str] = None
    forecasting_assistant_id: Optional[str] = None
    budget_assistant_id: Optional[str] = None

    @field_validator("backend_cors_origins", mode="before")
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

    @property
    def asyncpg_dsn(self) -> str:
        """DSN for asyncpg, without an SQLAlchemy-specific scheme."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def connection_string(self) -> str:
        """Alias for database_url."""
        return self.database_url

    def get_pool_kwargs(self) -> dict:
        """Get asyncpg pool configuration."""
        return {
            "min_size": self.async_db_min_pool_size,
            "max_size": self.async_db_max_pool_size,
            "max_queries": self.async_db_max_queries,
            "max_inactive_connection_lifetime": self.async_db_max_inactive_lifetime,
            "command_timeout": self.async_db_command_timeout,
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()