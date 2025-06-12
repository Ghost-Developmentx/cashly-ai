# app/core/config.py
"""
Centralized configuration using Pydantic Settings.
Replaces: config/database.py, config/openai.py
"""
from fastapi import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""

    # ------------------------------------------------------------------
    # Meta / runtime environment
    # ------------------------------------------------------------------
    ENVIRONMENT: str = Field("development", validation_alias="ENVIRONMENT")
    APP_NAME: str = Field("Cashly AI Service", validation_alias="APP_NAME")
    VERSION: str = Field("0.1.0", validation_alias="VERSION")
    TESTING: bool = Field(False, validation_alias="TESTING")
    DOCKER_ENV: bool = Field(False, validation_alias="DOCKER_ENV")

    # FastAPI / Uvicorn
    API_V1_PREFIX: str = Field("/api/v1", validation_alias="API_V1_PREFIX")
    HOST: str = Field("0.0.0.0", validation_alias="HOST")
    PORT: int = Field(8000, validation_alias="PORT")
    WORKERS: int = Field(1, validation_alias="WORKERS")
    ALLOWED_ORIGINS: List[str] = Field(default_factory=list, validation_alias="ALLOWED_ORIGINS")


    # ------------------------------------------------------------------
    # OpenAI Core
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = Field(..., validation_alias="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4o", validation_alias="OPENAI_MODEL")
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL")
    OPENAI_EMBEDDING_DIMENSIONS: int = Field(1536, validation_alias="OPENAI_EMBEDDING_DIMENSIONS")
    OPENAI_MAX_TOKENS: int = Field(8191, validation_alias="OPENAI_MAX_TOKENS")
    OPENAI_REQUEST_TIMEOUT: int = Field(30, validation_alias="OPENAI_REQUEST_TIMEOUT")
    OPENAI_MAX_RETRIES: int = Field(3, validation_alias="OPENAI_MAX_RETRIES")
    OPENAI_ORGANIZATION: Optional[str] = Field(None, validation_alias="OPENAI_ORGANIZATION")

    # Assistant runtime tuning
    ASSISTANT_TIMEOUT: int = Field(30, validation_alias="ASSISTANT_TIMEOUT")
    MAX_RETRIES: int = Field(3, validation_alias="MAX_RETRIES")

    # Assistant IDs
    TRANSACTION_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="TRANSACTION_ASSISTANT_ID")
    ACCOUNT_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="ACCOUNT_ASSISTANT_ID")
    INVOICE_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="INVOICE_ASSISTANT_ID")
    FORECASTING_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="FORECASTING_ASSISTANT_ID")
    BUDGET_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="BUDGET_ASSISTANT_ID")
    INSIGHTS_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="INSIGHTS_ASSISTANT_ID")
    BANK_CONNECTION_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="BANK_CONNECTION_ASSISTANT_ID")
    PAYMENT_PROCESSING_ASSISTANT_ID: Optional[str] = Field(None, validation_alias="PAYMENT_PROCESSING_ASSISTANT_ID")

    # ------------------------------------------------------------------
    # External service endpoints
    # ------------------------------------------------------------------
    RAILS_API_URL: str = Field("http://localhost:3000", validation_alias="RAILS_API_URL")
    INTERNAL_API_KEY: str = Field(..., validation_alias="INTERNAL_API_KEY")

    # Clerk / Auth
    CLERK_JWKS_URL: str = Field(..., validation_alias="CLERK_JWKS_URL")
    CLERK_ISSUER: str = Field(..., validation_alias="CLERK_ISSUER")

    # ------------------------------------------------------------------
    # Vector PostgreSQL (Application embeddings)
    # ------------------------------------------------------------------
    POSTGRES_HOST: str = Field("localhost", validation_alias="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(5433, validation_alias="POSTGRES_PORT")
    POSTGRES_DB: str = Field("cashly_ai_vectors", validation_alias="POSTGRES_DB")
    POSTGRES_USER: str = Field("cashly_ai", validation_alias="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., validation_alias="POSTGRES_PASSWORD")

    DB_MAX_OVERFLOW: int = Field(10, validation_alias="DB_MAX_OVERFLOW")
    DB_POOL_SIZE: int = Field(20, validation_alias="DB_POOL_SIZE")
    DB_POOL_PRE_PING: bool = Field(True, validation_alias="DB_POOL_PRE_PING")

    # AsyncPG‑specific pool tuning
    ASYNC_DB_MIN_POOL_SIZE: int = Field(10, validation_alias="ASYNC_DB_MIN_POOL_SIZE")
    ASYNC_DB_MAX_POOL_SIZE: int = Field(20, validation_alias="ASYNC_DB_MAX_POOL_SIZE")
    ASYNC_DB_COMMAND_TIMEOUT: float = Field(10.0, validation_alias="ASYNC_DB_COMMAND_TIMEOUT")
    ASYNC_DB_MAX_QUERIES: int = Field(50000, validation_alias="ASYNC_DB_MAX_QUERIES")
    ASYNC_DB_MAX_INACTIVE_LIFETIME: float = Field(300.0, validation_alias="ASYNC_DB_MAX_INACTIVE_LIFETIME")
    ASYNC_DB_STATEMENT_CACHE_SIZE: int = Field(1024, validation_alias="ASYNC_DB_STATEMENT_CACHE_SIZE")

    # ------------------------------------------------------------------
    # MLflow & model registry
    # ------------------------------------------------------------------
    # Backing PostgreSQL
    MLFLOW_POSTGRES_HOST: str = Field("postgres-mlflow", validation_alias="MLFLOW_POSTGRES_HOST")
    MLFLOW_POSTGRES_PORT: int = Field(5432, validation_alias="MLFLOW_POSTGRES_PORT")
    MLFLOW_POSTGRES_DB: str = Field("mlflow_tracking", validation_alias="MLFLOW_POSTGRES_DB")
    MLFLOW_POSTGRES_USER: str = Field("cashly_ml", validation_alias="MLFLOW_POSTGRES_USER")
    MLFLOW_POSTGRES_PASSWORD: str = Field(..., validation_alias="MLFLOW_POSTGRES_PASSWORD")

    # MLflow runtime
    MLFLOW_HOST: str = Field("localhost", validation_alias="MLFLOW_HOST")
    MLFLOW_TRACKING_URI: str = Field("http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field("cashly-ai-models", validation_alias="MLFLOW_EXPERIMENT_NAME")
    MLFLOW_S3_BUCKET: str = Field("cashly-ai-models", validation_alias="MLFLOW_S3_BUCKET")

    # ------------------------------------------------------------------
    # AWS
    # ------------------------------------------------------------------
    AWS_ACCESS_KEY_ID: Optional[str] = Field(None, validation_alias="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None, validation_alias="AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION: str = Field("us-east-1", validation_alias="AWS_DEFAULT_REGION")
    MLFLOW_S3_PREFIX: str = Field("mflow", validation_alias="MLFLOW_S3_PREFIX")

    # ------------------------------------------------------------------
    # Validators & helpers
    # ------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
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
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        """Construct async database URL."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        """DSN for asyncpg, without an SQLAlchemy-specific scheme."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def connection_string(self) -> str:
        """Alias for database_url."""
        return self.database_url

    def get_pool_kwargs(self) -> dict:
        """Get asyncpg pool configuration."""
        return {
            "min_size": self.ASYNC_DB_MIN_POOL_SIZE,
            "max_size": self.ASYNC_DB_MAX_POOL_SIZE,
            "max_queries": self.ASYNC_DB_MAX_QUERIES,
            "max_inactive_connection_lifetime": self.ASYNC_DB_MAX_INACTIVE_LIFETIME,
            "command_timeout": self.ASYNC_DB_COMMAND_TIMEOUT,
        }


@lru_cache(maxsize=None)
def get_settings() -> Settings:  # pragma: no cover – trivial helper
    """Return a cached :class:`Settings` instance."""
    return Settings()


# Global settings instance
settings = get_settings()