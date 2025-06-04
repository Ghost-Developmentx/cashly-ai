from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Config(BaseSettings):
    """Configuration settings for AI Service"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore"
    )

    # Flask Settings
    debug: bool = Field(default=False, validation_alias="DEBUG")
    port: int = Field(default=5000, validation_alias="PORT")

    # Database connection
    db_host: str = Field(default="localhost", validation_alias="DB_HOST")
    db_port: int = Field(default=5432, validation_alias="DB_PORT")
    db_name: str = Field(default="cashly_development", validation_alias="DB_NAME")
    db_user: str = Field(default="postgres", validation_alias="DB_USER")
    db_pass: str = Field(default="", validation_alias="DB_PASS")

    # Model settings
    model_dir: str = Field(default="data/trained_models", validation_alias="MODEL_DIR")
    training_data_dir: str = Field(default="data/training_data", validation_alias="TRAINING_DATA_DIR")

    # Feature settings
    use_advanced_features: bool = Field(default=False, validation_alias="USE_ADVANCED_FEATURES")

    # API Keys for external services
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    @property
    def database_uri(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
