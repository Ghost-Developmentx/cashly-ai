import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for AI Service"""

    # Flask Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))

    # Database connection
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "cashly_development")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS", "")

    # Model settings
    MODEL_DIR = os.getenv("MODEL_DIR", "data/trained_models")
    TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", "data/training_data")

    # Feature settings
    USE_ADVANCED_FEATURES = (
        os.getenv("USE_ADVANCED_FEATURES", "False").lower() == "true"
    )

    # API Keys for external services (if needed)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    @property
    def DATABASE_URI(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


# Create a configuration instance
config = Config()
