import os
import logging
import sys
from flask import Flask
from flask_cors import CORS

from routes.api_router import create_api_routes
from controllers.base_controller import DateTimeEncoder
import dotenv

dotenv.load_dotenv()


def create_app() -> Flask:
    """
    Application factory pattern for creating Flask app
    """
    # Check required environment variables
    _check_environment()

    app = Flask(__name__)

    # Configure CORS
    CORS(app)

    # Configure JSON encoder for datetime objects
    app.json_encoder = DateTimeEncoder

    # Configure logging
    _configure_logging()

    # Register API routes
    api_routes = create_api_routes()
    app.register_blueprint(api_routes)

    return app


def _check_environment():
    """Check required environment variables."""
    required_vars = [
        "OPENAI_API_KEY",
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logging.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        sys.exit(1)


def _configure_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    # Create a model directory
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create and run the app
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
