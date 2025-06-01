#!/usr/bin/env python
"""
Script to set up the vector database.
"""

import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add a project root to a path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.init import DatabaseInitializer

logging.basicConfig(level=logging.INFO)


def wait_for_database(max_attempts=30, delay=1):
    """Wait for a database to be ready."""
    import psycopg2
    from config.database import DatabaseConfig

    config = DatabaseConfig.from_env()
    print(
        f"🔍 Connecting to {config.host}:{config.port} as {config.user} to {config.database}"
    )

    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=config.host,
                port=config.port,
                database="cashly_ai_vectors",
                user=config.user,
                password=config.password,
            )
            conn.close()
            print("✅ Database is ready!")
            return True
        except psycopg2.OperationalError:
            if attempt < max_attempts - 1:
                print(f"⏳ Waiting for database... ({attempt + 1}/{max_attempts})")
                time.sleep(delay)
            else:
                print("❌ Database is not responding")
                return False
    return False


def main():
    """Set up the database."""
    # Wait for a database to be ready
    if not wait_for_database(max_attempts=120, delay=1):
        print("❌ Could not connect to database!")
        sys.exit(1)

    # Initialize database
    initializer = DatabaseInitializer()

    if initializer.initialize():
        print("✅ Database setup completed successfully!")
    else:
        print("❌ Database setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
