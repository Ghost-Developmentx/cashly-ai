"""
Test MLflow integration with the new setup.
"""

import os
import sys
import mlflow
from datetime import datetime
from dotenv import load_dotenv

# Load AWS credentials from .env.docker
load_dotenv(".env.docker")


# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.mlflow_config import mlflow_manager


def test_mlflow_connection():
    """Test MLflow connectivity and basic operations."""
    print("🧪 Testing MLflow Integration...")

    try:
        # Initialize MLflow
        mlflow_manager.initialize()
        print("✅ MLflow initialized")

        # Create a test run
        with mlflow.start_run(run_name="test_run"):
            # Log metrics
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_param("test_param", "value")

            # Log artifact
            with open("test_artifact.txt", "w") as f:
                f.write(f"Test run at {datetime.now()}")
            mlflow.log_artifact("test_artifact.txt")

            run_id = mlflow.active_run().info.run_id
            print(f"✅ Created test run: {run_id}")

        # List experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Found {len(experiments)} experiments")

        # Cleanup
        os.remove("test_artifact.txt")

        print("\n🎉 MLflow integration test passed!")
        print(f"View results at: http://localhost:5000/#/experiments")

    except Exception as e:
        print(f"❌ MLflow test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    test_mlflow_connection()