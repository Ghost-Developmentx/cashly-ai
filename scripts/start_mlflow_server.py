"""
Start MLflow tracking server with proper configuration.
"""

import os
import subprocess
import signal
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')
load_dotenv('.env.mlflow')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowServer:
    """Manage MLflow tracking server."""

    def __init__(self):
        self.mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        self.postgres_uri = os.getenv(
            "MLFLOW_BACKEND_STORE_URI",
            "postgresql://cashly_ai:qwerT12321@localhost:5433/cashly_ai_mlflow"
        )
        self.s3_bucket = os.getenv("MLFLOW_S3_BUCKET", "cashly-ai-models")
        self.s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        self.pid_file = Path("mlflow.pid")

    def start(self):
        """Start an MLflow server."""
        # Check if already running
        if self.is_running():
            logger.warning("MLflow server is already running")
            return

        # Build command
        cmd = [
            "mlflow", "server",
            "--backend-store-uri", self.postgres_uri,
            "--default-artifact-root", f"s3://{self.s3_bucket}/mlflow",
            "--host", "0.0.0.0",
            "--port", self.mlflow_port,
            "--serve-artifacts"
        ]

        # Add S3 endpoint if using MinIO
        env = os.environ.copy()
        if self.s3_endpoint:
            env["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint
            logger.info(f"Using S3 endpoint: {self.s3_endpoint}")

        logger.info("Starting MLflow server...")
        logger.info(f"Command: {' '.join(cmd)}")

        # Start server
        process = subprocess.Popen(cmd, env=env)

        # Save PID
        self.pid_file.write_text(str(process.pid))

        logger.info(f"MLflow server started with PID {process.pid}")
        logger.info(f"Access MLflow UI at http://localhost:{self.mlflow_port}")

        # Wait for server to be ready
        import time
        import requests

        for i in range(30):
            try:
                response = requests.get(f"http://localhost:{self.mlflow_port}/health")
                if response.status_code == 200:
                    logger.info("MLflow server is ready!")
                    break
            except:
                pass
            time.sleep(1)
        else:
            logger.warning("MLflow server may not be fully ready yet")

        return process

    def stop(self):
        """Stop MLflow server."""
        if not self.is_running():
            logger.info("MLflow server is not running")
            return

        try:
            pid = int(self.pid_file.read_text())
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Stopped MLflow server (PID {pid})")
            self.pid_file.unlink()
        except Exception as e:
            logger.error(f"Failed to stop MLflow server: {e}")


    def restart(self):
        """Restart MLflow server."""
        self.stop()
        import time
        time.sleep(2)
        self.start()

    def is_running(self) -> bool:
        """Check if the MLflow server is running."""
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text())
            os.kill(pid, 0)  # Check if process exists
            return True
        except (OSError, ValueError):
            return False

    def status(self):
        """Show MLflow server status."""
        if self.is_running():
            pid = int(self.pid_file.read_text())
            logger.info(f"MLflow server is running (PID {pid})")
            logger.info(f"UI: http://localhost:{self.mlflow_port}")
        else:
            logger.info("MLflow server is not running")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage MLflow server")
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status"],
        help="Action to perform"
    )

    args = parser.parse_args()

    server = MLflowServer()

    if args.action == "start":
        server.start()
    elif args.action == "stop":
        server.stop()
    elif args.action == "restart":
        server.restart()
    elif args.action == "status":
        server.status()


if __name__ == "__main__":
    main()