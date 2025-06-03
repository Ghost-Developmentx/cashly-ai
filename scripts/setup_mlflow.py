"""
Setup script for MLflow with S3 backend.
Run this to initialize MLflow tracking server and S3 bucket.
"""

import os
import subprocess
import boto3
from botocore.exceptions import ClientError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowSetup:
    """Set up MLflow with S3 backend storage."""

    def __init__(self):
        self.s3_bucket = os.getenv("MLFLOW_S3_BUCKET", "cashly-ai-models")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        self.postgres_uri = os.getenv(
            "MLFLOW_BACKEND_STORE_URI",
            "postgresql://cashly_ai:qwerT12321@localhost:5433/cashly_ai_mlflow"
        )

    def setup_s3_bucket(self):
        """Create S3 bucket for MLflow artifacts."""
        try:
            s3_client = boto3.client('s3', region_name=self.aws_region)

            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=self.s3_bucket)
                logger.info(f"S3 bucket {self.s3_bucket} already exists")
            except ClientError:
                # Create bucket
                if self.aws_region == 'us-east-1':
                    s3_client.create_bucket(Bucket=self.s3_bucket)
                else:
                    s3_client.create_bucket(
                        Bucket=self.s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                    )
                logger.info(f"Created S3 bucket {self.s3_bucket}")

            # Set bucket versioning
            s3_client.put_bucket_versioning(
                Bucket=self.s3_bucket,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            # Set lifecycle policy for old model versions
            lifecycle_policy = {
                'Rules': [{
                    'ID': 'DeleteOldModelVersions',
                    'Status': 'Enabled',
                    'NoncurrentVersionExpiration': {'NoncurrentDays': 90},
                    'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
                }]
            }

            s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.s3_bucket,
                LifecycleConfiguration=lifecycle_policy
            )

            logger.info("S3 bucket setup completed")

        except Exception as e:
            logger.error(f"Failed to setup S3 bucket: {e}")
            raise

    def setup_postgres_db(self):
        """Create a PostgreSQL database for MLflow backend."""
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

            # Parse connection string
            parts = self.postgres_uri.replace('postgresql://', '').split('@')
            user_pass = parts[0].split(':')
            host_db = parts[1].split('/')
            host_port = host_db[0].split(':')

            user = user_pass[0]
            password = user_pass[1]
            host = host_port[0]
            port = host_port[1] if len(host_port) > 1 else '5432'
            database = host_db[1]

            # Connect to postgres database
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if a database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database,)
            )

            if not cursor.fetchone():
                # Create database
                cursor.execute(f"CREATE DATABASE {database}")
                logger.info(f"Created database {database}")
            else:
                logger.info(f"Database {database} already exists")

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL database: {e}")
            raise

    def start_mlflow_server(self):
        """Start MLflow tracking server."""
        try:
            # Build MLflow command
            cmd = [
                "mlflow", "server",
                "--backend-store-uri", self.postgres_uri,
                "--default-artifact-root", f"s3://{self.s3_bucket}/mlflow",
                "--host", "0.0.0.0",
                "--port", self.mlflow_port,
                "--serve-artifacts"
            ]

            logger.info(f"Starting MLflow server on port {self.mlflow_port}")
            logger.info(f"Command: {' '.join(cmd)}")

            # Start server
            process = subprocess.Popen(cmd)

            logger.info(f"MLflow server started with PID {process.pid}")
            logger.info(f"Access MLflow UI at http://localhost:{self.mlflow_port}")

            # Save PID for shutdown
            with open('mlflow.pid', 'w') as f:
                f.write(str(process.pid))

            return process

        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
            raise

    def create_env_file(self):
        """Create an.env file with MLflow configuration."""
        env_content = f"""
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:{self.mlflow_port}
MLFLOW_S3_BUCKET={self.s3_bucket}
MLFLOW_S3_PREFIX=mlflow
MLFLOW_BACKEND_STORE_URI={self.postgres_uri}
MLFLOW_EXPERIMENT_NAME=cashly-ai-models
MLFLOW_MODEL_STAGE=Production

# AWS Configuration (update with your credentials)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION={self.aws_region}

# Optional: Use MinIO instead of AWS S3 for local development
# MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
"""

        # Append to existing .env or create new
        env_file = '.env.mlflow'
        with open(env_file, 'w') as f:
            f.write(env_content)

        logger.info(f"Created {env_file} - Please update with your AWS credentials")

    def setup_all(self):
        """Run complete MLflow setup."""
        logger.info("Starting MLflow setup...")

        # Create an env file
        self.create_env_file()

        # Setup S3 bucket
        try:
            self.setup_s3_bucket()
        except Exception as e:
            logger.warning(f"S3 setup skipped: {e}")
            logger.info("You can set up S3 manually or use MinIO for local development")

        # Setup PostgresSQL
        self.setup_postgres_db()

        # Start an MLflow server
        logger.info("\nSetup complete! You can now start MLflow server with:")
        logger.info(f"mlflow server --backend-store-uri {self.postgres_uri} \\")
        logger.info(f"  --default-artifact-root s3://{self.s3_bucket}/mlflow \\")
        logger.info(f"  --host 0.0.0.0 --port {self.mlflow_port} --serve-artifacts")

        logger.info("\nOr run: python scripts/start_mlflow_server.py")


if __name__ == "__main__":
    setup = MLflowSetup()
    setup.setup_all()