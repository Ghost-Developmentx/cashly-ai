"""
MLflow configuration for model management.
Handles S3 backend storage and experiment tracking.
"""

import os
from typing import Optional
from dataclasses import dataclass
import mlflow
from mlflow.tracking import MlflowClient
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

@dataclass
class MLflowConfig:
    """MLflow configuration settings."""

    # S3 settings
    s3_bucket: str
    s3_prefix: str = "mlflow"
    aws_region: str = "us-east-1"

    # MLflow settings
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "cashly-ai-models"
    registry_uri: Optional[str] = None

    # Model settings
    model_stage: str = "Production"  # Staging, Production, Archived

    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Create config from environment variables."""
        # Use container name when running in Docker
        mlflow_host = os.getenv("MLFLOW_HOST", "localhost")
        if os.getenv("DOCKER_ENV"):
            mlflow_host = "localhost"

        return cls(
            s3_bucket=os.getenv("MLFLOW_S3_BUCKET", "cashly-ai-models"),
            s3_prefix=os.getenv("MLFLOW_S3_PREFIX", "mlflow"),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            tracking_uri=f"http://{mlflow_host}:5000",
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "cashly-ai-models"),
            registry_uri=os.getenv("MLFLOW_REGISTRY_URI")
        )

    @property
    def artifact_location(self) -> str:
        """Get S3 artifact location."""
        return f"s3://{self.s3_bucket}/{self.s3_prefix}"

class MLflowManager:
    """Manages MLflow operations and model lifecycle."""

    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or MLflowConfig.from_env()
        self.client = None
        self._initialized = False

    def initialize(self):
        """Initialize MLflow connection (lazy loading)."""
        if self._initialized:
            return

        try:
            self._setup_mlflow()
            self._initialized = True
            logger.info("MLflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            # Don't raise - allow the app to start without MLflow

    def _setup_mlflow(self):
        """Initialize MLflow with S3 backend."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Set S3 endpoint if using MinIO or custom S3
        if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
            os.environ["AWS_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location
                )
        except Exception as e:
            logger.warning(f"Could not create experiment: {e}")

        # Set experiment as active
        mlflow.set_experiment(self.config.experiment_name)

        # Initialize client
        self.client = MlflowClient(
            tracking_uri=self.config.tracking_uri,
            registry_uri=self.config.registry_uri
        )

    @staticmethod
    def start_run(run_name: str, tags: Optional[dict] = None):
        """Start a new MLflow run."""
        return mlflow.start_run(
            run_name=run_name,
            tags=tags or {}
        )

    @staticmethod
    def log_model(
            model,
            artifact_path: str,
            model_type: str,
            input_example=None,
            signature=None,
            registered_model_name: Optional[str] = None
    ):
        """Log model to MLflow - FIXED version."""
        # This method is now only used by BaseModel which passes
        # the model directly for sklearn models
        # The BaseModel class handles the logging itself
        # This is kept for backward compatibility

        # Map model types to MLflow flavors
        flavor_map = {
            "sklearn": mlflow.sklearn,
            "custom": mlflow.pyfunc
        }

        flavor = flavor_map.get(model_type, mlflow.pyfunc)

        # Log the model with proper kwargs
        model_info = flavor.log_model(
            sk_model=model if model_type == "sklearn" else None,
            python_model=model if model_type == "custom" else None,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
            registered_model_name=registered_model_name
        )

        return model_info

    @staticmethod
    def load_model(model_uri: str, model_type: str = "sklearn"):
        """Load model from MLflow."""
        flavor_map = {
            "sklearn": mlflow.sklearn,
            "custom": mlflow.pyfunc
        }

        flavor = flavor_map.get(model_type, mlflow.pyfunc)
        return flavor.load_model(model_uri)

    def get_latest_model_version(self, model_name: str):
        """Get latest model version (updated for new MLflow API)."""
        try:
            # Use search_model_versions instead of deprecated get_latest_versions
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'",
                order_by=["version_number DESC"],
                max_results=1
            )

            if versions:
                return versions[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None

    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """Transition model to a new stage (updated for new API)."""
        try:
            # Use set_model_version_tag instead of deprecated transition_model_version_stage
            self.client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key="stage",
                value=stage
            )
            logger.info(f"Set {model_name} v{version} stage tag to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")

    @staticmethod
    def log_metrics(metrics: dict, step: Optional[int] = None):
        """Log metrics to current run."""
        for key, value in metrics.items():
            try:
                # Ensure value is numeric
                numeric_value = float(value)
                mlflow.log_metric(key, numeric_value, step=step)
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-numeric metric {key}: {value}")

    @staticmethod
    def log_params(params: dict):
        """Log parameters to current run."""
        # MLflow requires string values
        string_params = {}
        for key, value in params.items():
            string_params[key] = str(value)
        mlflow.log_params(string_params)

    @staticmethod
    def log_artifacts(local_path: str, artifact_path: Optional[str] = None):
        """Log artifacts to current run."""
        mlflow.log_artifacts(local_path, artifact_path)

    def search_runs(self, experiment_name: Optional[str] = None, filter_string: str = ""):
        """Search for runs in experiment."""
        experiment_name = experiment_name or self.config.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if not experiment:
            return []

        return mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string
        )

mlflow_manager = MLflowManager()