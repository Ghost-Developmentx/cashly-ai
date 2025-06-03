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
        return cls(
            s3_bucket=os.getenv("MLFLOW_S3_BUCKET", "cashly-ai-models"),
            s3_prefix=os.getenv("MLFLOW_S3_PREFIX", "mlflow"),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "cashly-ai-models"),
            registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
            model_stage=os.getenv("MLFLOW_MODEL_STAGE", "Production")
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
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow with S3 backend."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Set S3 endpoint if using MinIO or custom S3
        if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
            os.environ["AWS_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

        # Create experiment if it doesn't exist
        try:
            mlflow.create_experiment(
                self.config.experiment_name,
                artifact_location=self.config.artifact_location
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            pass

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
        """Log model to MLflow."""
        # Map model types to MLflow flavors
        flavor_map = {
            "sklearn": mlflow.sklearn,
            "prophet": mlflow.prophet,
            "pytorch": mlflow.pytorch,
            "tensorflow": mlflow.tensorflow,
            "custom": mlflow.pyfunc
        }

        flavor = flavor_map.get(model_type, mlflow.pyfunc)

        # Log the model
        model_info = flavor.log_model(
            model,
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
            "prophet": mlflow.prophet,
            "pytorch": mlflow.pytorch,
            "tensorflow": mlflow.tensorflow,
            "custom": mlflow.pyfunc
        }

        flavor = flavor_map.get(model_type, mlflow.pyfunc)
        return flavor.load_model(model_uri)

    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None):
        """Get latest model version from registry."""
        stage = stage or self.config.model_stage

        versions = self.client.get_latest_versions(
            name=model_name,
            stages=[stage]
        )

        if versions:
            return versions[0]
        return None

    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ):
        """Transition model to a new stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )

    @staticmethod
    def log_metrics(metrics: dict, step: Optional[int] = None):
        """Log metrics to current run."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    @staticmethod
    def log_params(params: dict):
        """Log parameters to current run."""
        mlflow.log_params(params)

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