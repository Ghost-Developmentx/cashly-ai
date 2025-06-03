"""
Model registry using MLflow for version control and deployment.
Replaces the file-based model registry.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import Model
import pandas as pd

from app.core.mlflow_config import mlflow_manager

logger = logging.getLogger(__name__)


class MLflowModelRegistry:
    """Model registry backed by MLflow."""

    def __init__(self):
        self.client = mlflow_manager.client
        self.model_cache = {}

    def save_model(
            self,
            model: Any,
            model_name: str,
            model_type: str,
            features: List[str],
            metrics: Optional[Dict[str, float]] = None,
            params: Optional[Dict[str, Any]] = None,
            tags: Optional[Dict[str, str]] = None,
            input_example: Optional[pd.DataFrame] = None
    ) -> str:
        """Save model to MLflow registry."""
        try:
            # Start MLflow run
            with mlflow_manager.start_run(
                    run_name=f"{model_name}_training",
                    tags=tags or {}
            ) as run:
                # Log parameters
                if params:
                    mlflow_manager.log_params(params)

                # Log metrics
                if metrics:
                    mlflow_manager.log_metrics(metrics)

                # Log features as artifact
                mlflow.log_dict({"features": features}, "features.json")

                # Log model type
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("model_name", model_name)

                # Log model
                model_info = mlflow_manager.log_model(
                    model=model,
                    artifact_path=model_name,
                    model_type=model_type,
                    input_example=input_example,
                    registered_model_name=model_name
                )

                logger.info(f"Model {model_name} saved with run_id: {run.info.run_id}")

                # Transition to production if no production model exists
                self._auto_promote_model(model_name)

                return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise

    def load_model(
            self,
            model_name: str,
            version: Optional[int] = None,
            stage: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load model from the MLflow registry."""
        try:
            # Check cache first
            cache_key = f"{model_name}_{version or stage or 'latest'}"
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]

            # Build model URI
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Get latest production version
                latest = mlflow_manager.get_latest_model_version(
                    model_name,
                    stage="Production"
                )
                if not latest:
                    raise ValueError(f"No production model found for {model_name}")
                model_uri = f"models:/{model_name}/{latest.version}"
                version = latest.version

            # Load model
            model = mlflow.pyfunc.load_model(model_uri)

            # Load model info
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)

            # Extract metadata
            model_info = {
                "model_name": model_name,
                "version": version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "created_at": model_version.creation_timestamp,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "model_type": run.data.tags.get("model_type", "unknown")
            }

            # Load features
            features_path = f"runs:/{model_version.run_id}/features.json"
            try:
                features = mlflow.artifacts.load_dict(features_path)
                model_info["features"] = features.get("features", [])
            except:
                model_info["features"] = []

            # Cache the model
            self.model_cache[cache_key] = (model, model_info)

            logger.info(f"Loaded {model_name} version {version}")
            return model, model_info

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models."""
        try:
            models = []

            # Get all registered models
            for rm in self.client.search_registered_models():
                # Get latest version info
                latest_versions = self.client.get_latest_versions(
                    rm.name,
                    stages=["Production", "Staging", "None"]
                )

                if latest_versions:
                    latest = latest_versions[0]
                    run = self.client.get_run(latest.run_id)

                    # Filter by model type if specified
                    if model_type and run.data.tags.get("model_type") != model_type:
                        continue

                    models.append({
                        "name": rm.name,
                        "latest_version": latest.version,
                        "stage": latest.current_stage,
                        "model_type": run.data.tags.get("model_type"),
                        "created_at": rm.creation_timestamp,
                        "updated_at": latest.creation_timestamp,
                        "description": rm.description
                    })

            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def compare_models(
            self,
            model_name: str,
            version1: int,
            version2: int
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        try:
            # Load both versions
            v1 = self.client.get_model_version(model_name, version1)
            v2 = self.client.get_model_version(model_name, version2)

            run1 = self.client.get_run(v1.run_id)
            run2 = self.client.get_run(v2.run_id)

            # Compare metrics
            metrics_diff = {}
            all_metrics = set(run1.data.metrics.keys()) | set(run2.data.metrics.keys())

            for metric in all_metrics:
                val1 = run1.data.metrics.get(metric, 0)
                val2 = run2.data.metrics.get(metric, 0)
                metrics_diff[metric] = {
                    "v1": val1,
                    "v2": val2,
                    "diff": val2 - val1,
                    "pct_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }

            return {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "metrics_comparison": metrics_diff,
                "v1_stage": v1.current_stage,
                "v2_stage": v2.current_stage,
                "v1_created": v1.creation_timestamp,
                "v2_created": v2.creation_timestamp
            }

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise

    def promote_model(
            self,
            model_name: str,
            version: int,
            stage: str,
            archive_existing: bool = True
    ):
        """Promote model to a new stage."""
        try:
            mlflow_manager.transition_model_stage(
                model_name=model_name,
                version=version,
                stage=stage,
                archive_existing=archive_existing
            )

            logger.info(f"Promoted {model_name} v{version} to {stage}")

            # Clear cache
            self.model_cache.clear()

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    def delete_model_version(self, model_name: str, version: int):
        """Delete a specific model version."""
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )

            logger.info(f"Deleted {model_name} version {version}")

            # Clear cache
            self.model_cache.clear()

        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise

    def _auto_promote_model(self, model_name: str):
        """Auto-promote model to production if no production version exists."""
        try:
            # Check if production version exists
            prod_versions = self.client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )

            if not prod_versions:
                # Get latest version
                latest_versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=["None"]
                )

                if latest_versions:
                    # Promote to production
                    self.promote_model(
                        model_name=model_name,
                        version=latest_versions[0].version,
                        stage="Production",
                        archive_existing=False
                    )

        except Exception as e:
            logger.warning(f"Auto-promotion failed for {model_name}: {e}")


# Global registry instance
mlflow_registry = MLflowModelRegistry()
