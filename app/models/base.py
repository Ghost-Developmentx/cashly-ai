"""
Base classes for all ML models in Cashly AI.
Provides common functionality for model training, prediction, and MLflow integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from mlflow.models import infer_signature
import logging

from app.core.mlflow_config import mlflow_manager


logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all Cashly AI models."""

    def __init__(self, model_name: str, model_type: str = "sklearn"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.model_version = None
        self.model_uri = None
        self.metrics = {}
        self.params = {}
        self.feature_names = []
        self.is_fitted = False

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data before training or prediction."""
        pass

    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from preprocessed data."""
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        pass

    def fit(self, data: pd.DataFrame, **kwargs) -> "BaseModel":
        """Fit model with MLflow tracking."""
        with mlflow_manager.start_run(run_name=f"{self.model_name}_training") as run:
            try:
                # Log parameters
                self.params.update(kwargs)
                # Convert all params to strings for MLflow
                mlflow_params = {k: str(v) for k, v in self.params.items()}
                mlflow_manager.log_params(mlflow_params)

                # Preprocess and extract features
                processed_data = self.preprocess(data)
                feature_data = self.extract_features(processed_data)

                # Train model
                self.train(feature_data)
                self.is_fitted = True

                # Log model
                if hasattr(self, 'model') and self.model is not None:
                    # For sklearn models, log the actual model
                    if self.model_type == "sklearn" and hasattr(self.model, 'predict'):
                        model_to_log = self.model
                    else:
                        # For custom models, create a wrapper
                        model_to_log = self._create_pyfunc_wrapper()

                    # Get input example
                    if len(feature_data) > 0:
                        if isinstance(feature_data, pd.DataFrame):
                            input_example = feature_data.head(5)
                        else:
                            input_example = None
                    else:
                        input_example = None

                    model_info = mlflow_manager.log_model(
                        model=model_to_log,
                        artifact_path=self.model_name,
                        model_type=self.model_type,
                        input_example=input_example,
                        registered_model_name=self.model_name
                    )

                    self.model_uri = model_info.model_uri

                # Log metrics - ensure all metrics are floats
                if self.metrics:
                    clean_metrics = {}
                    for k, v in self.metrics.items():
                        try:
                            clean_metrics[k] = float(v)
                        except (TypeError, ValueError):
                            logger.warning(f"Skipping metric {k} with non-numeric value: {v}")

                    if clean_metrics:
                        mlflow_manager.log_metrics(clean_metrics)

                logger.info(f"Model {self.model_name} trained successfully. URI: {self.model_uri}")

            except Exception as e:
                logger.error(f"Error training model {self.model_name}: {e}")
                mlflow.end_run(status="FAILED")
                raise

        return self

    def _create_pyfunc_wrapper(self):
        """Create a PythonModel wrapper for custom models."""
        import mlflow.pyfunc

        class ModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model

            def predict(self, context, model_input):
                return self.model.predict(model_input)

        return ModelWrapper(self)

    def load_latest(self, stage: Optional[str] = None) -> "BaseModel":
        """Load latest model version from MLflow."""
        try:
            version_info = mlflow_manager.get_latest_model_version(self.model_name)

            if not version_info:
                raise ValueError(f"No model found for {self.model_name}")

            model_uri = f"models:/{self.model_name}/{version_info.version}"

            # Load the model based on type
            if self.model_type == "sklearn":
                self.model = mlflow_manager.load_model(model_uri, self.model_type)
            else:
                # For custom models, load the wrapper
                loaded_model = mlflow_manager.load_model(model_uri, "custom")
                if hasattr(loaded_model, 'model'):
                    self.model = loaded_model.model
                else:
                    self.model = loaded_model

            self.model_version = version_info.version
            self.model_uri = model_uri
            self.is_fitted = True

            logger.info(f"Loaded {self.model_name} version {self.model_version}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

        return self

    def save_predictions(self, predictions: np.ndarray, metadata: Dict[str, Any]):
        """Save predictions with metadata for analysis."""
        timestamp = datetime.now().isoformat()

        prediction_data = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": timestamp,
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "metadata": metadata
        }

        # Log to MLflow as an artifact
        with mlflow.start_run(run_name=f"{self.model_name}_prediction"):
            mlflow.log_dict(prediction_data, "predictions.json")

class BaseTransformer(ABC):
    """Base class for feature transformers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseTransformer":
        """Fit the transformer."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class BaseEvaluator(ABC):
    """Base class for model evaluation."""

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions."""
        pass

    @abstractmethod
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate evaluation report."""
        pass