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

                # Train model - pass the original data, not the feature_data
                # The train method will handle its own feature extraction
                self.train(data)  # Changed from self.train(feature_data)
                self.is_fitted = True

                # Log model based on type
                if self.model_type == "sklearn" and hasattr(self.model, 'predict'):
                    # For sklearn models, log directly
                    input_example = None
                    signature = None

                    if len(feature_data) > 0 and hasattr(self, 'numeric_feature_names'):
                        # Use numeric_feature_names instead of feature_names
                        if hasattr(self, 'numeric_feature_names') and self.numeric_feature_names:
                            numeric_features = feature_data[self.numeric_feature_names]
                            if len(numeric_features.columns) > 0:
                                input_example = numeric_features.head(5)
                                # Create signature with numeric features only
                                signature = infer_signature(input_example, self.model.predict(input_example))

                    model_info = mlflow.sklearn.log_model(
                        self.model,
                        artifact_path=self.model_name,
                        input_example=input_example,
                        signature=signature,
                        registered_model_name=self.model_name
                    )
                else:
                    # For custom models, create a proper wrapper
                    model_wrapper = self._create_mlflow_wrapper()

                    # Get input example
                    input_example = None
                    if len(feature_data) > 0:
                        input_example = feature_data.head(5)

                    model_info = mlflow.pyfunc.log_model(
                        artifact_path=self.model_name,
                        python_model=model_wrapper,
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

    def _create_mlflow_wrapper(self):
        """Create a proper MLflow PythonModel wrapper for custom models."""
        import mlflow.pyfunc

        model_instance = self

        class ModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self):
                self.model = model_instance

            def predict(self, context, model_input):
                """MLflow 2.x compatible predict method."""
                # Handle both DataFrame and numpy array inputs
                if isinstance(model_input, pd.DataFrame):
                    return self.model.predict(model_input)
                else:
                    # Convert numpy array to DataFrame if needed
                    df = pd.DataFrame(model_input)
                    return self.model.predict(df)

        return ModelWrapper()


    def load_latest(self, stage: str = "latest") -> bool:
        """Load the latest model from MLflow with improved strategy."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            loaded = False

            # Strategy 1: Try to load from Model Registry by stage
            if stage != "latest":
                try:
                    model_uri = f"models:/{self.model_name}/{stage}"

                    if self.model_type == "sklearn":
                        self.model = mlflow.sklearn.load_model(model_uri)
                    elif self.model_type == "pyfunc":
                        self.model = mlflow.pyfunc.load_model(model_uri)
                    else:
                        self.model = mlflow.pyfunc.load_model(model_uri)

                    self.model_version = stage
                    loaded = True
                    logger.info(f"Loaded {self.model_name} from registry stage {stage}")

                except Exception as e:
                    logger.warning(f"Could not load from registry stage {stage}: {e}")

            # Strategy 2: Try to get latest version from registry
            if not loaded:
                try:
                    versions = client.search_model_versions(
                        filter_string=f"name='{self.model_name}'",
                        order_by=["version_number DESC"],
                        max_results=1
                    )

                    if versions:
                        version = versions[0]
                        model_uri = f"models:/{self.model_name}/{version.version}"

                        if self.model_type == "sklearn":
                            self.model = mlflow.sklearn.load_model(model_uri)
                        elif self.model_type == "pyfunc":
                            self.model = mlflow.pyfunc.load_model(model_uri)
                        else:
                            self.model = mlflow.pyfunc.load_model(model_uri)

                        self.model_version = version.version
                        loaded = True
                        logger.info(f"Loaded {self.model_name} version {version.version} from registry")

                except Exception as e:
                    logger.warning(f"Could not load from registry: {e}")

            # Strategy 3: Try to load from latest run artifacts
            if not loaded:
                try:
                    # Get the default experiment
                    experiment = client.get_experiment_by_name("Default")
                    if not experiment:
                        # Try to get experiment by ID 1 (often the default)
                        experiment = client.get_experiment("1")

                    if experiment:
                        # Search for runs with this model
                        runs = client.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            filter_string=f"tags.model_name = '{self.model_name}'",
                            order_by=["start_time DESC"],
                            max_results=1
                        )

                        if not runs:
                            # Fallback: search by artifact path pattern
                            runs = client.search_runs(
                                experiment_ids=[experiment.experiment_id],
                                order_by=["start_time DESC"],
                                max_results=50  # Check more runs
                            )

                            # Filter runs that have our model as an artifact
                            for run in runs:
                                artifacts = client.list_artifacts(run.info.run_id)
                                if any(self.model_name in artifact.path for artifact in artifacts):
                                    runs = [run]
                                    break
                            else:
                                runs = []

                        if runs:
                            run = runs[0]
                            model_uri = f"runs:/{run.info.run_id}/{self.model_name}"

                            try:
                                if self.model_type == "sklearn":
                                    self.model = mlflow.sklearn.load_model(model_uri)
                                elif self.model_type == "pyfunc":
                                    self.model = mlflow.pyfunc.load_model(model_uri)
                                else:
                                    self.model = mlflow.pyfunc.load_model(model_uri)

                                self.model_version = run.info.run_id[:8]
                                loaded = True
                                logger.info(f"Loaded {self.model_name} from run {run.info.run_id}")

                            except Exception as load_error:
                                logger.warning(f"Failed to load from run {run.info.run_id}: {load_error}")

                except Exception as e:
                    logger.warning(f"Could not load from runs: {e}")

            if not loaded:
                raise Exception(f"No model found for {self.model_name}")

            return True

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            return False


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