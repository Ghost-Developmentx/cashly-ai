import os
import joblib
import json
from datetime import datetime
import hashlib
import pandas as pd
from ai_config import config


class ModelRegistry:
    """
    Manages model versioning, storage, and retrieval
    """

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize the registry
        self.registry_file = os.path.join(self.model_dir, "registry.json")
        self.registry = self._load_registry()

    def _load_registry(self):
        """Load the model registry from disk"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Save the model registry to disk"""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    @staticmethod
    def _generate_model_id(model_name, model_type, features):
        """Generate a unique model ID based on name, type, and features"""
        features_str = str(sorted(features))
        unique_string = (
            f"{model_name}_{model_type}_{features_str}_{datetime.now().isoformat()}"
        )
        return hashlib.md5(unique_string.encode()).hexdigest()[:10]

    def save_model(
        self,
        model,
        model_name,
        model_type,
        features,
        metrics=None,
        metadata=None,
        keep_latest=3,
    ):
        """
        Save a model to the registry

        Args:
            model: The trained model object
            model_name: Name of the model
            model_type: Type of model (e.g., 'categorization', 'forecasting')
            features: List of features used to train the model
            metrics: Dictionary of evaluation metrics
            metadata: Additional metadata about the model
            keep_latest: Number of latest models to keep (0 to disable cleanup)

        Returns:
            model_id: The unique ID of the saved model
        """
        # Generate a unique ID for the model
        model_id = self._generate_model_id(model_name, model_type, features)

        # Create model metadata
        model_info = {
            "id": model_id,
            "name": model_name,
            "type": model_type,
            "features": features,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "metadata": metadata or {},
            "file_path": f"{model_id}.joblib",
        }

        # Save the model to disk
        model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
        joblib.dump(model, model_path)

        # Update the registry
        if model_type not in self.registry:
            self.registry[model_type] = {}

        self.registry[model_type][model_id] = model_info
        self._save_registry()

        # Clean up old models if requested
        if keep_latest > 0:
            self.cleanup_old_models(model_type, keep_latest)

        return model_id

    def load_model(self, model_id=None, model_type=None, latest=False):
        """
        Load a model from the registry

        Args:
            model_id: The unique ID of the model to load
            model_type: The type of model to load
            latest: Whether to load the latest model of the given type

        Returns:
            model: The loaded model object
            model_info: The model metadata
        """
        # If model_id is provided, load that specific model
        if model_id:
            # Find the model in the registry
            for type_name, models in self.registry.items():
                if model_id in models:
                    model_info = models[model_id]
                    model_path = os.path.join(self.model_dir, model_info["file_path"])
                    model = joblib.load(model_path)
                    return model, model_info

            raise ValueError(f"Model with ID {model_id} not found in registry")

        # If model_type is provided, load the latest model of that type
        elif model_type and latest:
            if model_type not in self.registry:
                raise ValueError(f"No models of type {model_type} found in registry")

            # Get all models of the specified type
            models = self.registry[model_type]

            if not models:
                raise ValueError(f"No models of type {model_type} found in registry")

            # Find the latest model based on created_at
            latest_model_id = max(models, key=lambda id: models[id]["created_at"])
            model_info = models[latest_model_id]
            model_path = os.path.join(self.model_dir, model_info["file_path"])
            model = joblib.load(model_path)

            return model, model_info

        else:
            raise ValueError(
                "Must provide either model_id or (model_type and latest=True)"
            )

    def list_models(self, model_type=None):
        """
        List all models in the registry, optionally filtered by type

        Args:
            model_type: The type of model to list

        Returns:
            list: List of model metadata
        """
        if model_type:
            if model_type not in self.registry:
                return []
            return list(self.registry[model_type].values())

        # Return all models
        all_models = []
        for type_name, models in self.registry.items():
            all_models.extend(list(models.values()))

        return all_models

    def delete_model(self, model_id):
        """
        Delete a model from the registry

        Args:
            model_id: The unique ID of the model to delete

        Returns:
            bool: Whether the deletion was successful
        """
        # Find the model in the registry
        for type_name, models in self.registry.items():
            if model_id in models:
                model_info = models[model_id]

                # Delete the model file
                model_path = os.path.join(self.model_dir, model_info["file_path"])
                if os.path.exists(model_path):
                    os.remove(model_path)

                # Remove from registry
                del self.registry[type_name][model_id]
                self._save_registry()
                return True

        return False

    def cleanup_old_models(self, model_type, keep_latest=3):
        """
        Keep only the latest N models of a given type

        Args:
            model_type: The type of model to cleanup
            keep_latest: Number of latest models to keep

        Returns:
            int: Number of models deleted
        """
        if model_type not in self.registry:
            return 0

        # Get all models of the specified type
        models = self.registry[model_type]

        # Sort models by creation date (newest first)
        sorted_model_ids = sorted(
            models.keys(), key=lambda id: models[id]["created_at"], reverse=True
        )

        # Keep track of deleted models
        deleted_count = 0

        # Delete older models beyond our keep limit
        if len(sorted_model_ids) > keep_latest:
            for old_model_id in sorted_model_ids[keep_latest:]:
                model_info = models[old_model_id]

                # Delete model file
                model_path = os.path.join(self.model_dir, model_info["file_path"])
                if os.path.exists(model_path):
                    os.remove(model_path)

                # Remove from registry
                del self.registry[model_type][old_model_id]
                deleted_count += 1

            # Save updated registry
            self._save_registry()

        return deleted_count
