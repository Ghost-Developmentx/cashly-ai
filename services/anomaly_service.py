import pandas as pd
import numpy as np
from datetime import datetime
from models.anomaly import AnomalyDetector
from util.model_registry import ModelRegistry


class AnomalyService:
    """
    Service for detecting anomalous transactions and spending patterns
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.detector = AnomalyDetector(registry=self.registry)

        # Try to load existing model
        try:
            self.detector.model, model_info = self.registry.load_model(
                model_type="anomaly_detection", latest=True
            )
            self.detector.global_model = self.detector.model["global_model"]
            self.detector.category_models = self.detector.model["category_models"]
            self.detector.scaler = self.detector.model["scaler"]
            self.detector.feature_cols = self.detector.model["feature_cols"]
            self.detector.model_id = model_info["id"]
            print(f"Loaded anomaly detection model: {model_info['id']}")
        except Exception as e:
            print(f"No existing anomaly detection model found: {str(e)}")

    def detect_anomalies(self, user_id, transactions, threshold=None):
        """
        Detect anomalous transactions

        Args:
            user_id: User ID
            transactions: List of transaction dictionaries
            threshold: Optional anomaly score threshold

        Returns:
            dict: Anomaly detection results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # If detector model doesn't exist, train a new one
        if self.detector.model is None:
            try:
                self.detector.fit(df)
            except Exception as e:
                print(f"Error training anomaly detection model: {str(e)}")
                return {
                    "user_id": user_id,
                    "anomalies": [],
                    "error": "Could not train anomaly detection model",
                }

        # Detect anomalies
        try:
            # Apply threshold if provided
            if threshold is not None:
                anomaly_df = self.detector.detect_anomalies(df, threshold=threshold)
            else:
                anomaly_df = self.detector.detect_anomalies(df)

            # Extract anomalies
            anomalies = []
            if "is_anomaly" in anomaly_df.columns:
                anomaly_rows = anomaly_df[anomaly_df["is_anomaly"]]

                for _, row in anomaly_rows.iterrows():
                    anomaly = {
                        "transaction_id": str(row.get("id", row.name)),
                        "date": (
                            row["date"]
                            if isinstance(row["date"], str)
                            else row["date"].strftime("%Y-%m-%d")
                        ),
                        "amount": float(row["amount"]),
                        "description": row.get("description", ""),
                        "category": row.get("category", "unknown"),
                        "anomaly_score": float(row["anomaly_score"]),
                        "anomaly_confidence": float(row["anomaly_confidence"]),
                        "anomaly_type": row["anomaly_type"],
                        "explanation": row["anomaly_explanation"],
                    }
                    anomalies.append(anomaly)

            # Summarize anomalies
            summary = {
                "total_transactions": len(df),
                "anomaly_count": len(anomalies),
                "anomaly_percentage": (
                    round(len(anomalies) / len(df) * 100, 1) if len(df) > 0 else 0
                ),
                "categories_with_anomalies": len(set(a["category"] for a in anomalies)),
            }

            return {"user_id": user_id, "anomalies": anomalies, "summary": summary}

        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return {"user_id": user_id, "anomalies": [], "error": str(e)}

    def train_anomaly_model(self, transactions_data):
        """
        Train a new anomaly detection model

        Args:
            transactions_data: List of transaction dictionaries

        Returns:
            dict: Training results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Train the model
        try:
            self.detector = AnomalyDetector(registry=self.registry)
            self.detector.fit(df)

            return {
                "success": True,
                "model_id": self.detector.model_id,
                "message": "Anomaly detection model trained successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_anomaly_model(self, transactions_data):
        """
        Update the anomaly detection model with new transaction data

        Args:
            transactions_data: List of transaction dictionaries

        Returns:
            dict: Update results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # If detector model doesn't exist, train a new one
        if self.detector.model is None:
            try:
                self.detector.fit(df)
                return {
                    "success": True,
                    "model_id": self.detector.model_id,
                    "message": "New anomaly detection model trained successfully",
                }
            except Exception as e:
                print(f"Error training anomaly detection model: {str(e)}")
                return {"success": False, "error": str(e)}

        # Update existing model
        try:
            # Update the model
            previous_id = self.detector.model_id
            self.detector.update_model(df)

            # Get information about what was updated
            new_categories = []
            model_info = next(
                (
                    m
                    for m in self.registry.list_models("anomaly_detection")
                    if m["id"] == self.detector.model_id
                ),
                {},
            )

            if model_info.get("metadata", {}).get("new_categories"):
                new_categories = model_info["metadata"]["new_categories"]

            return {
                "success": True,
                "model_id": self.detector.model_id,
                "previous_model_id": previous_id,
                "message": "Anomaly detection model updated successfully",
                "new_categories": new_categories,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
