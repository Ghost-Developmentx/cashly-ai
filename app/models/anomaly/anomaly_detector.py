"""
Anomaly detection model using Isolation Forest and statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.models.base import BaseModel
from app.models.anomaly.feature_engineering import (
    TransactionFeatureExtractor, BehaviorFeatureExtractor
)

logger = logging.getLogger(__name__)

class AnomalyDetector(BaseModel):
    """Model for detecting anomalous financial transactions."""

    def __init__(self, contamination: float = 0.05, method: str = "isolation_forest"):
        super().__init__(
            model_name="anomaly_detector",
            model_type="sklearn"
        )
        self.contamination = contamination
        self.method = method
        self.transaction_extractor = TransactionFeatureExtractor()
        self.behavior_extractor = BehaviorFeatureExtractor()
        self.scaler = StandardScaler()
        self.thresholds = {}

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for anomaly detection."""
        df = data.copy()

        # Handle categorical columns like description
        if 'description' in df.columns:
            # Use label encoding or drop the column if not needed
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['description_encoded'] = le.fit_transform(df['description'].fillna('unknown'))
            df = df.drop('description', axis=1)

        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df = df.drop(col, axis=1)

        # Fill any remaining NaN values
        df = df.fillna(0)

        return df


    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract anomaly detection features."""
        df = data.copy()

        # Extract transaction features
        df = self.transaction_extractor.fit_transform(df)

        # Extract behavioral features
        df = self.behavior_extractor.fit_transform(df)

        # Store feature names
        self.feature_names = [
            col for col in df.columns
            if col not in ['date', 'amount', 'category', 'description', 'transaction_id']
        ]

        return df



    def _calculate_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate anomaly scores for the input data."""
        # Handle both direct model and dictionary storage
        if isinstance(self.model, dict):
            # Model is stored as dictionary (after training)
            detector = self.model.get('detector')
            if detector is None and self.model.get('method') == 'statistical':
                # Statistical method - use thresholds
                scores = []
                thresholds = self.model.get('thresholds', {})
                for i, feature in enumerate(self.model.get('feature_names', [])):
                    if feature in thresholds:
                        feature_data = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i]
                        threshold_info = thresholds[feature]
                        # Calculate deviation from median in MAD units
                        deviation = np.abs(feature_data - threshold_info['median']) / (threshold_info['mad'] + 1e-8)
                        scores.append(deviation)
                return np.mean(scores, axis=0) if scores else np.zeros(len(X))
            elif detector is not None:
                # Use the stored detector
                model_to_use = detector
            else:
                raise ValueError("No trained detector found in model")
        else:
            # Direct model (during training)
            model_to_use = self.model

        if model_to_use is None:
            raise ValueError("Model must be fitted before calculating scores")

        # Use the model to predict anomaly scores
        if hasattr(model_to_use, 'decision_function'):
            # For models like IsolationForest, OneClassSVM
            scores = model_to_use.decision_function(X)
            # Convert to positive anomaly scores (higher = more anomalous)
            scores = -scores
        elif hasattr(model_to_use, 'score_samples'):
            # For models like LocalOutlierFactor
            scores = -model_to_use.score_samples(X)
        else:
            # Fallback: use predict_proba if available
            if hasattr(model_to_use, 'predict_proba'):
                proba = model_to_use.predict_proba(X)
                # Assume binary classification: anomaly score = P(anomaly)
                scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                # Last resort: distance-based scoring
                scores = np.linalg.norm(X - X.mean(), axis=1)

        return scores


    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AnomalyDetector":
        """Train the anomaly detection model."""
        # Extract features if needed
        if 'amount' in X.columns:
            X_features = self.extract_features(X)
            feature_data = X_features[self.feature_names]
        else:
            feature_data = X

        # Scale features
        X_scaled = self.scaler.fit_transform(feature_data)

        # Train model based on method
        trained_detector = None
        if self.method == "isolation_forest":
            trained_detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            trained_detector.fit(X_scaled)

            # Temporarily set the model for score calculation
            self.model = trained_detector

        elif self.method == "statistical":
            # Calculate statistical thresholds
            self._calculate_statistical_thresholds(X_scaled)

        # Calculate anomaly scores for training data (now model is set)
        anomaly_scores = self._calculate_anomaly_scores(X_scaled)

        # Now store all components in the model dictionary
        self.model = {
            'detector': trained_detector,
            'scaler': self.scaler,
            'thresholds': self.thresholds,
            'transaction_extractor': self.transaction_extractor,
            'behavior_extractor': self.behavior_extractor,
            'feature_names': self.feature_names,
            'method': self.method
        }

        # Mark as fitted
        self.is_fitted = True

        # Calculate metrics
        self.metrics = {
            'contamination_rate': self.contamination,
            'num_features': len(self.feature_names),
            'training_samples': len(X_scaled),
            'mean_anomaly_score': float(np.mean(anomaly_scores)),
            'std_anomaly_score': float(np.std(anomaly_scores))
        }

        return self

    def _calculate_statistical_thresholds(self, X_scaled: np.ndarray):
        """Calculate statistical thresholds for anomaly detection."""
        # Calculate z-score thresholds for each feature
        for i, feature in enumerate(self.feature_names):
            feature_data = X_scaled[:, i]

            # Robust statistics using median and MAD
            median = np.median(feature_data)
            mad = np.median(np.abs(feature_data - median))

            self.thresholds[feature] = {
                'median': median,
                'mad': mad,
                'lower': median - 3 * mad,
                'upper': median + 3 * mad
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Extract features if needed
        if 'amount' in X.columns:
            X_features = self.extract_features(X)
            feature_data = X_features[self.feature_names]
        else:
            feature_data = X

        # Scale features
        X_scaled = self.scaler.transform(feature_data)

        if self.method == "isolation_forest":
            return self.model['detector'].predict(X_scaled)
        else:
            # Statistical method
            scores = self._calculate_anomaly_scores(X_scaled)
            threshold = np.percentile(scores, self.contamination * 100)
            return np.where(scores < threshold, -1, 1)

    def predict_with_details(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict anomalies with detailed explanations."""
        # Get predictions
        predictions = self.predict(X)
        scores = self.score_samples(X)

        # Extract features for analysis
        if 'amount' in X.columns:
            X_features = self.extract_features(X)
        else:
            X_features = X

        results = []

        for i in range(len(X)):
            is_anomaly = predictions[i] == -1

            result = {
                'is_anomaly': is_anomaly,
                'anomaly_score': float(scores[i]),
                'confidence': self._calculate_confidence(scores[i]),
                'anomaly_type': self._determine_anomaly_type(X_features.iloc[i]) if is_anomaly else None,
                'explanation': self._generate_explanation(X_features.iloc[i]) if is_anomaly else None
            }

            results.append(result)

        return results

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores for samples."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Extract and scale features
        if 'amount' in X.columns:
            X_features = self.extract_features(X)
            feature_data = X_features[self.feature_names]
        else:
            feature_data = X

        X_scaled = self.scaler.transform(feature_data)

        return self._calculate_anomaly_scores(X_scaled)

    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence in anomaly detection."""
        # Normalize score to 0-1 range
        if hasattr(self, 'training_scores'):
            min_score = np.min(self.training_scores)
            max_score = np.max(self.training_scores)
            normalized = (score - min_score) / (max_score - min_score + 1e-8)
        else:
            # Use a default normalization
            normalized = 1 / (1 + np.exp(-abs(score)))

        return float(np.clip(normalized, 0, 1))

    @staticmethod
    def _determine_anomaly_type(transaction: pd.Series) -> str:
        """Determine the type of anomaly."""
        anomaly_types = []

        # Amount anomaly
        if 'amount_zscore' in transaction and abs(transaction['amount_zscore']) > 3:
            anomaly_types.append('unusual_amount')

        # Timing anomaly
        if 'hour' in transaction and (transaction['hour'] < 6 or transaction['hour'] > 22):
            anomaly_types.append('unusual_timing')

        # Frequency anomaly
        if 'daily_transaction_count' in transaction and transaction['daily_transaction_count'] > 10:
            anomaly_types.append('high_frequency')

        # Pattern anomaly
        if 'days_since_last' in transaction and transaction['days_since_last'] > 30:
            anomaly_types.append('unusual_gap')

        return ', '.join(anomaly_types) if anomaly_types else 'complex_pattern'

    @staticmethod
    def _generate_explanation(transaction: pd.Series) -> str:
        """Generate human-readable explanation for anomaly."""
        explanations = []

        # Amount-based explanation
        if 'amount_zscore' in transaction and abs(transaction['amount_zscore']) > 3:
            if transaction['amount_zscore'] > 0:
                explanations.append(
                    f"Transaction amount is unusually large ({transaction['amount_zscore']:.1f} standard deviations above normal)"
                )
            else:
                explanations.append(
                    f"Transaction amount is unusually small ({abs(transaction['amount_zscore']):.1f} standard deviations below normal)"
                )

        # Time-based explanation
        if 'hour' in transaction:
            if transaction['hour'] < 6:
                explanations.append(f"Transaction occurred unusually early ({transaction['hour']}:00)")
            elif transaction['hour'] > 22:
                explanations.append(f"Transaction occurred unusually late ({transaction['hour']}:00)")

        # Frequency-based explanation
        if 'daily_transaction_count' in transaction and transaction['daily_transaction_count'] > 10:
            explanations.append(
                f"High number of transactions on this day ({int(transaction['daily_transaction_count'])} transactions)"
            )

        # Gap-based explanation
        if 'days_since_last' in transaction and transaction['days_since_last'] > 30:
            explanations.append(
                f"Long gap since last transaction ({int(transaction['days_since_last'])} days)"
            )

        return '; '.join(explanations) if explanations else "Unusual transaction pattern detected"

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate anomaly detection performance."""
        y_pred = self.predict(X)

        # Convert to binary (1 for anomaly, 0 for normal)
        y_true_binary = (y == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)

        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positive_rate': float(recall),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        }
