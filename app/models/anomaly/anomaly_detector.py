"""
Anomaly detection model using Isolation Forest and statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

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
        """Preprocess transaction data for anomaly detection."""
        required_cols = ['date', 'amount']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Add transaction index
        df['transaction_id'] = range(len(df))

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
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X_scaled)

        elif self.method == "statistical":
            # Calculate statistical thresholds
            self._calculate_statistical_thresholds(X_scaled)

        # Calculate anomaly scores for training data
        anomaly_scores = self._calculate_anomaly_scores(X_scaled)

        # Store components
        self.model = {
            'detector': self.model if self.method == "isolation_forest" else None,
            'scaler': self.scaler,
            'thresholds': self.thresholds,
            'transaction_extractor': self.transaction_extractor,
            'behavior_extractor': self.behavior_extractor,
            'feature_names': self.feature_names,
            'method': self.method
        }

        # Calculate metrics
        self.metrics = {
            'contamination_rate': self.contamination,
            'num_features': len(self.feature_names),
            'training_samples': len(X_scaled)
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
