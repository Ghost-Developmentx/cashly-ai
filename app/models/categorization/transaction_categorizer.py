"""
Transaction categorization model using ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from app.models.base import BaseModel
from app.models.categorization.feature_engineering import (
    TextFeatureExtractor,
    MerchantFeatureExtractor,
    AmountFeatureExtractor
)


logger = logging.getLogger(__name__)

class TransactionCategorizer(BaseModel):
    """Model for categorizing financial transactions."""

    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__(
            model_name="transaction_categorizer",
            model_type="sklearn"
        )
        self.confidence_threshold = confidence_threshold
        self.label_encoder = LabelEncoder()
        self.text_extractor = TextFeatureExtractor()
        self.merchant_extractor = MerchantFeatureExtractor()
        self.amount_extractor = AmountFeatureExtractor()
        self.categories = []
        self.numeric_feature_names = []  # Track numeric features only

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess transaction data."""
        # Ensure required columns
        required_cols = ['description', 'amount']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        df = data.copy()

        # Fill missing values
        df['description'] = df['description'].fillna('')
        df['amount'] = df['amount'].fillna(0)

        # Add temporal features if date is available
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        else:
            # Add dummy temporal features
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['month'] = 1
            df['is_weekend'] = 0

        return df

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features for categorization."""
        df = data.copy()

        # Ensure we have the required columns
        if 'description' not in df.columns:
            raise ValueError("Data must contain 'description' column")

        # Extract text features
        df = self.text_extractor.fit_transform(df)

        # Extract merchant features
        df = self.merchant_extractor.transform(df)

        # Extract amount features
        df = self.amount_extractor.transform(df)

        # Store feature names - ensure we have features
        self.feature_names = [
            col for col in df.columns
            if col not in ['description', 'amount', 'date', 'category',
                           'merchant_name', 'amount_category', 'type']
               and not col.startswith('Unnamed')  # Skip index columns
        ]

        # FIXED: Populate numeric_feature_names with actual numeric features
        # Check each column individually, not the whole DataFrame
        self.numeric_feature_names = []
        for col in self.feature_names:
            if col in df.columns:  # Ensure column exists
                col_dtype = df[col].dtype
                if col_dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'int8', 'int16', 'float16']:
                    self.numeric_feature_names.append(col)

        if not self.feature_names:
            raise ValueError("No features extracted from data")

        if not self.numeric_feature_names:
            raise ValueError("No numeric features extracted from data")

        return df

    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransactionCategorizer":
        """Train the categorization model."""
        # Extract features first to populate numeric_feature_names
        X_processed = self.extract_features(X)

        # Extract target
        if y is None:
            if 'category' not in X.columns:
                raise ValueError("Training data must include 'category' column")
            y = X['category']

        # Use only numeric features for training
        X_features = X_processed[self.numeric_feature_names]

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.categories = self.label_encoder.classes_.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'categories': len(self.categories)
        }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict categories for transactions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Extract features if raw data provided
        if 'description' in X.columns:
            X_preprocessed = self.preprocess(X)
            X_features = self.extract_features(X_preprocessed)[self.numeric_feature_names]
        else:
            # Ensure we only use numeric features
            X_features = X[self.numeric_feature_names]

        # Make predictions
        y_pred_encoded = self.model.predict(X_features)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each category."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Extract features if raw data provided
        if 'description' in X.columns:
            X_preprocessed = self.preprocess(X)
            X_features = self.extract_features(X_preprocessed)[self.numeric_feature_names]
        else:
            X_features = X[self.numeric_feature_names]

        # Get probabilities
        return self.model.predict_proba(X_features)

    def predict_with_confidence(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict categories with confidence scores and alternatives."""
        # Get predictions and probabilities
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        results = []

        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Get confidence of predicted category
            max_prob_idx = np.argmax(probs)
            confidence = float(probs[max_prob_idx])

            # Get alternative categories
            sorted_indices = np.argsort(probs)[::-1]
            alternatives = []

            for idx in sorted_indices[1:4]:  # Top 3 alternatives
                if probs[idx] > 0.05:  # Only include if > 5% probability
                    alternatives.append({
                        'category': self.categories[idx],
                        'confidence': float(probs[idx])
                    })

            results.append({
                'category': pred,
                'confidence': confidence,
                'is_confident': confidence >= self.confidence_threshold,
                'alternatives': alternatives
            })

        return results

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(X)

        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted')
        }

        # Per-category metrics
        report = classification_report(y, y_pred, output_dict=True)

        # Add category-specific F1 scores
        category_f1 = {}
        for category in self.categories:
            if category in report:
                category_f1[f'f1_{category}'] = report[category]['f1-score']

        metrics.update(category_f1)

        return metrics


    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get feature importances
        importances = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.numeric_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)