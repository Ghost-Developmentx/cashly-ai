from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.utils.data_processing import clean_transaction_description
from app.utils.model_registry import ModelRegistry


class DescriptionExtractor(BaseEstimator, TransformerMixin):
    """Extracts and cleans transaction descriptions"""

    def __init__(self, description_col="description"):
        self.description_col = description_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            descriptions = X[self.description_col].fillna("")
        else:
            descriptions = X
        return descriptions.apply(clean_transaction_description)


class TransactionCategorizer:
    """
    ML model for categorizing financial transactions based on description and metadata
    """

    def __init__(self, registry=None):
        self.model = None
        self.registry = registry or ModelRegistry()
        self.categories = None
        self.model_id = None
        self.feature_names = None

    def build_pipeline(self, numerical_features, categorical_features):
        """
        Build the ML pipeline for transaction categorization

        Args:
            numerical_features: List of numerical feature columns
            categorical_features: List of categorical feature columns

        Returns:
            sklearn.pipeline.Pipeline: The ML pipeline
        """
        # Text processing pipeline for transaction descriptions
        text_pipeline = Pipeline(
            [
                ("extractor", DescriptionExtractor()),
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2),
                ),
            ]
        )

        # Preprocessing for numerical features
        numerical_transformer = Pipeline([("scaler", StandardScaler())])

        # Preprocessing for categorical features
        categorical_transformer = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        # Combine text and numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_pipeline, "description"),
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        # Full pipeline with classifier
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    GradientBoostingClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )

        return pipeline

    def fit(self, transactions_df, categories=None):
        """
        Train the categorization model

        Args:
            transactions_df: DataFrame with transaction data
            categories: Optional list of known categories

        Returns:
            self: The trained model
        """
        # Extract features and target
        X = (
            transactions_df.drop("category", axis=1)
            if "category" in transactions_df.columns
            else transactions_df
        )
        y = (
            transactions_df["category"]
            if "category" in transactions_df.columns
            else None
        )

        if y is None and categories is None:
            raise ValueError(
                "Either 'category' column must be present in transactions_df or categories must be provided"
            )

        # Store categories
        self.categories = categories if categories is not None else sorted(y.unique())

        # Define features
        numerical_features = ["amount", "day_of_week", "day_of_month", "month"]
        categorical_features = []

        # Add features if they exist in the dataframe
        for col in ["is_weekend", "amount_abs", "is_expense", "is_income"]:
            if col in X.columns:
                numerical_features.append(col)

        # Make sure required features exist
        for feature in ["description", "amount"]:
            if feature not in X.columns:
                raise ValueError(
                    f"Required feature '{feature}' not found in transactions_df"
                )

        # Build and fit the pipeline
        self.model = self.build_pipeline(numerical_features, categorical_features)
        self.feature_names = numerical_features + categorical_features + ["description"]

        # If we have labels, fit the model
        if y is not None:
            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit the model
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            }

            print("Model performance:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
            print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")

            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Save the model to registry
            self.model_id = self.registry.save_model(
                model=self.model,
                model_name="transaction_categorizer",
                model_type="categorization",
                features=self.feature_names,
                metrics=metrics,
                metadata={"categories": self.categories},
            )

        return self

    def predict(self, transactions):
        """
        Predict categories for new transactions

        Args:
            transactions: DataFrame with transaction data or single transaction dict

        Returns:
            predictions: List of predicted categories
        """
        if self.model is None:
            # Try to load the latest model
            self.model, model_info = self.registry.load_model(
                model_type="categorization", latest=True
            )
            self.categories = model_info["metadata"]["categories"]
            self.feature_names = model_info["features"]
            self.model_id = model_info["id"]

        # Convert single transaction to DataFrame if needed
        if isinstance(transactions, dict):
            transactions = pd.DataFrame([transactions])

        # Ensure all required features are present
        for feature in ["description", "amount"]:
            if feature not in transactions.columns:
                raise ValueError(
                    f"Required feature '{feature}' not found in transactions"
                )

        # Add date-based features if not present
        if "day_of_week" not in transactions.columns and "date" in transactions.columns:
            transactions["day_of_week"] = pd.to_datetime(
                transactions["date"]
            ).dt.dayofweek
            transactions["day_of_month"] = pd.to_datetime(transactions["date"]).dt.day
            transactions["month"] = pd.to_datetime(transactions["date"]).dt.month

        # Make predictions
        predictions = self.model.predict(transactions)
        probabilities = self.model.predict_proba(transactions)

        # Get confidence scores (probability of predicted class)
        confidence = np.max(probabilities, axis=1)

        # Combine results
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            result = {
                "category": pred,
                "confidence": float(conf),
                "alternative_categories": [],
            }

            # Add alternative categories (top 3 excluding the prediction)
            if len(self.categories) > 1:
                # Get indices of predictions sorted by probability (descending)
                sorted_indices = np.argsort(probabilities[i])[::-1]

                # Add top 3 alternatives
                for idx in sorted_indices[1:4]:  # Skip the top one (prediction)
                    if idx < len(self.categories):  # Ensure index is valid
                        alt_category = self.categories[idx]
                        alt_confidence = float(probabilities[i][idx])

                        if alt_confidence > 0.05:  # Only include if confidence > 5%
                            result["alternative_categories"].append(
                                {"category": alt_category, "confidence": alt_confidence}
                            )

            results.append(result)

        return results

    def update_model(self, new_transactions_df):
        """
        Update the existing model with new transaction data

        Args:
            new_transactions_df: DataFrame with new transaction data

        Returns:
            self: Updated model with new training data incorporated
        """
        if self.model is None:
            # If no model exists, just train from scratch
            return self.fit(new_transactions_df)

        # Ensure the new data has the required columns
        if "category" not in new_transactions_df.columns:
            raise ValueError(
                "New data must include 'category' column for model updating"
            )

        # Extract features and targets
        X_new = new_transactions_df.drop("category", axis=1)
        y_new = new_transactions_df["category"]

        # Store the previous model info for reference
        previous_model_id = self.model_id
        previous_categories = self.categories.copy() if self.categories else []

        # Update categories list with any new categories
        new_categories = set(y_new.unique()) - set(previous_categories)
        if new_categories:
            self.categories = sorted(list(set(previous_categories) | new_categories))

        # Update the model (for scikit-learn pipeline, this means retraining)
        self.model.fit(X_new, y_new)

        # Calculate updated metrics
        X_test = X_new.sample(frac=0.2, random_state=42) if len(X_new) > 10 else X_new
        y_test = y_new.loc[X_test.index]
        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "samples_used": len(X_new),
            "new_categories_added": list(new_categories) if new_categories else [],
        }

        # Print performance metrics
        print("Updated model performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")

        # Save as new version with reference to previous version
        metadata = {
            "categories": self.categories,
            "previous_model_id": previous_model_id,
            "update_time": datetime.now().isoformat(),
            "training_samples": len(X_new),
        }

        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="transaction_categorizer_updated",
            model_type="categorization",
            features=self.feature_names,
            metrics=metrics,
            metadata=metadata,
        )

        return self
