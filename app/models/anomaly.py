import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.utils.model_registry import ModelRegistry
from app.utils.data_processing import extract_transaction_features


class AnomalyDetector:
    """
    Model for detecting unusual transactions and spending patterns
    """

    def __init__(self, registry=None):
        self.model = None
        self.registry = registry or ModelRegistry()
        self.model_id = None
        self.scaler = None
        self.feature_cols = None
        self.category_models = {}  # Separate model for each category
        self.global_model = None  # Model for all transactions

    def fit(self, transactions_df):
        """
        Train the anomaly detection model

        Args:
            transactions_df: DataFrame with transaction data

        Returns:
            self: The trained model
        """
        # Check required columns
        required_cols = ["amount", "date"]
        for col in required_cols:
            if col not in transactions_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Extract features
        df = extract_transaction_features(transactions_df)

        # Define features to use for anomaly detection
        if "category" in df.columns:
            self.feature_cols = [
                "amount_abs",
                "day_of_week",
                "day_of_month",
                "month",
                "is_weekend",
            ]
        else:
            self.feature_cols = [
                "amount_abs",
                "day_of_week",
                "day_of_month",
                "month",
                "is_weekend",
            ]

        # Add more features if available
        for col in ["is_expense", "is_income"]:
            if col in df.columns:
                self.feature_cols.append(col)

        # Global model (all transactions)
        X = df[self.feature_cols].copy()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train global model
        self.global_model = IsolationForest(
            contamination=0.05,  # Assume 5% of transactions are anomalies
            random_state=42,
        )
        self.global_model.fit(X_scaled)

        # Train category-specific models if category exists
        if "category" in df.columns:
            for category, group in df.groupby("category"):
                # Only create model if we have enough data
                if len(group) >= 10:
                    X_cat = group[self.feature_cols].copy()
                    X_cat_scaled = self.scaler.transform(X_cat)

                    # Train model with appropriate contamination
                    if len(X_cat) >= 50:
                        contamination = 0.05  # 5% for larger groups
                    else:
                        contamination = 0.1  # 10% for smaller groups

                    model = IsolationForest(
                        contamination=contamination, random_state=42
                    )
                    model.fit(X_cat_scaled)

                    self.category_models[category] = model

        # Create composite model
        self.model = {
            "global_model": self.global_model,
            "category_models": self.category_models,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }

        # Save model to registry
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="anomaly_detector",
            model_type="anomaly_detection",
            features=self.feature_cols,
            metrics=None,
            metadata={"has_category_models": len(self.category_models) > 0},
        )

        return self

    def detect_anomalies(self, transactions_df, threshold=-0.5):
        """
        Detect anomalies in transaction data

        Args:
            transactions_df: DataFrame with transaction data
            threshold: Anomaly score threshold (lower = more anomalous)

        Returns:
            DataFrame: Original data with anomaly scores and flags
        """
        if self.model is None:
            # Try to load the latest model
            self.model, model_info = self.registry.load_model(
                model_type="anomaly_detection", latest=True
            )
            self.global_model = self.model["global_model"]
            self.category_models = self.model["category_models"]
            self.scaler = self.model["scaler"]
            self.feature_cols = self.model["feature_cols"]
            self.model_id = model_info["id"]

        # Extract features
        df = extract_transaction_features(transactions_df).copy()

        # Ensure all required feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                # Try to derive the feature
                if col == "amount_abs":
                    df["amount_abs"] = df["amount"].abs()
                elif col == "is_expense":
                    df["is_expense"] = (df["amount"] < 0).astype(int)
                elif col == "is_income":
                    df["is_income"] = (df["amount"] > 0).astype(int)
                elif col == "is_weekend":
                    if "day_of_week" in df.columns:
                        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
                else:
                    # Use a default value
                    df[col] = 0

        # Scale features
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)

        # Get anomaly scores from global model
        df["anomaly_score"] = self.global_model.decision_function(X_scaled)
        df["is_anomaly"] = self.global_model.predict(X_scaled) == -1

        # Apply category-specific models if available
        if "category" in df.columns and self.category_models:
            # Create a copy of the global anomaly results
            df["global_anomaly_score"] = df["anomaly_score"]
            df["global_is_anomaly"] = df["is_anomaly"]

            # Process each category
            for category, model in self.category_models.items():
                mask = df["category"] == category
                if mask.sum() > 0:
                    X_cat = df.loc[mask, self.feature_cols]
                    X_cat_scaled = self.scaler.transform(X_cat)

                    # Get category-specific anomaly scores
                    cat_scores = model.decision_function(X_cat_scaled)
                    cat_predictions = model.predict(X_cat_scaled) == -1

                    # Update scores for this category
                    df.loc[mask, "anomaly_score"] = cat_scores
                    df.loc[mask, "is_anomaly"] = cat_predictions

        # Apply custom threshold
        if threshold != -0.5:  # Default threshold from IsolationForest
            df["is_anomaly"] = df["anomaly_score"] <= threshold

        # Calculate anomaly confidence
        df["anomaly_confidence"] = 1.0 - (
            df["anomaly_score"] - df["anomaly_score"].min()
        ) / (df["anomaly_score"].max() - df["anomaly_score"].min())

        # Format the confidence as a percentage between 50-100%
        # Even the least anomalous transactions have 50% confidence
        df["anomaly_confidence"] = 0.5 + df["anomaly_confidence"] * 0.5

        # Determine anomaly type
        df["anomaly_type"] = self._determine_anomaly_type(df)

        # Generate explanation for anomalies
        df["anomaly_explanation"] = self._generate_explanations(df)

        return df

    def _determine_anomaly_type(self, df):
        """
        Determine the type of anomaly for each transaction

        Args:
            df: DataFrame with anomaly scores

        Returns:
            Series: Anomaly types
        """
        types = pd.Series(["normal"] * len(df), index=df.index)

        # Only process actual anomalies
        anomalies = df[df["is_anomaly"]].copy()

        if len(anomalies) == 0:
            return types

        # Check for unusually large amounts
        amount_mean = df["amount_abs"].mean()
        amount_std = df["amount_abs"].std()
        threshold = amount_mean + 3 * amount_std

        large_amount_mask = anomalies["amount_abs"] > threshold
        types.loc[anomalies[large_amount_mask].index] = "large_amount"

        # Check for transactions on unusual days
        if "day_of_week" in df.columns:
            # Find common transaction days
            common_days = df.groupby("day_of_week").size() / len(df)
            rare_days = common_days[common_days < 0.05].index.tolist()

            unusual_day_mask = anomalies["day_of_week"].isin(rare_days)
            # Only mark as unusual day if not already marked as large amount
            unusual_day_mask = unusual_day_mask & ~anomalies.index.isin(
                types[types == "large_amount"].index
            )
            types.loc[anomalies[unusual_day_mask].index] = "unusual_day"

        # Check for category-based anomalies
        if "category" in df.columns:
            # Process each category
            for category, group in anomalies.groupby("category"):
                # Get normal transactions for this category
                normal_group = df[(df["category"] == category) & ~df["is_anomaly"]]

                if len(normal_group) > 0:
                    # Check for amount anomalies within category
                    cat_mean = normal_group["amount_abs"].mean()
                    cat_std = normal_group["amount_abs"].std()
                    cat_threshold = cat_mean + 2 * cat_std

                    cat_amount_mask = group["amount_abs"] > cat_threshold
                    # Only mark if not already categorized
                    unmarked = ~group.index.isin(types[types != "normal"].index)
                    types.loc[group[cat_amount_mask & unmarked].index] = (
                        "unusual_for_category"
                    )

        # Mark remaining anomalies as 'pattern_anomaly'
        remaining_mask = df["is_anomaly"] & types.isin(["normal"])
        types.loc[remaining_mask.index] = "pattern_anomaly"

        return types

    def _generate_explanations(self, df):
        """
        Generate text explanations for anomalies

        Args:
            df: DataFrame with anomaly information

        Returns:
            Series: Explanations for each transaction
        """
        explanations = pd.Series([""] * len(df), index=df.index)

        # Only generate explanations for anomalies
        anomalies = df[df["is_anomaly"]].copy()

        if len(anomalies) == 0:
            return explanations

        # Generate explanations based on anomaly type
        for idx, row in anomalies.iterrows():
            if row["anomaly_type"] == "large_amount":
                avg_amount = df[df["is_anomaly"] == False]["amount_abs"].mean()
                times_larger = row["amount_abs"] / avg_amount
                explanations.loc[idx] = (
                    f"Amount is {times_larger:.1f}x larger than your average transaction"
                )

            elif row["anomaly_type"] == "unusual_day":
                day_names = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                day_name = day_names[int(row["day_of_week"])]
                explanations.loc[idx] = (
                    f"Unusual transaction on {day_name}, which is not your typical spending day"
                )

            elif row["anomaly_type"] == "unusual_for_category":
                if "category" in row:
                    category_avg = df[
                        (df["category"] == row["category"])
                        & (df["is_anomaly"] == False)
                    ]["amount_abs"].mean()
                    times_larger = row["amount_abs"] / category_avg
                    explanations.loc[idx] = (
                        f"Amount is {times_larger:.1f}x larger than your typical {row['category']} transaction"
                    )
                else:
                    explanations.loc[idx] = "Unusual transaction pattern detected"

            elif row["anomaly_type"] == "pattern_anomaly":
                explanations.loc[idx] = (
                    "Unusual combination of transaction characteristics"
                )

        return explanations

    def update_model(self, new_transactions_df):
        """
        Update the anomaly detection model with new transaction data

        Args:
            new_transactions_df: DataFrame with new transaction data

        Returns:
            self: The updated model
        """
        if self.model is None:
            # If no model exists, just train from scratch
            return self.fit(new_transactions_df)

        # Store previous model ID and components
        previous_model_id = self.model_id
        previous_category_models = self.category_models.copy()

        # Extract features for the new data
        df = extract_transaction_features(new_transactions_df)

        # Update the global model
        X = df[self.feature_cols].copy()

        # Re-fit the scaler with combined data
        X_scaled = self.scaler.fit_transform(X)

        # Update global model
        contamination = 0.05  # Keep the same contamination rate
        self.global_model = IsolationForest(
            contamination=contamination, random_state=42
        )
        self.global_model.fit(X_scaled)

        # Update category-specific models if category exists
        if "category" in df.columns:
            for category, group in df.groupby("category"):
                # Only create/update model if we have enough data
                if len(group) >= 10:
                    X_cat = group[self.feature_cols].copy()
                    X_cat_scaled = self.scaler.transform(X_cat)

                    # Set appropriate contamination
                    if len(X_cat) >= 50:
                        contamination = 0.05  # 5% for larger groups
                    else:
                        contamination = 0.1  # 10% for smaller groups

                    # Create or update model
                    model = IsolationForest(
                        contamination=contamination, random_state=42
                    )
                    model.fit(X_cat_scaled)

                    self.category_models[category] = model

        # Update the composite model
        self.model = {
            "global_model": self.global_model,
            "category_models": self.category_models,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }

        # Save updated model
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="anomaly_detector_updated",
            model_type="anomaly_detection",
            features=self.feature_cols,
            metrics=None,
            metadata={
                "has_category_models": len(self.category_models) > 0,
                "previous_model_id": previous_model_id,
                "update_time": datetime.now().isoformat(),
                "new_categories": list(
                    set(self.category_models.keys())
                    - set(previous_category_models.keys())
                ),
            },
        )

        return self
