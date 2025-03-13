import pandas as pd
from datetime import datetime
import json
from models.categorization import TransactionCategorizer
from util.data_processing import extract_transaction_features
from util.model_registry import ModelRegistry


class CategorizationService:
    """
    Service for categorizing transactions
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.categorizer = TransactionCategorizer(registry=self.registry)

        # Try to load existing model
        try:
            self.categorizer.model, model_info = self.registry.load_model(
                model_type="categorization", latest=True
            )
            self.categorizer.categories = model_info["metadata"]["categories"]
            self.categorizer.feature_names = model_info["features"]
            self.categorizer.model_id = model_info["id"]
            print(f"Loaded categorization model: {model_info['id']}")
        except Exception as e:
            print(f"No existing categorization model found: {str(e)}")

    def categorize_transaction(self, description, amount, date=None):
        """
        Categorize a single transaction

        Args:
            description: Transaction description
            amount: Transaction amount
            date: Transaction date (optional)

        Returns:
            dict: Categorization result
        """
        # Create transaction object
        transaction = {
            "description": description,
            "amount": float(amount),
        }

        # Add date-related features if date is provided
        if date:
            transaction["date"] = date
            if isinstance(date, str):
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            else:
                date_obj = date

            transaction["day_of_week"] = date_obj.weekday()
            transaction["day_of_month"] = date_obj.day
            transaction["month"] = date_obj.month
            transaction["is_weekend"] = 1 if date_obj.weekday() >= 5 else 0

        # Add derived features
        transaction["amount_abs"] = abs(float(amount))
        transaction["is_expense"] = 1 if float(amount) < 0 else 0
        transaction["is_income"] = 1 if float(amount) > 0 else 0

        # If no model is loaded yet, use rule-based fallback
        if self.categorizer.model is None:
            return self._rule_based_categorization(transaction)

        # Use ML model for categorization
        try:
            results = self.categorizer.predict(pd.DataFrame([transaction]))

            if results and len(results) > 0:
                return results[0]
            else:
                return self._rule_based_categorization(transaction)

        except Exception as e:
            print(f"Error in ML categorization: {str(e)}")
            # Fallback to rule-based approach
            return self._rule_based_categorization(transaction)

    def _rule_based_categorization(self, transaction):
        """
        Fallback rule-based categorization when ML model is unavailable

        Args:
            transaction: Transaction dict

        Returns:
            dict: Categorization result
        """
        description = transaction["description"].lower()
        amount = transaction["amount"]

        # Simple rule-based categorization
        categories = {
            "grocery": ["grocery", "supermarket", "food", "walmart", "target"],
            "utilities": ["electric", "water", "gas", "utility", "utilities"],
            "subscription": ["netflix", "spotify", "subscription", "monthly"],
            "shopping": ["amazon", "ebay", "purchase", "buy"],
            "dining": ["restaurant", "cafe", "coffee", "dining"],
            "income": ["salary", "deposit", "income", "payment received"],
            "transportation": ["uber", "lyft", "taxi", "train", "transit"],
            "housing": ["rent", "mortgage", "housing"],
            "healthcare": ["doctor", "pharmacy", "medical", "health"],
            "entertainment": ["movie", "ticket", "entertainment"],
        }

        # Simple amount-based categorization
        if amount > 0:
            return {
                "category": "income",
                "confidence": 0.9,
                "alternative_categories": [
                    {"category": "transfer", "confidence": 0.05},
                    {"category": "refund", "confidence": 0.03},
                ],
            }

        # Text-based categorization
        best_category = "uncategorized"
        best_score = 0

        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > best_score:
                best_score = score
                best_category = category

        confidence = min(0.5 + (best_score * 0.1), 0.95) if best_score > 0 else 0.3

        # Generate alternative categories
        alternatives = []
        for category, keywords in categories.items():
            if category != best_category:
                score = sum(1 for keyword in keywords if keyword in description)
                if score > 0:
                    alt_confidence = min(0.3 + (score * 0.1), 0.6)
                    alternatives.append(
                        {"category": category, "confidence": alt_confidence}
                    )

        # Sort alternatives by confidence and take top 3
        alternatives = sorted(
            alternatives, key=lambda x: x["confidence"], reverse=True
        )[:3]

        return {
            "category": best_category,
            "confidence": confidence,
            "alternative_categories": alternatives,
        }

    def train_model(self, transactions_data):
        """
        Train a new categorization model with provided transaction data

        Args:
            transactions_data: List of transaction dictionaries with 'category' labels

        Returns:
            dict: Training results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["description", "amount", "category"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Extract features
        df = extract_transaction_features(df)

        # Train the model
        try:
            self.categorizer.fit(df)

            return {
                "success": True,
                "model_id": self.categorizer.model_id,
                "categories": self.categorizer.categories,
                "message": "Model trained successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_info(self):
        """
        Get information about the current categorization model

        Returns:
            dict: Model information
        """
        if self.categorizer.model is None:
            return {"error": "No model loaded"}

        # Get model details from registry
        models = self.registry.list_models(model_type="categorization")

        if not models:
            return {"error": "No models found in registry"}

        # Find the current model
        current_model = None
        for model in models:
            if model["id"] == self.categorizer.model_id:
                current_model = model
                break

        if current_model:
            return {
                "model_id": current_model["id"],
                "created_at": current_model["created_at"],
                "categories": current_model["metadata"]["categories"],
                "metrics": current_model["metrics"],
            }
        else:
            return {"error": "Current model not found in registry"}

    def update_model(self, transactions_data):
        """
        Update the categorization model with new data

        Args:
            transactions_data: List of transaction dictionaries with 'category' labels

        Returns:
            dict: Update results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["description", "amount", "category"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Extract features
        df = extract_transaction_features(df)

        # Update the model
        try:
            self.categorizer.update_model(df)

            return {
                "success": True,
                "model_id": self.categorizer.model_id,
                "categories": self.categorizer.categories,
                "message": "Model updated successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
