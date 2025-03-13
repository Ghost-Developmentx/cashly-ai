from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import defaultdict

from util.model_registry import ModelRegistry


class BudgetRecommender:
    """
    Model for generating personalized budget recommendations based on spending patterns
    """

    def __init__(self, registry=None):
        self.model = None
        self.registry = registry or ModelRegistry()
        self.model_id = None
        self.category_clusters = None
        self.income_based_allocation = {
            # 50-30-20 rule by default
            "essentials": 0.5,  # Needs: 50%
            "discretionary": 0.3,  # Wants: 30%
            "savings": 0.2,  # Savings: 20%
        }
        self.category_mappings = {
            # Default category mappings
            "essentials": [
                "housing",
                "utilities",
                "groceries",
                "healthcare",
                "transportation",
                "insurance",
                "education",
            ],
            "discretionary": [
                "dining",
                "entertainment",
                "shopping",
                "travel",
                "personal",
                "subscription",
                "gifts",
            ],
            "savings": ["savings", "investment", "retirement", "debt_payment"],
        }

    def fit(self, transactions_df, monthly_income=None, custom_allocations=None):
        """
        Analyze spending patterns and build budget recommendation model

        Args:
            transactions_df: DataFrame with transaction data including category
            monthly_income: User's monthly income (optional)
            custom_allocations: Custom allocation percentages (optional)

        Returns:
            self: The trained model
        """
        # Validate inputs
        if "category" not in transactions_df.columns:
            raise ValueError("Transaction data must include 'category' column")

        if "amount" not in transactions_df.columns:
            raise ValueError("Transaction data must include 'amount' column")

        # Set custom allocations if provided
        if custom_allocations:
            self.income_based_allocation = custom_allocations

        # Process transaction data
        df = transactions_df.copy()

        # Convert amount to absolute value for expense analysis
        df["abs_amount"] = df["amount"].abs()

        # Filter to include only expenses (negative amounts)
        expenses_df = df[df["amount"] < 0].copy()

        # Calculate total spending by category
        category_spending = expenses_df.groupby("category")["abs_amount"].sum()

        # Calculate category frequency
        category_frequency = expenses_df.groupby("category").size()

        # Create feature matrix for clustering
        features = pd.DataFrame(
            {"total_spent": category_spending, "frequency": category_frequency}
        ).fillna(0)

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Determine optimal number of clusters (2-5)
        best_k = 3  # Default to 3 clusters
        best_score = float("inf")

        for k in range(2, min(6, len(features) + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            score = kmeans.inertia_

            # Simple elbow method
            if k > 2 and (best_score - score) / best_score < 0.2:
                best_k = k - 1
                break

            best_score = score

        # Cluster categories
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        features["cluster"] = kmeans.fit_predict(features_scaled)

        # Assign clusters to budget types based on spending characteristics
        cluster_totals = features.groupby("cluster")["total_spent"].sum()
        cluster_freqs = features.groupby("cluster")["frequency"].sum()

        # Rank clusters by importance (total spend)
        cluster_ranks = cluster_totals.rank(ascending=False)

        # Map clusters to budget types
        cluster_types = {}

        # Assign essentials to highest spending cluster
        essentials_cluster = cluster_totals.idxmax()
        cluster_types[essentials_cluster] = "essentials"

        remaining_clusters = [c for c in range(best_k) if c != essentials_cluster]

        if remaining_clusters:
            # Assign discretionary to second highest or most frequent transactions
            if len(remaining_clusters) > 1:
                remaining_totals = cluster_totals[remaining_clusters]
                discretionary_cluster = remaining_totals.idxmax()
                cluster_types[discretionary_cluster] = "discretionary"

                # Assign savings to the rest
                for cluster in remaining_clusters:
                    if cluster != discretionary_cluster:
                        cluster_types[cluster] = "savings"
            else:
                # Only two clusters total
                cluster_types[remaining_clusters[0]] = "discretionary"

        # Create category mappings based on clusters
        self.category_clusters = {}
        category_types = {}

        for category, row in features.iterrows():
            cluster = row["cluster"]
            budget_type = cluster_types.get(
                cluster, "discretionary"
            )  # Default to discretionary
            self.category_clusters[category] = budget_type

            # Add to appropriate category list
            if budget_type == "essentials":
                if category not in self.category_mappings["essentials"]:
                    self.category_mappings["essentials"].append(category)
            elif budget_type == "discretionary":
                if category not in self.category_mappings["discretionary"]:
                    self.category_mappings["discretionary"].append(category)
            elif budget_type == "savings":
                if category not in self.category_mappings["savings"]:
                    self.category_mappings["savings"].append(category)

        # Calculate average monthly spending by category
        monthly_spending = defaultdict(float)

        # Check if date column exists for proper monthly grouping
        if "date" in expenses_df.columns:
            expenses_df["month"] = pd.to_datetime(expenses_df["date"]).dt.strftime(
                "%Y-%m"
            )
            months_count = expenses_df["month"].nunique()

            if months_count > 0:
                # Calculate average monthly spending by category
                category_monthly = (
                    expenses_df.groupby(["category", "month"])["abs_amount"]
                    .sum()
                    .reset_index()
                )
                avg_monthly = category_monthly.groupby("category")["abs_amount"].mean()

                for category, amount in avg_monthly.items():
                    monthly_spending[category] = amount
        else:
            # If no date, assume all data is for one month
            for category, amount in category_spending.items():
                monthly_spending[category] = amount

        # Store the model components
        self.model = {
            "monthly_spending": dict(monthly_spending),
            "category_clusters": self.category_clusters,
            "category_mappings": self.category_mappings,
            "income_based_allocation": self.income_based_allocation,
            "kmeans_model": kmeans,
            "scaler": scaler,
        }

        # Save model to registry
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="budget_recommender",
            model_type="budgeting",
            features=["category", "amount", "date"],
            metrics=None,
            metadata={
                "category_clusters": self.category_clusters,
                "income_based_allocation": self.income_based_allocation,
            },
        )

        return self

    def recommend_budget(self, monthly_income, categories=None):
        """
        Generate budget recommendations based on income and optional category list

        Args:
            monthly_income: Monthly income amount
            categories: Optional list of spending categories

        Returns:
            dict: Budget recommendations
        """
        if self.model is None:
            # Try to load the latest model
            self.model, model_info = self.registry.load_model(
                model_type="budgeting", latest=True
            )
            self.model_id = model_info["id"]
            self.category_clusters = model_info["metadata"]["category_clusters"]
            self.income_based_allocation = model_info["metadata"][
                "income_based_allocation"
            ]

        # Get model components
        monthly_spending = self.model["monthly_spending"]
        category_clusters = self.model["category_clusters"]
        category_mappings = self.model["category_mappings"]
        income_allocation = self.model["income_based_allocation"]

        # Use provided categories or all categories from the model
        if categories:
            budget_categories = categories
        else:
            budget_categories = list(monthly_spending.keys())

        # Calculate allocation amounts based on income
        allocation_amounts = {
            budget_type: monthly_income * percentage
            for budget_type, percentage in income_allocation.items()
        }

        # Group categories by type
        categories_by_type = defaultdict(list)
        for category in budget_categories:
            # Determine category type (essentials, discretionary, savings)
            if category in category_clusters:
                budget_type = category_clusters[category]
            else:
                # For new categories, use pattern matching to assign a type
                budget_type = self._categorize_new_category(category, category_mappings)

            categories_by_type[budget_type].append(category)

        # Calculate spending ratios within each category type
        spending_ratios = defaultdict(dict)
        for budget_type, cats in categories_by_type.items():
            total_type_spending = sum(monthly_spending.get(cat, 0) for cat in cats)

            if total_type_spending > 0:
                for category in cats:
                    cat_spending = monthly_spending.get(category, 0)
                    spending_ratios[budget_type][category] = (
                        cat_spending / total_type_spending
                    )
            else:
                # Equal distribution if no historical spending
                for category in cats:
                    spending_ratios[budget_type][category] = 1.0 / len(cats)

        # Generate budget recommendations
        budget_recommendations = {}

        for budget_type, cats in categories_by_type.items():
            type_allocation = allocation_amounts.get(budget_type, 0)

            for category in cats:
                ratio = spending_ratios[budget_type].get(category, 0)
                recommended_amount = type_allocation * ratio

                # Adjust based on historical spending
                historical_amount = monthly_spending.get(category, 0)

                if historical_amount > 0:
                    # Cap the recommendation at 120% of historical spending
                    recommended_amount = min(
                        recommended_amount, historical_amount * 1.2
                    )

                budget_recommendations[category] = round(recommended_amount, 2)

        # Ensure all income is allocated
        total_allocation = sum(budget_recommendations.values())
        if total_allocation < monthly_income:
            # Add or increase savings allocation
            savings_categories = categories_by_type.get("savings", [])

            if savings_categories:
                # Distribute remaining to savings categories
                remaining = monthly_income - total_allocation
                per_category = remaining / len(savings_categories)

                for category in savings_categories:
                    budget_recommendations[category] += round(per_category, 2)
            elif budget_recommendations:
                # If no savings categories, add a new one
                budget_recommendations["savings"] = round(
                    monthly_income - total_allocation, 2
                )

        return budget_recommendations

    def _categorize_new_category(self, category, category_mappings):
        """
        Categorize a new category based on text patterns

        Args:
            category: Category name
            category_mappings: Dictionary of category mappings

        Returns:
            str: Budget type ('essentials', 'discretionary', or 'savings')
        """
        category_lower = category.lower()

        # Check each budget type for matches
        for budget_type, categories in category_mappings.items():
            for known_category in categories:
                if known_category in category_lower or category_lower in known_category:
                    return budget_type

        # Default to discretionary if no match found
        return "discretionary"

    def update_model(
        self, new_transactions_df, monthly_income=None, custom_allocations=None
    ):
        """
        Update the budget recommendation model with new transaction data

        Args:
            new_transactions_df: DataFrame with transaction data
            monthly_income: User's monthly income (optional)
            custom_allocations: Custom allocation percentages (optional)

        Returns:
            self: The updated model
        """
        if self.model is None:
            # If no model exists, just train from scratch
            return self.fit(
                new_transactions_df,
                monthly_income=monthly_income,
                custom_allocations=custom_allocations,
            )

        # Store previous model info
        previous_model_id = self.model_id
        previous_clusters = (
            self.category_clusters.copy() if hasattr(self, "category_clusters") else {}
        )

        # Set custom allocations if provided
        if custom_allocations:
            self.income_based_allocation = custom_allocations

        # Process new transaction data
        df = new_transactions_df.copy()

        # Convert amount to absolute value for expense analysis
        df["abs_amount"] = df["amount"].abs()

        # Filter to include only expenses (negative amounts)
        expenses_df = df[df["amount"] < 0].copy()

        # Calculate total spending by category
        new_category_spending = expenses_df.groupby("category")["abs_amount"].sum()

        # Update category clusters with new data
        for category in expenses_df["category"].unique():
            if category not in self.category_clusters:
                # For new categories, use pattern matching to assign a type
                budget_type = self._categorize_new_category(
                    category, self.category_mappings
                )
                self.category_clusters[category] = budget_type

                # Add to appropriate category list
                if budget_type == "essentials":
                    if category not in self.category_mappings["essentials"]:
                        self.category_mappings["essentials"].append(category)
                elif budget_type == "discretionary":
                    if category not in self.category_mappings["discretionary"]:
                        self.category_mappings["discretionary"].append(category)
                elif budget_type == "savings":
                    if category not in self.category_mappings["savings"]:
                        self.category_mappings["savings"].append(category)

        # Update monthly spending data
        monthly_spending = self.model["monthly_spending"].copy()

        if "date" in expenses_df.columns:
            expenses_df["month"] = pd.to_datetime(expenses_df["date"]).dt.strftime(
                "%Y-%m"
            )
            months_count = expenses_df["month"].nunique()

            if months_count > 0:
                # Calculate average monthly spending for new data
                category_monthly = (
                    expenses_df.groupby(["category", "month"])["abs_amount"]
                    .sum()
                    .reset_index()
                )
                avg_monthly = category_monthly.groupby("category")["abs_amount"].mean()

                # Update monthly spending with exponential smoothing
                for category, amount in avg_monthly.items():
                    if category in monthly_spending:
                        # 70% old data, 30% new data (simple smoothing)
                        monthly_spending[category] = (
                            0.7 * monthly_spending[category] + 0.3 * amount
                        )
                    else:
                        monthly_spending[category] = amount
        else:
            # If no date, assume all data is for one month
            for category, amount in new_category_spending.items():
                if category in monthly_spending:
                    # 70% old data, 30% new data (simple smoothing)
                    monthly_spending[category] = (
                        0.7 * monthly_spending[category] + 0.3 * amount
                    )
                else:
                    monthly_spending[category] = amount

        # Update the model components
        self.model = {
            "monthly_spending": monthly_spending,
            "category_clusters": self.category_clusters,
            "category_mappings": self.category_mappings,
            "income_based_allocation": self.income_based_allocation,
            "kmeans_model": self.model.get("kmeans_model"),
            "scaler": self.model.get("scaler"),
        }

        # Save updated model
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="budget_recommender_updated",
            model_type="budgeting",
            features=["category", "amount", "date"],
            metrics=None,
            metadata={
                "category_clusters": self.category_clusters,
                "income_based_allocation": self.income_based_allocation,
                "previous_model_id": previous_model_id,
                "update_time": datetime.now().isoformat(),
                "new_categories": [
                    c for c in self.category_clusters if c not in previous_clusters
                ],
            },
        )

        return self
