import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from models.budgeting import BudgetRecommender
from util.model_registry import ModelRegistry


class BudgetService:
    """
    Service for generating budget recommendations and insights
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.recommender = BudgetRecommender(registry=self.registry)

        # Try to load existing model
        try:
            self.recommender.model, model_info = self.registry.load_model(
                model_type="budgeting", latest=True
            )
            self.recommender.model_id = model_info["id"]
            self.recommender.category_clusters = model_info["metadata"][
                "category_clusters"
            ]
            self.recommender.income_based_allocation = model_info["metadata"][
                "income_based_allocation"
            ]
            print(f"Loaded budget recommendation model: {model_info['id']}")
        except Exception as e:
            print(f"No existing budget recommendation model found: {str(e)}")

    def generate_budget(self, user_id, transactions, monthly_income):
        """
        Generate budget recommendations based on transactions and income

        Args:
            user_id: User ID
            transactions: List of transaction dictionaries
            monthly_income: Monthly income amount

        Returns:
            dict: Budget recommendations and insights
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Ensure required columns exist
        required_cols = ["amount", "category"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # If recommender model doesn't exist, train a new one
        if self.recommender.model is None:
            try:
                self.recommender.fit(df, monthly_income=monthly_income)
            except Exception as e:
                print(f"Error training budget recommendation model: {str(e)}")
                # Fall back to rule-based recommendations
                return self._rule_based_budget(df, monthly_income)

        # Get the unique categories from transactions
        categories = df["category"].unique().tolist()

        # Generate budget recommendations
        try:
            recommendations = self.recommender.recommend_budget(
                monthly_income=monthly_income, categories=categories
            )

            # Group recommendations by type
            grouped_recommendations = self._group_recommendations_by_type(
                recommendations
            )

            # Calculate insights
            insights = self._calculate_budget_insights(
                df, recommendations, monthly_income
            )

            return {
                "monthly_income": monthly_income,
                "recommended_budget": {
                    "total": sum(recommendations.values()),
                    "by_category": recommendations,
                    "by_type": grouped_recommendations,
                },
                "insights": insights,
            }
        except Exception as e:
            print(f"Error generating budget recommendations: {str(e)}")
            # Fall back to rule-based recommendations
            return self._rule_based_budget(df, monthly_income)

    def _group_recommendations_by_type(self, recommendations):
        """Group budget recommendations by type (essentials, discretionary, savings)"""
        if (
            not hasattr(self.recommender, "category_clusters")
            or not self.recommender.category_clusters
        ):
            return {}

        grouped = {"essentials": {}, "discretionary": {}, "savings": {}}

        for category, amount in recommendations.items():
            budget_type = self.recommender.category_clusters.get(
                category,
                self.recommender._categorize_new_category(
                    category, self.recommender.category_mappings
                ),
            )
            grouped[budget_type][category] = amount

        # Add totals for each group
        result = {
            "essentials": {
                "categories": grouped["essentials"],
                "total": sum(grouped["essentials"].values()),
            },
            "discretionary": {
                "categories": grouped["discretionary"],
                "total": sum(grouped["discretionary"].values()),
            },
            "savings": {
                "categories": grouped["savings"],
                "total": sum(grouped["savings"].values()),
            },
        }

        return result

    def _calculate_budget_insights(
        self, transactions_df, recommendations, monthly_income
    ):
        """Calculate budget insights based on transactions and recommendations"""
        insights = {}

        # Calculate current spending by category
        df = transactions_df.copy()
        df["abs_amount"] = df["amount"].abs()

        # Filter to include only expenses (negative amounts)
        expenses_df = df[df["amount"] < 0].copy()

        # Check if date column exists
        has_date = "date" in expenses_df.columns

        if has_date:
            # Convert date to datetime
            expenses_df["date"] = pd.to_datetime(expenses_df["date"])

            # Calculate current month spending
            current_month = datetime.now().strftime("%Y-%m")
            current_month_df = expenses_df[
                expenses_df["date"].dt.strftime("%Y-%m") == current_month
            ]

            if len(current_month_df) > 0:
                current_spending = (
                    current_month_df.groupby("category")["abs_amount"].sum().to_dict()
                )
            else:
                # If no current month data, use all data
                current_spending = (
                    expenses_df.groupby("category")["abs_amount"].sum().to_dict()
                )
        else:
            # If no date column, use all data
            current_spending = (
                expenses_df.groupby("category")["abs_amount"].sum().to_dict()
            )

        # Calculate over/under budget for each category
        budget_status = {}
        for category, recommended in recommendations.items():
            current = current_spending.get(category, 0)
            difference = recommended - current
            status = "under_budget" if difference >= 0 else "over_budget"

            budget_status[category] = {
                "recommended": recommended,
                "current": current,
                "difference": difference,
                "status": status,
                "percent_used": min(
                    round((current / recommended * 100) if recommended > 0 else 0, 1),
                    100,
                ),
            }

        insights["category_status"] = budget_status

        # Calculate allocation percentages
        total_budget = sum(recommendations.values())
        allocation_percentages = {
            category: round(
                (amount / monthly_income * 100) if monthly_income > 0 else 0, 1
            )
            for category, amount in recommendations.items()
        }

        insights["allocation_percentages"] = allocation_percentages

        # Compare to 50/30/20 rule
        if (
            hasattr(self.recommender, "category_clusters")
            and self.recommender.category_clusters
        ):
            type_totals = {"essentials": 0, "discretionary": 0, "savings": 0}

            for category, amount in recommendations.items():
                budget_type = self.recommender.category_clusters.get(
                    category,
                    self.recommender._categorize_new_category(
                        category, self.recommender.category_mappings
                    ),
                )
                type_totals[budget_type] += amount

            # Calculate percentages
            type_percentages = {
                budget_type: round(
                    (amount / monthly_income * 100) if monthly_income > 0 else 0, 1
                )
                for budget_type, amount in type_totals.items()
            }

            # Compare to ideal allocations
            rule_comparison = {
                "current_allocation": type_percentages,
                "ideal_allocation": {
                    "essentials": self.recommender.income_based_allocation.get(
                        "essentials", 0.5
                    )
                    * 100,
                    "discretionary": self.recommender.income_based_allocation.get(
                        "discretionary", 0.3
                    )
                    * 100,
                    "savings": self.recommender.income_based_allocation.get(
                        "savings", 0.2
                    )
                    * 100,
                },
            }

            insights["rule_comparison"] = rule_comparison

        # Add top spending categories
        top_categories = sorted(
            [(category, amount) for category, amount in current_spending.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        insights["top_spending_categories"] = [
            {"category": category, "amount": amount}
            for category, amount in top_categories
        ]

        return insights

    def _rule_based_budget(self, transactions_df, monthly_income):
        """
        Generate rule-based budget recommendations when ML model is unavailable

        Args:
            transactions_df: DataFrame with transaction data
            monthly_income: Monthly income amount

        Returns:
            dict: Budget recommendations
        """
        # Apply the 50/30/20 rule
        essentials_budget = monthly_income * 0.5
        discretionary_budget = monthly_income * 0.3
        savings_budget = monthly_income * 0.2

        # Default category mappings
        category_types = {
            # Essentials
            "housing": "essentials",
            "rent": "essentials",
            "mortgage": "essentials",
            "utilities": "essentials",
            "groceries": "essentials",
            "healthcare": "essentials",
            "insurance": "essentials",
            "transportation": "essentials",
            "education": "essentials",
            # Discretionary
            "dining": "discretionary",
            "restaurant": "discretionary",
            "entertainment": "discretionary",
            "shopping": "discretionary",
            "travel": "discretionary",
            "subscription": "discretionary",
            "personal": "discretionary",
            "gifts": "discretionary",
            # Savings
            "savings": "savings",
            "investment": "savings",
            "retirement": "savings",
            "debt_payment": "savings",
        }

        # Calculate current spending by category (if available)
        current_spending = {}

        if len(transactions_df) > 0:
            # Filter expenses
            expenses_df = transactions_df[transactions_df["amount"] < 0].copy()
            expenses_df["abs_amount"] = expenses_df["amount"].abs()

            # Group by category
            current_spending = (
                expenses_df.groupby("category")["abs_amount"].sum().to_dict()
            )

        # Get unique categories from transactions
        categories = transactions_df["category"].unique().tolist()

        # Group categories by type
        categories_by_type = {"essentials": [], "discretionary": [], "savings": []}

        for category in categories:
            category_lower = category.lower()

            # Try to match to a known category type
            found_match = False
            for known_category, budget_type in category_types.items():
                if known_category in category_lower or category_lower in known_category:
                    categories_by_type[budget_type].append(category)
                    found_match = True
                    break

            # If no match found, make an educated guess based on keywords
            if not found_match:
                if any(
                    keyword in category_lower
                    for keyword in ["bill", "essential", "necessity", "base"]
                ):
                    categories_by_type["essentials"].append(category)
                elif any(
                    keyword in category_lower
                    for keyword in ["save", "invest", "debt", "loan"]
                ):
                    categories_by_type["savings"].append(category)
                else:
                    # Default to discretionary
                    categories_by_type["discretionary"].append(category)

        # Allocate budget to each category
        recommendations = {}

        def allocate_budget(budget_amount, categories, spending_data):
            if not categories:
                return {}

            result = {}
            total_spending = sum(
                spending_data.get(category, 0) for category in categories
            )

            if total_spending > 0:
                # Allocate based on historical spending ratios
                for category in categories:
                    category_spending = spending_data.get(category, 0)
                    ratio = category_spending / total_spending
                    result[category] = round(budget_amount * ratio, 2)
            else:
                # Equal allocation if no historical data
                per_category = budget_amount / len(categories)
                for category in categories:
                    result[category] = round(per_category, 2)

            return result

        # Allocate budget for each type
        essentials_allocation = allocate_budget(
            essentials_budget, categories_by_type["essentials"], current_spending
        )

        discretionary_allocation = allocate_budget(
            discretionary_budget, categories_by_type["discretionary"], current_spending
        )

        savings_allocation = allocate_budget(
            savings_budget, categories_by_type["savings"], current_spending
        )

        # Combine allocations
        recommendations = {
            **essentials_allocation,
            **discretionary_allocation,
            **savings_allocation,
        }

        # Prepare the response
        grouped_recommendations = {
            "essentials": {
                "categories": essentials_allocation,
                "total": sum(essentials_allocation.values()),
            },
            "discretionary": {
                "categories": discretionary_allocation,
                "total": sum(discretionary_allocation.values()),
            },
            "savings": {
                "categories": savings_allocation,
                "total": sum(savings_allocation.values()),
            },
        }

        # Basic insights
        insights = {
            "rule_comparison": {
                "current_allocation": {
                    "essentials": round(
                        (
                            sum(essentials_allocation.values()) / monthly_income * 100
                            if monthly_income > 0
                            else 0
                        ),
                        1,
                    ),
                    "discretionary": round(
                        (
                            sum(discretionary_allocation.values())
                            / monthly_income
                            * 100
                            if monthly_income > 0
                            else 0
                        ),
                        1,
                    ),
                    "savings": round(
                        (
                            sum(savings_allocation.values()) / monthly_income * 100
                            if monthly_income > 0
                            else 0
                        ),
                        1,
                    ),
                },
                "ideal_allocation": {
                    "essentials": 50.0,
                    "discretionary": 30.0,
                    "savings": 20.0,
                },
            },
            "top_spending_categories": [
                {"category": category, "amount": amount}
                for category, amount in sorted(
                    current_spending.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ],
        }

        return {
            "monthly_income": monthly_income,
            "recommended_budget": {
                "total": sum(recommendations.values()),
                "by_category": recommendations,
                "by_type": grouped_recommendations,
            },
            "insights": insights,
        }

    def train_budget_model(
        self, transactions_data, monthly_income, custom_allocations=None
    ):
        """
        Train a new budget recommendation model

        Args:
            transactions_data: List of transaction dictionaries
            monthly_income: Monthly income amount
            custom_allocations: Custom allocation percentages (optional)

        Returns:
            dict: Training results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["amount", "category"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Create a new recommender
        self.recommender = BudgetRecommender(registry=self.registry)

        # Train the model
        try:
            self.recommender.fit(
                df, monthly_income=monthly_income, custom_allocations=custom_allocations
            )

            return {
                "success": True,
                "model_id": self.recommender.model_id,
                "category_clusters": self.recommender.category_clusters,
                "message": "Budget recommendation model trained successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_budget_model(
        self, transactions_data, monthly_income=None, custom_allocations=None
    ):
        """
        Update the budget recommendation model with new data

        Args:
            transactions_data: List of transaction dictionaries
            monthly_income: Monthly income amount (optional)
            custom_allocations: Custom allocation percentages (optional)

        Returns:
            dict: Update results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["amount", "category"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # If recommender model doesn't exist, train a new one
        if self.recommender.model is None:
            try:
                self.recommender.fit(
                    df,
                    monthly_income=monthly_income,
                    custom_allocations=custom_allocations,
                )
                return {
                    "success": True,
                    "model_id": self.recommender.model_id,
                    "message": "New budget recommendation model trained successfully",
                }
            except Exception as e:
                print(f"Error training budget recommendation model: {str(e)}")
                return {"success": False, "error": str(e)}

        # Update existing model
        try:
            # Update the model
            self.recommender.update_model(
                df, monthly_income=monthly_income, custom_allocations=custom_allocations
            )

            # Get updated category mappings
            updated_clusters = self.recommender.category_clusters

            return {
                "success": True,
                "model_id": self.recommender.model_id,
                "category_clusters": updated_clusters,
                "message": "Budget recommendation model updated successfully",
                "updated_categories": list(
                    set(updated_clusters.keys())
                    - set(self.recommender.model.get("category_clusters", {}).keys())
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
