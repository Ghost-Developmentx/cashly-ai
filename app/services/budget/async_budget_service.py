"""
Async budget generation and management service.
"""

import logging
from typing import Dict, List, Any, Optional

from .budget_calculator import BudgetCalculator
from .budget_analyzer import BudgetAnalyzer
from .budget_recommender import BudgetRecommender

logger = logging.getLogger(__name__)


class AsyncBudgetService:
    """
    Asynchronous service for budget management and recommendation.

    This class is designed to assist users in generating personalized budget
    recommendations, analyzing spending patterns, and identifying savings
    opportunities. By leveraging historical transaction data and optional
    income information, the service performs complex calculations, including
    income estimation and budget allocation. Additionally, it offers tailored
    recommendations to guide users toward optimal financial management.

    Attributes
    ----------
    calculator : BudgetCalculator
        The component responsible for calculating income and budget allocations.
    analyzer : BudgetAnalyzer
        The component used for analyzing spending patterns from historical transactions.
    recommender : BudgetRecommender
        The component used for generating personalized recommendations.

    """

    def __init__(self):
        self.calculator = BudgetCalculator()
        self.analyzer = BudgetAnalyzer()
        self.recommender = BudgetRecommender()

    async def generate_budget(
        self,
        user_id: str,
        transactions: List[Dict[str, Any]],
        monthly_income: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates a detailed budget for a user based on transaction history, optional
        monthly income, and spending analysis. The process includes analyzing spending
        patterns, calculating income if not provided, generating budget allocations,
        and providing recommendations for savings and spending.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user for whom the budget is being generated.
        transactions : List[Dict[str, Any]]
            A list of transactions where each transaction is represented as a dictionary
            containing details like category, amount, and date, which are used for
            spending analysis and budget generation.
        monthly_income : Optional[float], default=None
            The user's monthly income. If not provided, it will be calculated based
            on the transaction data.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the generated budget details, including monthly
            income, budget allocations for various spending categories, analyzed
            current spending per category, actionable recommendations, savings
            potential, and the analysis period. If there's an error during processing,
            an error message is returned instead.
        """
        try:
            if not transactions:
                return self._empty_budget_response()

            # Analyze spending patterns
            spending_analysis = await self.analyzer.analyze_spending(transactions)

            # Calculate income if not provided
            if monthly_income is None:
                monthly_income = await self.calculator.calculate_monthly_income(
                    transactions
                )

            # Generate budget allocations
            budget_allocations = await self.calculator.calculate_budget(
                spending_analysis, monthly_income
            )

            # Generate recommendations
            recommendations = await self.recommender.generate_recommendations(
                spending_analysis, budget_allocations
            )

            return {
                "monthly_income": monthly_income,
                "budget_allocations": budget_allocations,
                "current_spending": spending_analysis["category_averages"],
                "recommendations": recommendations,
                "savings_potential": self._calculate_savings_potential(
                    spending_analysis, budget_allocations
                ),
                "analysis_period": spending_analysis["period"],
            }

        except Exception as e:
            logger.error(f"Budget generation failed: {e}")
            return {"error": f"Failed to generate budget: {str(e)}"}

    @staticmethod
    def _calculate_savings_potential(
        spending_analysis: Dict[str, Any], budget_allocations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate the potential savings based on spending analysis and budget allocations.

        This method computes the monthly and annual savings a user could achieve when
        comparing their actual spending habits to the budget they plan to adhere to. It
        also calculates the percentage of savings relative to the current total spending.

        Parameters
        ----------
        spending_analysis : Dict[str, Any]
            A dictionary containing analysis of spending data, typically including
            averages for each spending category.

        budget_allocations : Dict[str, float]
            A dictionary containing budget allocations with category names as keys
            and allocated amounts as values.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the following calculated savings data:
                - "monthly": Monthly savings based on current spending and budget.
                - "annual": Annual savings projected from the monthly savings.
                - "percentage": Percentage of savings compared to current spending.
        """
        current_total = sum(spending_analysis["category_averages"].values())
        budget_total = sum(budget_allocations.values())

        monthly_savings = max(0, current_total - budget_total)
        annual_savings = monthly_savings * 12

        return {
            "monthly": round(monthly_savings, 2),
            "annual": round(annual_savings, 2),
            "percentage": round(
                (monthly_savings / current_total * 100) if current_total > 0 else 0, 1
            ),
        }

    @staticmethod
    def _empty_budget_response() -> Dict[str, Any]:
        """Return empty budget response."""
        return {
            "message": "No transaction data available for budget generation",
            "monthly_income": 0,
            "budget_allocations": {},
            "recommendations": [],
        }
