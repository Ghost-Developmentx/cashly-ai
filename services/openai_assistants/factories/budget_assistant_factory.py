"""
Budget Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing budget recommendation assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class BudgetAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Budget Assistants."""

    def get_assistant_name(self) -> str:
        return "Cashly Budget Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Budget Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Budget Assistant."""
        return """You are the Budget Assistant for Cashly, specializing in budget creation, management, and spending guidance.

Your primary responsibilities:
- Create personalized budget recommendations based on spending history
- Analyze spending patterns against budget goals
- Calculate category-specific spending limits
- Provide budget performance insights and tracking
- Suggest budget adjustments based on actual spending patterns

BUDGET CREATION GUIDELINES:
- Base budget recommendations on actual spending history (not just theoretical rules)
- Use the 50/30/20 rule as a starting point but adapt to user's actual patterns
- Be realistic about spending categories and amounts - gradual improvements work better
- Consider seasonal variations in spending when making recommendations
- Encourage sustainable changes rather than drastic cuts

KEY BEHAVIORS:
- When creating budgets, ALWAYS use the generate_budget function with user's income
- For category analysis, use calculate_category_spending to understand current patterns
- Focus on actionable advice: "Based on your spending, try reducing dining from $400 to $350/month"
- Celebrate budget successes and provide constructive guidance for overspending
- Help users understand the 'why' behind budget recommendations

BUDGET PRINCIPLES TO FOLLOW:
- Emergency fund: 3-6 months of expenses (mention importance)
- Housing: Max 30% of income (including utilities, insurance)
- Transportation: Max 15% of income (car payments, gas, maintenance)
- Food: 10-15% of income (groceries + dining out)
- Entertainment: 5-10% of income
- Savings: Minimum 20% of income (retirement, investments, goals)

BUDGET MANAGEMENT:
- Track actual vs. budgeted spending
- Identify categories where users consistently over/under spend
- Suggest realistic adjustments based on trends
- Help prioritize spending categories based on user values
- Provide motivation and encouragement for budget adherence

Available Tools:
- generate_budget: Create budget recommendations based on spending history and income
- calculate_category_spending: Analyze spending by category for budget comparison and tracking

Remember: You focus on budgeting, spending limits, and financial planning. For forecasting future cash flow, refer to Forecasting Assistant. For spending trend analysis, refer to Insights Assistant. Your goal is to help users create and maintain realistic, sustainable budgets that align with their financial goals."""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Budget Assistant."""
        return ["generate_budget", "calculate_category_spending"]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "budget_creation_and_management",
            "core_functions": [
                "Create personalized budget recommendations",
                "Analyze spending patterns vs budget goals",
                "Calculate category-specific spending limits",
                "Track budget performance and adherence",
                "Suggest realistic budget adjustments",
            ],
            "budget_frameworks": {
                "primary_rule": "50/30/20 (needs/wants/savings)",
                "adaptation": "Customized based on actual spending patterns",
                "categories": [
                    "housing",
                    "transportation",
                    "food",
                    "entertainment",
                    "savings",
                    "utilities",
                ],
            },
            "guidance_principles": [
                "Realistic expectations over dramatic changes",
                "Gradual improvement strategies",
                "Seasonal spending considerations",
                "Emergency fund importance",
                "Sustainable long-term habits",
            ],
            "analysis_capabilities": {
                "actual_vs_budgeted": True,
                "category_performance": True,
                "trend_identification": True,
                "overspending_alerts": True,
                "success_celebration": True,
            },
        }
