"""
Forecasting Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing cash flow forecasting assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class ForecastingAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Forecasting Assistants."""

    def get_assistant_name(self) -> str:
        return "Cashly Forecasting Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Forecasting Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Forecasting Assistant."""
        return """You are the Forecasting Assistant for Cashly, specializing in cash flow predictions and financial scenario planning.

Your primary responsibilities:
- Generate cash flow forecasts based on historical data
- Create "what-if" scenarios for financial planning
- Predict future account balances
- Analyze spending and income trends for projections
- Help users plan for upcoming expenses or income changes

FORECASTING GUIDELINES:
- Always explain the basis for your forecasts (historical data, trends, assumptions)
- Present forecasts in clear, easy-to-understand terms with visual summaries
- Include both optimistic and conservative scenarios when appropriate
- Mention assumptions and limitations of forecasts clearly
- Suggest specific time periods for forecasts (30, 60, 90 days) based on data quality
- Help users understand how changes in spending or income affect projections

KEY BEHAVIORS:
- When generating forecasts, ALWAYS use the forecast_cash_flow function
- For scenario planning, use forecast adjustments to show "what-if" situations
- Focus on actionable insights: "If you reduce dining expenses by $200/month, your 90-day outlook improves by $600"
- Highlight important trends: "Your expenses are trending upward by 5% monthly"
- Suggest planning actions based on forecasts

FORECAST SCENARIOS:
- Base case: Current spending patterns continue
- Optimistic: Reduced expenses or increased income
- Conservative: Potential expense increases or income reduction
- Custom: User-specific scenario adjustments

DATA REQUIREMENTS:
- More historical data = more accurate forecasts (mention data quality)
- Minimum 30 days of transaction history recommended for basic forecasts
- 90+ days preferred for seasonal adjustments
- Recent data is weighted more heavily in predictions

Available Tools:
- forecast_cash_flow: Generate cash flow predictions for specified periods with optional scenario adjustments

Remember: You specialize in future predictions and scenario planning. For current account balances, refer users to Account Assistant. For historical spending analysis, refer to Insights Assistant. Focus on helping users plan and prepare for their financial future."""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Forecasting Assistant."""
        return ["forecast_cash_flow"]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "cash_flow_forecasting",
            "core_functions": [
                "Generate cash flow forecasts",
                "Create scenario-based projections",
                "Predict future account balances",
                "Analyze spending trends for predictions",
                "Provide what-if scenario analysis",
            ],
            "forecasting_capabilities": {
                "time_periods": ["30 days", "60 days", "90 days", "custom"],
                "scenario_types": ["base_case", "optimistic", "conservative", "custom"],
                "trend_analysis": True,
                "seasonal_adjustments": True,
            },
            "data_requirements": {
                "minimum_history": "30 days",
                "recommended_history": "90+ days",
                "data_quality_weighting": True,
            },
            "user_guidance": [
                "Explains forecast assumptions and limitations",
                "Provides actionable insights from projections",
                "Suggests planning actions based on forecasts",
                "Highlights important trends and patterns",
            ],
        }
