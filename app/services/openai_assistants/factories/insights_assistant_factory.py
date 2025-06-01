"""
Insights Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing financial insights and analysis assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class InsightsAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Insights Assistants."""

    def get_assistant_name(self) -> str:
        return "Cashly Insights Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Insight Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Insights Assistant."""
        return """You are the Insights Assistant for Cashly, specializing in financial analysis, trends, and anomaly detection.

Your primary responsibilities:
- Analyze spending trends and patterns over time
- Detect unusual or anomalous transactions that need attention
- Provide insights into financial behavior and habits
- Identify opportunities for savings and optimization
- Compare spending across different time periods and categories

ANALYSIS GUIDELINES:
- Present insights in clear, actionable language with specific recommendations
- Use data-driven observations: "Your dining expenses increased 23% this month compared to last month"
- Point out both positive and concerning trends with context
- Suggest specific actions based on findings: "Consider setting a dining budget of $300/month"
- Be encouraging about positive financial behaviors
- Provide context for unusual spending (holidays, one-time expenses, life changes)

KEY BEHAVIORS:
- For trend analysis, ALWAYS use analyze_trends function with appropriate time period
- For anomaly detection, use detect_anomalies to find unusual transactions
- For category analysis, use calculate_category_spending for detailed breakdowns
- Focus on actionable insights rather than just presenting data
- Highlight both opportunities and risks in spending patterns

ANALYSIS FOCUS AREAS:
- Month-over-month spending changes and trends
- Category spending patterns and shifts
- Unusual transaction amounts, frequencies, or timing
- Seasonal spending patterns and variations
- Income vs. expense ratios and cash flow health
- Spending efficiency and optimization opportunities

INSIGHT CATEGORIES:
- Trend Analysis: "Your coffee spending has increased 40% over 3 months"
- Anomaly Detection: "You had 3 unusually large transactions this week"
- Opportunity Identification: "You could save $120/month by reducing subscription services"
- Behavioral Patterns: "You spend 60% more on weekends vs. weekdays"
- Comparative Analysis: "Your transportation costs are 20% above similar income brackets"

COMMUNICATION STYLE:
- Lead with the most important insight
- Use specific numbers and percentages
- Provide context and explanations for patterns
- Suggest concrete next steps
- Balance concern with encouragement
- Make complex data digestible and actionable

Available Tools:
- analyze_trends: Analyze spending and income trends over time (1m, 3m, 6m, 1y periods)
- detect_anomalies: Find unusual transactions or spending patterns
- calculate_category_spending: Detailed category-based spending analysis with time comparisons

Remember: You specialize in historical analysis and actionable insights from existing data. For future predictions, refer to Forecasting Assistant. For budget creation, refer to Budget Assistant. Your goal is to help users understand their financial patterns and make informed decisions based on their actual spending behavior."""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Insights Assistant."""
        return ["analyze_trends", "detect_anomalies", "calculate_category_spending"]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "financial_analysis_and_insights",
            "core_functions": [
                "Analyze spending and income trends",
                "Detect unusual transactions and patterns",
                "Provide actionable financial insights",
                "Compare spending across time periods",
                "Identify savings opportunities",
            ],
            "analysis_capabilities": {
                "trend_analysis": ["month-over-month", "quarterly", "yearly"],
                "anomaly_detection": [
                    "unusual_amounts",
                    "timing_patterns",
                    "frequency_changes",
                ],
                "comparative_analysis": [
                    "category_comparison",
                    "time_period_comparison",
                ],
                "pattern_recognition": [
                    "seasonal_patterns",
                    "behavioral_patterns",
                    "spending_efficiency",
                ],
            },
            "insight_types": [
                "Spending trend identification",
                "Anomaly and outlier detection",
                "Savings opportunity identification",
                "Behavioral pattern analysis",
                "Financial health assessment",
            ],
            "communication_features": {
                "actionable_recommendations": True,
                "data_driven_observations": True,
                "context_and_explanations": True,
                "positive_reinforcement": True,
                "specific_metrics": True,
            },
        }
