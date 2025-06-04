"""
Async financial insights and analytics service.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .trend_analyzer import TrendAnalyzer
from .pattern_detector import PatternDetector
from .insight_generator import InsightGenerator

logger = logging.getLogger(__name__)


class AsyncInsightService:
    """
    Asynchronous financial trend and insight analysis service.

    This service provides methods for analyzing financial trends, detecting patterns, and
    generating comprehensive insights from transactional data. It processes user transaction
    history to provide meaningful financial insights such as spending trends, income trends,
    recurring patterns, and financial health metrics. The main goal of this service is to
    help users better understand and improve their financial health.

    Attributes
    ----------
    trend_analyzer : TrendAnalyzer
        Component responsible for analyzing financial spending and income trends.
    pattern_detector : PatternDetector
        Component responsible for detecting recurring financial patterns.
    insight_generator : InsightGenerator
        Component responsible for generating insights based on trends and patterns.
    """

    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.pattern_detector = PatternDetector()
        self.insight_generator = InsightGenerator()

    async def analyze_trends(
            self, user_id: str, transactions: List[Dict[str, Any]], period: str = "3m"
    ) -> Dict[str, Any]:
        """
        Analyze financial trends.

        Args:
            user_id: User identifier
            transactions: Transaction history
            period: Analysis period

        Returns:
            Trend analysis results
        """
        try:
            if not transactions:
                return self._empty_analysis_response()

            # Clean and validate transaction data
            cleaned_transactions = self._clean_transactions(transactions)

            if not cleaned_transactions:
                return self._empty_analysis_response()

            # Parse period
            days = self._parse_period(period)
            filtered_transactions = self._filter_by_period(cleaned_transactions, days)

            # Analyze trends
            spending_trends = await self.trend_analyzer.analyze_spending_trends(
                filtered_transactions
            )

            income_trends = await self.trend_analyzer.analyze_income_trends(
                filtered_transactions
            )

            # Detect patterns
            patterns = await self.pattern_detector.detect_patterns(
                filtered_transactions
            )

            # Generate insights
            insights = await self.insight_generator.generate_insights(
                spending_trends, income_trends, patterns
            )

            return {
                "period": period,
                "transaction_count": len(filtered_transactions),
                "spending_trends": spending_trends,
                "income_trends": income_trends,
                "patterns": patterns,
                "insights": insights,
                "summary": self._create_summary(spending_trends, income_trends),
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": f"Failed to analyze trends: {str(e)}", "period": period}

    @staticmethod
    def _clean_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate transaction data."""
        cleaned = []

        for txn in transactions:
            try:
                # Clean and validate required fields
                cleaned_txn = txn.copy()

                # Ensure category is a string
                if cleaned_txn.get('category') is None:
                    cleaned_txn['category'] = 'Other'
                elif not isinstance(cleaned_txn['category'], str):
                    cleaned_txn['category'] = str(cleaned_txn['category'])
                else:
                    cleaned_txn['category'] = cleaned_txn['category'].strip()

                # Ensure description is a string
                if cleaned_txn.get('description') is None:
                    cleaned_txn['description'] = 'Transaction'
                elif not isinstance(cleaned_txn['description'], str):
                    cleaned_txn['description'] = str(cleaned_txn['description'])
                else:
                    cleaned_txn['description'] = cleaned_txn['description'].strip()

                # Ensure amount is numeric
                if 'amount' in cleaned_txn:
                    try:
                        cleaned_txn['amount'] = float(cleaned_txn['amount'])
                    except (TypeError, ValueError):
                        continue  # Skip invalid amounts
                else:
                    continue  # Skip transactions without amounts

            except Exception as e:
                logger.warning(f"Skipping invalid transaction: {e}")
                continue

        return cleaned


    def _create_summary(
        self, spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create analysis summary."""
        return {
            "average_monthly_income": income_trends.get("monthly_average", 0),
            "average_monthly_spending": spending_trends.get("monthly_average", 0),
            "trend_direction": self._determine_trend_direction(
                spending_trends, income_trends
            ),
            "financial_health_score": self._calculate_health_score(
                spending_trends, income_trends
            ),
            "key_findings": self._extract_key_findings(spending_trends, income_trends),
        }

    @staticmethod
    def _determine_trend_direction(
        spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> str:
        """Determine overall financial trend direction."""
        spending_change = spending_trends.get("trend_percentage", 0)
        income_change = income_trends.get("trend_percentage", 0)

        if income_change > spending_change + 5:
            return "improving"
        elif spending_change > income_change + 5:
            return "declining"
        else:
            return "stable"

    @staticmethod
    def _calculate_health_score(
        spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> float:
        """Calculate financial health score (0-100)."""
        # Simple scoring based on income vs spending
        income = income_trends.get("monthly_average", 0)
        spending = spending_trends.get("monthly_average", 0)

        if income == 0:
            return 0.0

        savings_rate = (income - spending) / income
        base_score = min(savings_rate * 200, 100)  # 50% savings = 100 score

        # Adjust for trend
        trend_adjustment = 0
        if income_trends.get("trend_percentage", 0) > 0:
            trend_adjustment += 10
        if spending_trends.get("trend_percentage", 0) < 0:
            trend_adjustment += 10

        return min(100, max(0, base_score + trend_adjustment))

    @staticmethod
    def _extract_key_findings(
        spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from analysis."""
        findings = []

        # Spending findings
        if spending_trends.get("trend_percentage", 0) > 10:
            findings.append("Spending has increased significantly")
        elif spending_trends.get("trend_percentage", 0) < -10:
            findings.append("Spending has decreased significantly")

        # Income findings
        if income_trends.get("trend_percentage", 0) > 10:
            findings.append("Income has increased significantly")
        elif income_trends.get("trend_percentage", 0) < -10:
            findings.append("Income has decreased significantly")

        # Category findings
        top_category = spending_trends.get("top_category")
        if top_category:
            findings.append(f"Highest spending category: {top_category}")

        return findings[:3]  # Limit to top 3 findings

    @staticmethod
    def _parse_period(period: str) -> int:
        """Parse period string to days."""
        period_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
        return period_map.get(period, 90)

    @staticmethod
    def _filter_by_period(
        transactions: List[Dict[str, Any]], days: int
    ) -> List[Dict[str, Any]]:
        """Filter transactions by period."""
        cutoff_date = datetime.now().date() - timedelta(days=days)

        filtered = []
        for txn in transactions:
            try:
                txn_date = datetime.strptime(txn["date"], "%Y-%m-%d").date()
                if txn_date >= cutoff_date:
                    filtered.append(txn)
            except ValueError:
                continue

        return filtered

    @staticmethod
    def _empty_analysis_response() -> Dict[str, Any]:
        """Return empty analysis response."""
        return {
            "message": "No transaction data available for analysis",
            "spending_trends": {},
            "income_trends": {},
            "patterns": [],
            "insights": [],
        }
