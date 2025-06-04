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

            # Debug: Check what we received
            logger.info(f"Received {len(transactions)} transactions")
            logger.info(f"First transaction type: {type(transactions[0]) if transactions else 'None'}")

            # Clean and validate transaction data
            cleaned_transactions = self._clean_transactions(transactions)

            if not cleaned_transactions:
                return self._empty_analysis_response()

            logger.info(f"After cleaning: {len(cleaned_transactions)} transactions")

            # Parse period
            days = self._parse_period(period)
            filtered_transactions = self._filter_by_period(cleaned_transactions, days)

            logger.info(f"After filtering: {len(filtered_transactions)} transactions")

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

            # Calculate category analysis for the response
            category_analysis = self._calculate_category_analysis(filtered_transactions)

            # Calculate date range
            date_range = self._calculate_date_range(filtered_transactions)

            return {
                "period": period,
                "transaction_count": len(filtered_transactions),
                "date_range": date_range,
                "spending_trends": self._format_spending_trend(spending_trends),
                "income_trends": self._format_income_trend(income_trends) if income_trends else None,
                "category_analysis": category_analysis,
                "patterns": patterns,
                "insights": insights,
                "summary": self._create_summary(spending_trends, income_trends),
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}", exc_info=True)
            return {"error": f"Failed to analyze trends: {str(e)}", "period": period}

    @staticmethod
    def _format_spending_trend(spending_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Format spending trends to match SpendingTrend schema."""
        if not spending_trends:
            return {
                "direction": "stable",
                "change_percentage": 0.0,
                "average_monthly": 0.0,
                "highest_month": {"month": 0.0, "amount": 0.0},
                "lowest_month": {"month": 0.0, "amount": 0.0},
                "volatility_score": 0.0
            }

        # Handle highest_month - convert month string to numeric representation
        highest_month = spending_trends.get("highest_month")
        if isinstance(highest_month, str) and highest_month:
            # Convert "YYYY-MM" to a float like 202505.0 for 2025-05
            try:
                year, month = highest_month.split("-")
                month_numeric = float(year) * 100 + float(month)
            except (ValueError, AttributeError):
                month_numeric = 0.0
            # Find the amount for this month
            monthly_data = spending_trends.get("monthly_data", {})
            highest_amount = monthly_data.get(highest_month, 0.0)
            highest_month_data = {"month": month_numeric, "amount": highest_amount}
        elif isinstance(highest_month, dict):
            highest_month_data = highest_month
        else:
            highest_month_data = {"month": 0.0, "amount": 0.0}

        # Handle lowest_month - convert month string to numeric representation
        lowest_month = spending_trends.get("lowest_month")
        if isinstance(lowest_month, str) and lowest_month:
            # Convert "YYYY-MM" to a float like 202505.0 for 2025-05
            try:
                year, month = lowest_month.split("-")
                month_numeric = float(year) * 100 + float(month)
            except (ValueError, AttributeError):
                month_numeric = 0.0
            # Find the amount for this month
            monthly_data = spending_trends.get("monthly_data", {})
            lowest_amount = monthly_data.get(lowest_month, 0.0)
            lowest_month_data = {"month": month_numeric, "amount": lowest_amount}
        elif isinstance(lowest_month, dict):
            lowest_month_data = lowest_month
        else:
            lowest_month_data = {"month": 0.0, "amount": 0.0}

        return {
            "direction": spending_trends.get("trend_direction", "stable"),
            "change_percentage": spending_trends.get("trend_percentage", 0.0),
            "average_monthly": spending_trends.get("monthly_average", 0.0),
            "highest_month": highest_month_data,
            "lowest_month": lowest_month_data,
            "volatility_score": min(1.0, spending_trends.get("volatility", 0.0) / 100.0)
        }

    @staticmethod
    def _format_income_trend(income_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Format income trends to match SpendingTrend schema."""
        if not income_trends:
            return None

        # Handle highest_month - convert month string to numeric representation
        highest_month = income_trends.get("highest_month")
        if isinstance(highest_month, str) and highest_month:
            # Convert "YYYY-MM" to a float like 202505.0 for 2025-05
            try:
                year, month = highest_month.split("-")
                month_numeric = float(year) * 100 + float(month)
            except (ValueError, AttributeError):
                month_numeric = 0.0
            # Find the amount for this month
            monthly_data = income_trends.get("monthly_data", {})
            highest_amount = monthly_data.get(highest_month, 0.0)
            highest_month_data = {"month": month_numeric, "amount": highest_amount}
        elif isinstance(highest_month, dict):
            highest_month_data = highest_month
        else:
            highest_month_data = {"month": 0.0, "amount": 0.0}

        # Handle lowest_month - convert month string to numeric representation
        lowest_month = income_trends.get("lowest_month")
        if isinstance(lowest_month, str) and lowest_month:
            # Convert "YYYY-MM" to a float like 202505.0 for 2025-05
            try:
                year, month = lowest_month.split("-")
                month_numeric = float(year) * 100 + float(month)
            except (ValueError, AttributeError):
                month_numeric = 0.0
            # Find the amount for this month
            monthly_data = income_trends.get("monthly_data", {})
            lowest_amount = monthly_data.get(lowest_month, 0.0)
            lowest_month_data = {"month": month_numeric, "amount": lowest_amount}
        elif isinstance(lowest_month, dict):
            lowest_month_data = lowest_month
        else:
            lowest_month_data = {"month": 0.0, "amount": 0.0}

        return {
            "direction": income_trends.get("trend_direction", "stable"),
            "change_percentage": income_trends.get("trend_percentage", 0.0),
            "average_monthly": income_trends.get("monthly_average", 0.0),
            "highest_month": highest_month_data,
            "lowest_month": lowest_month_data,
            "volatility_score": min(1.0, income_trends.get("volatility", 0.0) / 100.0)
        }

    @staticmethod
    def _calculate_category_analysis(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate category analysis for response."""
        category_totals = {}
        category_counts = {}
        total_spending = 0

        # Only include expenses for category analysis
        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses
                amount = abs(float(txn["amount"]))
                category = txn.get("category", "Other")

                category_totals[category] = category_totals.get(category, 0) + amount
                category_counts[category] = category_counts.get(category, 0) + 1
                total_spending += amount

        category_analysis = {}
        for category, total in category_totals.items():
            category_analysis[category] = {
                "total": round(total, 2),
                "average": round(total / category_counts[category], 2),
                "count": category_counts[category],
                "percentage": round((total / total_spending * 100), 2) if total_spending > 0 else 0,
                "trend": "stable",  # Simplified
                "change": 0  # Simplified
            }

        return category_analysis

    @staticmethod
    def _calculate_date_range(transactions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate date range from transactions."""
        if not transactions:
            return {"start": "N/A", "end": "N/A"}

        dates = [txn["date"] for txn in transactions]
        return {
            "start": min(dates),
            "end": max(dates)
        }


    @staticmethod
    def _clean_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate transaction data."""
        cleaned = []

        # Debug: Check input type
        logger.info(f"_clean_transactions received: {type(transactions)}")
        if transactions:
            logger.info(f"First item type: {type(transactions[0])}")
            logger.info(f"First item value: {transactions[0]}")

        # Ensure transactions is a list
        if not isinstance(transactions, list):
            logger.error(f"Expected list of transactions, got {type(transactions)}")
            return []

        for i, txn in enumerate(transactions):
            try:
                # Ensure txn is a dictionary
                if not isinstance(txn, dict):
                    logger.warning(f"Skipping non-dict transaction at index {i}: {type(txn)} - {txn}")
                    continue

                # Clean and validate required fields
                cleaned_txn = txn.copy()

                # Ensure date is present and valid first (required for trend analysis)
                if 'date' not in cleaned_txn:
                    logger.warning(f"Skipping transaction {i}: missing date field")
                    continue

                # Ensure amount is numeric and present
                if 'amount' not in cleaned_txn:
                    logger.warning(f"Skipping transaction {i}: missing amount field")
                    continue

                try:
                    cleaned_txn['amount'] = float(cleaned_txn['amount'])
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping transaction {i}: invalid amount {cleaned_txn.get('amount')} - {e}")
                    continue

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

                # Only append if we get here (all validations passed)
                cleaned.append(cleaned_txn)

            except Exception as e:
                logger.warning(f"Skipping invalid transaction at index {i}: {e}")
                continue

        logger.info(f"Cleaned {len(cleaned)} transactions from {len(transactions)} input")
        return cleaned

    def _create_summary(
            self, spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an analysis summary."""
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
            "date_range": {"start": "N/A", "end": "N/A"},
            "spending_trends": {
                "direction": "stable",
                "change_percentage": 0.0,
                "average_monthly": 0.0,
                "highest_month": {"month": "N/A", "amount": 0.0},
                "lowest_month": {"month": "N/A", "amount": 0.0},
                "volatility_score": 0.0
            },
            "income_trends": {
                "direction": "stable",
                "change_percentage": 0.0,
                "average_monthly": 0.0,
                "highest_month": {"month": "N/A", "amount": 0.0},
                "lowest_month": {"month": "N/A", "amount": 0.0},
                "volatility_score": 0.0
            },
            "category_analysis": {},
            "patterns": [],
            "insights": [],
            "summary": {
                "average_monthly_income": 0,
                "average_monthly_spending": 0,
                "trend_direction": "stable",
                "financial_health_score": 0,
                "key_findings": []
            }
        }
