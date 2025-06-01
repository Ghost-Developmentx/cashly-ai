"""
Financial insights and trend analysis endpoints.
Replaces Flask InsightsController.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from logging import getLogger

from app.core.dependencies import get_insight_service
from app.core.exceptions import ValidationError
from app.api.v1.schemas.insights import (
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    FinancialSummaryRequest,
    FinancialSummaryResponse,
    CategoryTrend,
    FinancialInsight,
)
from app.services.insights import AsyncInsightService

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    "/trends",
    response_model=TrendAnalysisResponse,
    summary="Analyze financial trends",
    description="Analyze spending patterns and financial trends",
)
async def analyze_trends(
    request: TrendAnalysisRequest,
    service: AsyncInsightService = Depends(get_insight_service),
) -> TrendAnalysisResponse:
    """Analyze financial trends and patterns."""
    try:
        # Convert transactions to dict format
        transactions = [t.model_dump() for t in request.transactions]

        logger.info(
            f"Analyzing {request.period} trends for user {request.user_id} "
            f"with {len(transactions)} transactions"
        )

        result = await service.analyze_trends(
            user_id=request.user_id,
            transactions=transactions,
            period=request.period.value,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Format category trends
        category_trends = [
            CategoryTrend(
                category=cat,
                total_spent=data["total"],
                average_transaction=data["average"],
                transaction_count=data["count"],
                percentage_of_total=data["percentage"],
                trend=data.get("trend", "stable"),
                change_percentage=data.get("change", 0),
            )
            for cat, data in result.get("category_analysis", {}).items()
        ]

        # Format insights
        insights = [
            FinancialInsight(**insight) for insight in result.get("insights", [])
        ]

        response = TrendAnalysisResponse(
            period_analyzed=request.period.value,
            date_range=result.get("date_range", {}),
            summary=result.get("summary", {}),
            spending_trends=result.get("spending_trends", {}),
            income_trends=result.get("income_trends"),
            category_trends=category_trends,
            insights=insights,
            patterns=result.get("patterns", []),
        )

        logger.info(f"Generated {len(insights)} insights for user {request.user_id}")

        return response

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze trends: {str(e)}"
        )


@router.post(
    "/summary",
    response_model=FinancialSummaryResponse,
    summary="Get financial summary",
    description="Get comprehensive financial summary with insights",
)
async def get_financial_summary(
    request: FinancialSummaryRequest,
    service: AsyncInsightService = Depends(get_insight_service),
) -> FinancialSummaryResponse:
    """Get comprehensive financial summary."""
    try:
        transactions = [t.model_dump() for t in request.transactions]

        # Analyze trends for summary
        trends_result = await service.analyze_trends(
            user_id=request.user_id,
            transactions=transactions,
            period="3m",  # Default to 3 months
        )

        if "error" in trends_result:
            raise HTTPException(status_code=400, detail=trends_result["error"])

        # Calculate financial metrics
        total_income = sum(t.amount for t in request.transactions if t.amount > 0)
        total_expenses = sum(
            abs(t.amount) for t in request.transactions if t.amount < 0
        )
        net_cash_flow = total_income - total_expenses

        # Calculate savings rate
        savings_rate = (net_cash_flow / total_income * 100) if total_income > 0 else 0

        # Calculate financial health score (simplified)
        health_score = min(
            100, max(0, 50 + (savings_rate * 0.5) + (25 if net_cash_flow > 0 else -25))
        )

        # Get spending breakdown
        spending_breakdown = {}
        income_sources = {}

        for t in request.transactions:
            if t.amount < 0:
                cat = t.category
                spending_breakdown[cat] = spending_breakdown.get(cat, 0) + abs(t.amount)
            else:
                source = t.category or "Other Income"
                income_sources[source] = income_sources.get(source, 0) + t.amount

        # Generate recommendations
        recommendations = _generate_recommendations(
            savings_rate, health_score, trends_result
        )

        response = FinancialSummaryResponse(
            user_id=request.user_id,
            summary={
                "total_income": round(total_income, 2),
                "total_expenses": round(total_expenses, 2),
                "net_cash_flow": round(net_cash_flow, 2),
                "transaction_count": len(transactions),
            },
            spending_breakdown=spending_breakdown,
            income_sources=income_sources,
            net_cash_flow=round(net_cash_flow, 2),
            savings_rate=round(savings_rate, 1),
            financial_health_score=round(health_score, 1),
            insights=(
                trends_result.get("insights") if request.include_insights else None
            ),
            recommendations=recommendations,
        )

        return response

    except Exception as e:
        logger.error(f"Financial summary failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate financial summary: {str(e)}"
        )


@router.get(
    "/categories/{category}",
    response_model=Dict[str, Any],
    summary="Get category insights",
    description="Get detailed insights for a specific category",
)
async def get_category_insights(
    category: str,
    user_id: str = Query(..., min_length=1),
    days: int = Query(default=90, ge=30, le=365),
    service: AsyncInsightService = Depends(get_insight_service),
) -> Dict[str, Any]:
    """Get detailed insights for a specific spending category."""
    # This would fetch and analyze transactions for the category
    # Simplified implementation
    return {
        "category": category,
        "analysis_period_days": days,
        "statistics": {
            "total_spent": 1500.00,
            "average_transaction": 50.00,
            "transaction_count": 30,
            "highest_day": {"date": "2024-01-15", "amount": 250.00},
            "lowest_day": {"date": "2024-01-20", "amount": 10.00},
        },
        "trends": {
            "direction": "decreasing",
            "change_percentage": -15.5,
            "projected_next_month": 1275.00,
        },
        "insights": [
            {
                "type": "saving_opportunity",
                "title": f"Reduce {category} spending",
                "description": f"Your {category} spending has decreased by 15.5%",
                "impact": "Could save $225 next month",
                "priority": "medium",
            }
        ],
    }


def _generate_recommendations(
    savings_rate: float, health_score: float, trends_result: Dict[str, Any]
) -> List[str]:
    """Generate personalized recommendations."""
    recommendations = []

    if savings_rate < 10:
        recommendations.append(
            "Aim to save at least 10-20% of your income for financial security"
        )

    if health_score < 50:
        recommendations.append(
            "Focus on reducing expenses to improve your financial health"
        )

    # Check for high spending categories
    spending_trends = trends_result.get("spending_trends", {})
    if spending_trends.get("direction") == "increasing":
        recommendations.append(
            "Your spending is trending upward - consider reviewing your budget"
        )

    if not recommendations:
        recommendations.append(
            "You're doing well! Keep monitoring your spending patterns"
        )

    return recommendations[:3]  # Top 3 recommendations
