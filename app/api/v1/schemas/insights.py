"""
Financial insights and trend analysis schemas.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class AnalysisPeriod(str, Enum):
    """Valid analysis periods."""

    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    ONE_YEAR = "1y"


class InsightType(str, Enum):
    """Types of insights."""

    SPENDING_TREND = "spending_trend"
    INCOME_PATTERN = "income_pattern"
    CATEGORY_CHANGE = "category_change"
    SAVING_OPPORTUNITY = "saving_opportunity"
    RECURRING_DETECTION = "recurring_detection"
    CATEGORY_CONCENTRATION = "category_concentration"


class TransactionForAnalysis(BaseModel):
    """Transaction data for analysis."""

    date: str = Field(..., description="YYYY-MM-DD format")
    amount: float
    category: str = "Uncategorized"
    description: Optional[str] = None

    @field_validator("date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class TrendAnalysisRequest(BaseModel):
    """Request for trend analysis."""

    user_id: str = Field(..., min_length=1)
    transactions: List[TransactionForAnalysis] = Field(
        ..., min_length=1, description="Transactions to analyze"
    )
    period: AnalysisPeriod = Field(
        default=AnalysisPeriod.THREE_MONTHS, description="Analysis period"
    )


class TrendDirection(str, Enum):
    """Trend direction indicators."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class SpendingTrend(BaseModel):
    """Spending trend information."""

    direction: TrendDirection
    change_percentage: float = Field(..., description="Percentage change")
    average_monthly: float = Field(..., ge=0)
    highest_month: Dict[str, float]
    lowest_month: Dict[str, float]
    volatility_score: float = Field(..., ge=0, le=1)


class CategoryTrend(BaseModel):
    """Category-specific trend."""

    category: str
    total_spent: float = Field(..., ge=0)
    average_transaction: float
    transaction_count: int = Field(..., ge=0)
    percentage_of_total: float = Field(..., ge=0, le=100)
    trend: TrendDirection
    change_percentage: float


class FinancialInsight(BaseModel):
    """Individual financial insight."""

    type: InsightType
    title: str
    description: str
    impact: str = Field(..., description="Financial impact description")
    priority: str = Field(..., pattern="^(high|medium|low)$")
    action_required: bool = False
    metadata: Optional[Dict[str, Any]] = None


class TrendAnalysisResponse(BaseModel):
    """Complete trend analysis response."""

    period_analyzed: str
    date_range: Dict[str, str]
    summary: Dict[str, Any]
    spending_trends: SpendingTrend
    income_trends: Optional[SpendingTrend] = None
    category_trends: List[CategoryTrend]
    insights: List[FinancialInsight]
    patterns: List[Dict[str, Any]]
    generated_at: datetime = Field(default_factory=datetime.now)


class FinancialSummaryRequest(BaseModel):
    """Request for financial summary."""

    user_id: str = Field(..., min_length=1)
    transactions: List[TransactionForAnalysis]
    include_insights: bool = True
    include_predictions: bool = False


class FinancialSummaryResponse(BaseModel):
    """Comprehensive financial summary."""

    user_id: str
    summary: Dict[str, float]
    spending_breakdown: Dict[str, float]
    income_sources: Dict[str, float]
    net_cash_flow: float
    savings_rate: float = Field(..., ge=0, le=100)
    financial_health_score: float = Field(..., ge=0, le=100)
    insights: Optional[List[FinancialInsight]] = None
    recommendations: List[str]
