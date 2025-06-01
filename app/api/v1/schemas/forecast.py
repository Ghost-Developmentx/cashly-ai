"""
Cash flow forecast schemas.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ForecastTransaction(BaseModel):
    """Transaction data for forecasting."""

    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    amount: float = Field(..., description="Positive for income, negative for expense")
    category: Optional[str] = "Uncategorized"
    description: Optional[str] = None
    recurring: bool = False

    @field_validator("date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class ForecastRequest(BaseModel):
    """Cash flow forecast request."""

    user_id: str = Field(..., min_length=1)
    transactions: List[ForecastTransaction] = Field(
        ..., min_length=1, description="Historical transactions"
    )
    forecast_days: int = Field(
        default=30, ge=1, le=365, description="Number of days to forecast"
    )


class ScenarioAdjustments(BaseModel):
    """Scenario adjustments for forecasting."""

    income_adjustment: Optional[float] = Field(
        None, description="Monthly income adjustment amount"
    )
    expense_adjustment: Optional[float] = Field(
        None, description="Monthly expense adjustment amount"
    )
    category_adjustments: Optional[Dict[str, float]] = Field(
        None, description="Category-specific adjustments"
    )


class ScenarioForecastRequest(ForecastRequest):
    """Scenario-based forecast request."""

    adjustments: ScenarioAdjustments = Field(
        default_factory=ScenarioAdjustments, description="Scenario adjustments"
    )


class DailyForecast(BaseModel):
    """Daily forecast data point."""

    date: str
    predicted_income: float = Field(..., ge=0)
    predicted_expenses: float = Field(..., ge=0)
    net_change: float
    confidence: float = Field(..., ge=0, le=1)
    running_balance: Optional[float] = None


class ForecastSummary(BaseModel):
    """Forecast summary statistics."""

    projected_income: float = Field(..., ge=0)
    projected_expenses: float = Field(..., ge=0)
    projected_net: float
    ending_balance: float
    confidence_score: float = Field(..., ge=0, le=1)

    @property
    def is_positive(self) -> bool:
        return self.projected_net >= 0


class HistoricalContext(BaseModel):
    """Historical data context."""

    avg_daily_income: float = Field(..., ge=0)
    avg_daily_expenses: float = Field(..., ge=0)
    transaction_count: int = Field(..., ge=0)
    date_range: Optional[str] = None


class ForecastResponse(BaseModel):
    """Complete forecast response."""

    forecast_days: int
    start_date: str
    end_date: str
    daily_forecast: List[DailyForecast]
    summary: ForecastSummary
    historical_context: HistoricalContext
    generated_at: datetime = Field(default_factory=datetime.now)
    scenario: Optional[Dict[str, Any]] = None


class ForecastAccuracy(BaseModel):
    """Forecast accuracy metrics."""

    mean_absolute_error: float
    mean_percentage_error: float
    confidence_interval: Dict[str, float]
    is_reliable: bool
