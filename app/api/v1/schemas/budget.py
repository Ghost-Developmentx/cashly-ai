"""
Budget-related schemas for request/response validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class TransactionData(BaseModel):
    """Transaction data for budget analysis."""

    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    amount: float = Field(..., description="Positive for income, negative for expense")
    category: str = Field(default="Uncategorized")
    description: Optional[str] = None

    @field_validator("date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class BudgetRequest(BaseModel):
    """Budget generation request."""

    user_id: str = Field(..., min_length=1)
    transactions: List[TransactionData] = Field(
        ..., min_length=1, description="Historical transactions for analysis"
    )
    monthly_income: Optional[float] = Field(
        None, gt=0, description="Override calculated monthly income"
    )


class CategoryBudget(BaseModel):
    """Budget allocation for a category."""

    category: str
    allocated_amount: float = Field(..., ge=0)
    current_spending: float = Field(..., ge=0)
    difference: float
    percentage_of_income: float = Field(..., ge=0, le=100)


class BudgetRecommendation(BaseModel):
    """Budget recommendation item."""

    type: str = Field(
        ..., pattern="^(reduce_spending|increase_savings|monitor_spending|general)$"
    )
    category: Optional[str] = None
    priority: str = Field(..., pattern="^(high|medium|low)$")
    message: str
    impact: float = Field(..., ge=0)


class SavingsPotential(BaseModel):
    """Potential savings information."""

    monthly: float = Field(..., ge=0)
    annual: float = Field(..., ge=0)
    percentage: float = Field(..., ge=0, le=100)


class BudgetResponse(BaseModel):
    """Complete budget generation response."""

    monthly_income: float = Field(..., ge=0)
    budget_allocations: Dict[str, float]
    category_budgets: Optional[List[CategoryBudget]] = None
    current_spending: Dict[str, float]
    recommendations: List[BudgetRecommendation]
    savings_potential: SavingsPotential
    analysis_period: str
    generated_at: datetime = Field(default_factory=datetime.now)


class BudgetSummary(BaseModel):
    """Simplified budget summary."""

    total_income: float
    total_allocated: float
    total_spending: float
    savings_rate: float = Field(..., ge=0, le=100)
    is_over_budget: bool
    problem_categories: List[str]
