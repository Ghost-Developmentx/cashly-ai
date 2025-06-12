"""
Categorization schemas for request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Single transaction categorization request."""

    description: str = Field(..., min_length=1, max_length=500)
    amount: float = Field(..., description="Transaction amount")
    merchant: Optional[str] = Field(None, max_length=200)

    @field_validator("amount")
    def amount_not_zero(cls, v):
        if v == 0:
            raise ValueError("Amount cannot be zero")
        return v


class BatchTransactionInput(BaseModel):
    """Batch transaction categorization request."""

    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to categorize",
    )


class CategoryResult(BaseModel):
    """Categorization result for a single transaction."""

    category: str
    confidence: float = Field(..., ge=0, le=1)
    method: str = Field(..., description="Categorization method used")
    error: Optional[str] = None


class CategoryResponse(CategoryResult):
    """Single transaction categorization response."""

    pass


class BatchCategoryResponse(BaseModel):
    """Batch categorization response."""

    categorized_count: int
    results: List[CategoryResult]
    processing_time: Optional[float] = None


class CategoryFeedback(BaseModel):
    """User feedback for improving categorization."""

    description: str = Field(..., min_length=1, max_length=500)
    amount: float
    correct_category: str = Field(..., min_length=1, max_length=100)

    @field_validator("correct_category")
    def validate_category(cls, v):
        # Add valid category validation if needed
        return v.strip().title()


class CategoryStatistics(BaseModel):
    """Category statistics response."""

    total_transactions: int
    categorized: int
    uncategorized: int
    categorization_rate: float
    category_distribution: Dict[str, int]
    top_categories: List[tuple[str, int]]
