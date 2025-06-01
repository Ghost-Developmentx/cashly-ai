"""
Base schemas for API responses.
Ensures Rails compatibility and consistent structure.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Generic, TypeVar
from datetime import datetime


T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model for all API responses."""

    class Config:
        """Pydantic config for all responses."""

        # Ensure datetime serialization matches Rails
        json_encoders = {datetime: lambda v: v.isoformat()}
        # Allow ORM mode for SQLAlchemy models
        from_attributes = True


class SuccessResponse(BaseResponse, Generic[T]):
    """Generic success response wrapper."""

    success: bool = True
    data: T
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    """Standard error response."""

    success: bool = False
    error: str
    detail: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated list response."""

    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


class Transaction(BaseResponse):
    """Base transaction model matching Rails structure."""

    id: Optional[str] = None
    account_id: str
    account_name: Optional[str] = None
    amount: float
    description: str
    category: str = "Uncategorized"
    date: str  # Keep as string for Rails compatibility
    recurring: bool = False
    created_via_ai: bool = False
    metadata: Optional[Dict[str, Any]] = None
