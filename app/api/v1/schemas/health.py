"""
Health check schemas.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ComponentHealth(BaseModel):
    """Individual component health status."""

    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Basic health check response."""

    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    service: str = "cashly-ai-service"
    version: str = "2.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class DetailedHealthResponse(BaseModel):
    """Detailed health check with components."""

    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    service: str = "cashly-ai-service"
    version: str = "2.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, ComponentHealth]
    uptime: Optional[float] = None
