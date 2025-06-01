"""
API v1 route aggregator.
Replaces Flask's blueprint system.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    health,
    categorization,
    forecast,
    budget,
    insights,
    anomaly,
    accounts,
    fin,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])

api_router.include_router(
    categorization.router, prefix="/categorize", tags=["categorization"]
)

api_router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])

api_router.include_router(budget.router, prefix="/budget", tags=["budget"])

api_router.include_router(insights.router, prefix="/insights", tags=["insights"])

api_router.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])

api_router.include_router(accounts.router, prefix="/accounts", tags=["accounts"])

api_router.include_router(fin.router, prefix="/fin", tags=["fin"])
