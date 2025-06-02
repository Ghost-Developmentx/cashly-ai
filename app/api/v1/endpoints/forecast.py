"""
Cash flow forecasting endpoints.
Replaces Flask ForecastController.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from logging import getLogger
from datetime import datetime, timedelta

from app.core.dependencies import get_forecast_service
from app.core.exceptions import ValidationError
from app.api.v1.schemas.forecast import (
    ForecastRequest,
    ScenarioForecastRequest,
    ForecastResponse,
    DailyForecast,
    ForecastAccuracy,
)
from app.api.v1.schemas.base import SuccessResponse, ErrorResponse
from app.services.forecast import AsyncForecastService

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    "/cash_flow",
    response_model=ForecastResponse,
    summary="Generate cash flow forecast",
    description="Forecast future cash flow based on historical transactions",
)
async def forecast_cash_flow(
    request: ForecastRequest,
    service: AsyncForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    """Generate cash flow forecast."""
    try:
        # Convert transactions to dict format
        transactions = [t.model_dump() for t in request.transactions]

        logger.info(
            f"Generating {request.forecast_days}-day forecast for "
            f"user {request.user_id} with {len(transactions)} transactions"
        )

        result = await service.forecast_cash_flow(
            user_id=request.user_id,
            transactions=transactions,
            forecast_days=request.forecast_days,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Add running balance to daily forecasts
        running_balance = 0
        enhanced_daily = []
        for day in result["daily_forecast"]:
            running_balance += day["net_change"]
            enhanced_daily.append(DailyForecast(**day, running_balance=running_balance))

        response = ForecastResponse(
            forecast_days=result["forecast_days"],
            start_date=result["start_date"],
            end_date=result["end_date"],
            daily_forecast=enhanced_daily,
            summary=result["summary"],
            historical_context=result["historical_context"],
        )

        logger.info(
            f"Generated forecast with confidence {response.summary.confidence_score}"
        )

        return response

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate forecast: {str(e)}"
        )


@router.post(
    "/cash_flow/scenario",
    response_model=ForecastResponse,
    summary="Generate scenario-based forecast",
    description="Forecast with what-if adjustments",
)
async def forecast_scenario(
    request: ScenarioForecastRequest,
    service: AsyncForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    """Generate scenario-based forecast."""
    try:
        transactions = [t.model_dump() for t in request.transactions]
        adjustments = request.adjustments.model_dump(exclude_none=True)

        logger.info(f"Generating scenario forecast with adjustments: {adjustments}")

        result = await service.forecast_cash_flow_scenario(
            user_id=request.user_id,
            transactions=transactions,
            forecast_days=request.forecast_days,
            adjustments=adjustments,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Process same as regular forecast
        running_balance = 0
        enhanced_daily = []
        for day in result["daily_forecast"]:
            running_balance += day["net_change"]
            enhanced_daily.append(DailyForecast(**day, running_balance=running_balance))

        response = ForecastResponse(
            forecast_days=result["forecast_days"],
            start_date=result["start_date"],
            end_date=result["end_date"],
            daily_forecast=enhanced_daily,
            summary=result["summary"],
            historical_context=result["historical_context"],
            scenario=result.get("scenario"),
        )

        return response

    except Exception as e:
        logger.error(f"Scenario forecast failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate scenario forecast: {str(e)}"
        )


@router.post(
    "/accuracy",
    response_model=ForecastAccuracy,
    summary="Calculate forecast accuracy",
    description="Analyze forecast reliability based on historical data",
)
async def calculate_accuracy(
    historical_transactions: List[Dict[str, Any]],
    forecast_days: int = Query(default=30, ge=1, le=90),
) -> ForecastAccuracy:
    """Calculate forecast accuracy metrics."""
    try:
        # Simple accuracy calculation
        # In production, this would compare past forecasts with actuals

        transaction_count = len(historical_transactions)
        data_days = 30  # Simplified

        # Calculate confidence based on data quality
        confidence = min(transaction_count / 100, 1.0) * 0.8

        # Mock accuracy metrics
        mae = 50.0 - (confidence * 30)  # Better confidence = lower error
        mpe = 10.0 - (confidence * 5)

        return ForecastAccuracy(
            mean_absolute_error=round(mae, 2),
            mean_percentage_error=round(mpe, 2),
            confidence_interval={"lower": 0.8, "upper": 1.2},
            is_reliable=confidence > 0.7,
        )

    except Exception as e:
        logger.error(f"Accuracy calculation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate accuracy: {str(e)}"
        )


@router.get(
    "/patterns",
    response_model=Dict[str, Any],
    summary="Get spending patterns",
    description="Analyze historical spending patterns",
)
async def get_spending_patterns(
    user_id: str,
    days: int = Query(default=90, ge=30, le=365),
    service: AsyncForecastService = Depends(get_forecast_service),
) -> Dict[str, Any]:
    """Analyze spending patterns from historical data."""
    # This would be implemented with actual pattern analysis
    return {
        "weekly_pattern": {
            "Monday": 1.2,
            "Tuesday": 0.9,
            "Wednesday": 1.0,
            "Thursday": 0.8,
            "Friday": 1.5,
            "Saturday": 1.3,
            "Sunday": 0.7,
        },
        "monthly_pattern": {"week_1": 1.3, "week_2": 0.9, "week_3": 0.8, "week_4": 1.1},
        "recurring_transactions": [
            {
                "description": "Subscription",
                "amount": -50.0,
                "frequency": "monthly",
                "day_of_month": 1,
            }
        ],
    }
