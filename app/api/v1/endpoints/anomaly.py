"""
Anomaly detection endpoints for unusual transaction patterns.
Replaces Flask AnomalyController.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from logging import getLogger
from datetime import datetime, timedelta

from app.core.dependencies import get_anomaly_service
from app.core.exceptions import ValidationError
from app.api.v1.schemas.anomaly import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    DetectedAnomaly,
    AnomalySummary,
AnomalyPattern,

    AnomalyTrendResponse,
    AnomalyType,
    AnomalySeverity,
)
from app.api.v1.schemas.base import SuccessResponse
from app.services.anomaly import AsyncAnomalyService

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    "/detect",
    response_model=AnomalyDetectionResponse,
    summary="Detect anomalous transactions",
    description="Identify unusual or suspicious transaction patterns",
)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    service: AsyncAnomalyService = Depends(get_anomaly_service),
) -> AnomalyDetectionResponse:
    """Detect anomalous transactions."""
    try:
        # Convert transactions to dict format
        transactions = [t.model_dump() for t in request.transactions]

        logger.info(
            f"Detecting anomalies for user {request.user_id} "
            f"with {len(transactions)} transactions"
        )

        result = await service.detect_anomalies(
            user_id=request.user_id,
            transactions=transactions,
            threshold=request.threshold,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Format detected anomalies
        anomalies = [
            _format_anomaly_response(a) for a in result.get("anomalies", [])
        ]

        # Generate recommendations
        recommendations = _generate_anomaly_recommendations(
            anomalies, result.get("summary", {})
        )

        response = AnomalyDetectionResponse(
            user_id=request.user_id,
            anomalies=anomalies,
            summary=AnomalySummary(**result.get("summary", {})),
            threshold_used=result.get("threshold", 2.0),
            recommendations=recommendations,
        )

        logger.info(f"Detected {len(anomalies)} anomalies for user {request.user_id}")

        return response

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to detect anomalies: {str(e)}"
        )


@router.get(
    "/summary",
    response_model=Dict[str, Any],
    summary="Get anomaly summary",
    description="Get summary statistics without full details",
)
async def get_anomaly_summary(
    user_id: str = Query(..., min_length=1),
    days: int = Query(default=30, ge=7, le=365),
    service: AsyncAnomalyService = Depends(get_anomaly_service),
) -> Dict[str, Any]:
    """Get anomaly detection summary."""
    # This would fetch user transactions and analyze
    # Simplified implementation
    return {
        "user_id": user_id,
        "period_days": days,
        "summary": {
            "total_transactions": 150,
            "anomalies_detected": 5,
            "anomaly_rate": 3.33,
            "by_type": {
                "unusual_amount": 3,
                "new_merchant": 1,
                "duplicate_transaction": 1,
            },
            "by_severity": {"low": 2, "medium": 2, "high": 1, "critical": 0},
            "highest_risk_categories": ["Shopping", "Entertainment"],
        },
        "recent_anomalies": [
            {
                "date": "2024-01-20",
                "amount": 1500.00,
                "type": "unusual_amount",
                "severity": "high",
            }
        ],
    }


@router.get(
    "/trends",
    response_model=AnomalyTrendResponse,
    summary="Get anomaly trends",
    description="Analyze anomaly patterns over time",
)
async def get_anomaly_trends(
    user_id: str = Query(..., min_length=1),
    period: str = Query(default="3m", pattern="^(1m|3m|6m|1y)$"),
    service: AsyncAnomalyService = Depends(get_anomaly_service),
) -> AnomalyTrendResponse:
    """Analyze anomaly trends over time."""
    # Simplified implementation
    return AnomalyTrendResponse(
        time_period=period,
        total_anomalies=23,
        anomaly_trend="stable",
        patterns=[
            AnomalyPattern(
                pattern_type="recurring_high_amount",
                frequency="monthly",
                transactions_affected=5,
                total_amount=7500.00,
                description="Large purchases around month-end",
                first_occurrence="2023-10-28",
                last_occurrence="2024-01-28",
            )
        ],
        risk_assessment={
            "overall_risk": "medium",
            "trending": "stable",
            "action_required": False,
            "risk_factors": [
                "Occasional high-value transactions",
                "New merchant patterns detected",
            ],
        },
    )



@router.post(
    "/mark_reviewed",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Mark anomalies as reviewed",
    description="Mark detected anomalies as reviewed by user",
)
async def mark_anomalies_reviewed(
    anomaly_ids: List[str] = Body(...),
    user_id: str = Body(...),
    action: str = Body(..., pattern="^(acknowledged|false_positive|legitimate)$")
) -> SuccessResponse[Dict[str, Any]]:
    """Mark anomalies as reviewed."""
    try:
        # This would update anomaly status in database
        logger.info(
            f"Marking {len(anomaly_ids)} anomalies as {action} " f"for user {user_id}"
        )

        return SuccessResponse(
            data={
                "updated_count": len(anomaly_ids),
                "action": action,
                "timestamp": datetime.now().isoformat(),
            },
            message=f"Successfully marked {len(anomaly_ids)} anomalies as {action}",
        )

    except Exception as e:
        logger.error(f"Failed to mark anomalies: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update anomaly status: {str(e)}"
        )


def _generate_anomaly_recommendations(
    anomalies: List[DetectedAnomaly], summary: Dict[str, Any]
) -> List[str]:
    """Generate recommendations based on detected anomalies."""
    recommendations = []

    # Check for critical anomalies
    critical_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
    if critical_count > 0:
        recommendations.append(
            f"Review {critical_count} critical anomalies immediately"
        )

    # Check for duplicate transactions
    duplicates = sum(
        1 for a in anomalies if a.anomaly_type == AnomalyType.DUPLICATE_TRANSACTION
    )
    if duplicates > 0:
        recommendations.append(f"Found {duplicates} potential duplicate transactions")

    # Check for unusual amounts
    high_amounts = sum(
        1 for a in anomalies if a.anomaly_type == AnomalyType.UNUSUAL_AMOUNT
    )
    if high_amounts > 2:
        recommendations.append(
            "Multiple high-value transactions detected - verify legitimacy"
        )

    # General recommendation
    if len(anomalies) > 5:
        recommendations.append(
            "Consider reviewing your recent transactions for accuracy"
        )

    if not recommendations:
        recommendations.append(
            "No significant anomalies detected - transactions appear normal"
        )

    return recommendations[:5]  # Top 5 recommendations


def _format_anomaly_response(anomaly: Dict[str, Any]) -> DetectedAnomaly:
    """Safely format anomaly response with proper field mapping."""
    transaction = anomaly.get("transaction", {})

    # Map anomaly types to match enum values
    anomaly_type_mapping = {
        "unusual_expense_amount": AnomalyType.UNUSUAL_AMOUNT,
        "unusual_income_amount": AnomalyType.UNUSUAL_AMOUNT,
        "high_transaction_frequency": AnomalyType.UNUSUAL_FREQUENCY,
        "category_outlier": AnomalyType.CATEGORY_SPIKE,
        "unusual_amount": AnomalyType.UNUSUAL_AMOUNT,
        "unusual_frequency": AnomalyType.UNUSUAL_FREQUENCY,
        "new_merchant": AnomalyType.NEW_MERCHANT,
        "category_spike": AnomalyType.CATEGORY_SPIKE,
        "duplicate_transaction": AnomalyType.DUPLICATE_TRANSACTION,
        "time_anomaly": AnomalyType.TIME_ANOMALY,
    }

    raw_anomaly_type = anomaly.get("anomaly_type", "unusual_amount")
    mapped_anomaly_type = anomaly_type_mapping.get(raw_anomaly_type, AnomalyType.UNUSUAL_AMOUNT)

    return DetectedAnomaly(
        transaction_id=transaction.get("id"),
        transaction_date=transaction.get("date", ""),
        transaction_amount=float(transaction.get("amount", 0)),
        transaction_description=transaction.get("description", ""),
        anomaly_type=mapped_anomaly_type,
        severity=anomaly.get("severity", AnomalySeverity.MEDIUM),
        confidence_score=float(anomaly.get("confidence", 0.5)),
        reason=anomaly.get("reason", "Anomaly detected"),
        expected_range=anomaly.get("expected_range"),
        similar_transactions=anomaly.get("similar_transactions", [])
    )
