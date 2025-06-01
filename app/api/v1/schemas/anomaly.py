"""
Anomaly detection schemas for unusual transaction patterns.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class AnomalyType(str, Enum):
    """Types of transaction anomalies."""

    UNUSUAL_AMOUNT = "unusual_amount"
    UNUSUAL_FREQUENCY = "unusual_frequency"
    NEW_MERCHANT = "new_merchant"
    CATEGORY_SPIKE = "category_spike"
    DUPLICATE_TRANSACTION = "duplicate_transaction"
    TIME_ANOMALY = "time_anomaly"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionForAnomaly(BaseModel):
    """Transaction data for anomaly detection."""

    id: Optional[str] = None
    date: str = Field(..., description="YYYY-MM-DD format")
    amount: float
    description: str
    category: str = "Uncategorized"
    merchant: Optional[str] = None

    @field_validator("date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""

    user_id: str = Field(..., min_length=1)
    transactions: List[TransactionForAnomaly] = Field(
        ..., min_length=1, description="Transactions to analyze"
    )
    threshold: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Anomaly detection threshold (standard deviations)",
    )
    check_duplicates: bool = Field(
        default=True, description="Check for duplicate transactions"
    )


class DetectedAnomaly(BaseModel):
    """Individual detected anomaly."""

    transaction_id: Optional[str] = None
    transaction_date: str
    transaction_amount: float
    transaction_description: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence_score: float = Field(..., ge=0, le=1)
    reason: str
    expected_range: Optional[Dict[str, float]] = None
    similar_transactions: Optional[List[Dict[str, Any]]] = None


class AnomalySummary(BaseModel):
    """Summary of anomaly detection results."""

    total_transactions: int
    anomalies_detected: int
    anomaly_rate: float = Field(..., ge=0, le=100)
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    highest_risk_categories: List[str]


class AnomalyDetectionResponse(BaseModel):
    """Complete anomaly detection response."""

    user_id: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    anomalies: List[DetectedAnomaly]
    summary: AnomalySummary
    threshold_used: float
    recommendations: List[str]


class AnomalyPattern(BaseModel):
    """Recurring anomaly pattern."""

    pattern_type: str
    frequency: str
    transactions_affected: int
    total_amount: float
    description: str
    first_occurrence: str
    last_occurrence: str


class AnomalyTrendResponse(BaseModel):
    """Anomaly trends over time."""

    time_period: str
    total_anomalies: int
    anomaly_trend: str = Field(..., pattern="^(increasing|decreasing|stable)$")
    patterns: List[AnomalyPattern]
    risk_assessment: Dict[str, Any]
