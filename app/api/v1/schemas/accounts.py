"""
Account management and Plaid connection schemas.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class AccountType(str, Enum):
    """Bank account types."""

    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit"
    INVESTMENT = "investment"
    OTHER = "other"


class AccountStatus(str, Enum):
    """Account connection status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class InstitutionPreference(str, Enum):
    """Bank institution preferences."""

    MAJOR_BANK = "major_bank"
    CREDIT_UNION = "credit_union"
    ONLINE_BANK = "online_bank"
    ANY = "any"


class BankAccount(BaseModel):
    """Bank account information."""

    id: str
    name: str
    official_name: Optional[str] = None
    account_type: AccountType
    balance: float
    available_balance: Optional[float] = None
    currency: str = "USD"
    institution: str
    last_four: Optional[str] = Field(None, pattern="^[0-9]{4}$")
    status: AccountStatus
    last_updated: datetime
    metadata: Optional[Dict[str, Any]] = None


class AccountStatusRequest(BaseModel):
    """Request for account status check."""

    user_id: str = Field(..., min_length=1)
    user_context: Dict[str, Any] = Field(
        default_factory=dict, description="User context including existing accounts"
    )


class AccountStatusResponse(BaseModel):
    """Account status response."""

    has_accounts: bool
    account_count: int
    accounts: List[BankAccount]
    total_balance: float
    status: Dict[str, Any]
    last_updated: datetime = Field(default_factory=datetime.now)


class PlaidConnectionRequest(BaseModel):
    """Request to initiate Plaid connection."""

    user_id: str = Field(..., min_length=1)
    institution_preference: Optional[InstitutionPreference] = None
    account_types: Optional[List[AccountType]] = None

    @field_validator("account_types")
    def validate_account_types(cls, v):
        if v and len(v) > 5:
            raise ValueError("Cannot select more than 5 account types")
        return v


class PlaidConnectionResponse(BaseModel):
    """Plaid connection initiation response."""

    action: str = "initiate_plaid_connection"
    user_id: str
    link_token: Optional[str] = None
    expiration: Optional[datetime] = None
    institution_preference: Optional[str] = None
    message: str
    next_step: str


class DisconnectAccountRequest(BaseModel):
    """Request to disconnect an account."""

    user_id: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)
    reason: Optional[str] = Field(None, max_length=500)


class DisconnectAccountResponse(BaseModel):
    """Account disconnection response."""

    action: str = "disconnect_account"
    account_id: str
    user_id: str
    success: bool
    message: str
    requires_confirmation: bool = True
    disconnected_at: Optional[datetime] = None


class AccountDetailsRequest(BaseModel):
    """Request for specific account details."""

    user_id: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)
    include_transactions: bool = False
    transaction_days: int = Field(default=30, ge=1, le=90)


class AccountDetailsResponse(BaseModel):
    """Detailed account information."""

    account: BankAccount
    recent_transactions: Optional[List[Dict[str, Any]]] = None
    transaction_summary: Optional[Dict[str, float]] = None
    insights: Optional[List[Dict[str, Any]]] = None
