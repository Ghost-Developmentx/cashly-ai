"""
Account management and Plaid connection endpoints.
Replaces Flask AccountController.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from logging import getLogger
from datetime import datetime, timedelta

from app.core.dependencies import get_db
from app.core.exceptions import ValidationError, ResourceNotFoundError
from app.api.v1.schemas.accounts import (
    AccountStatusRequest,
    AccountStatusResponse,
    PlaidConnectionRequest,
    PlaidConnectionResponse,
    DisconnectAccountRequest,
    DisconnectAccountResponse,
    AccountDetailsRequest,
    AccountDetailsResponse,
    BankAccount,
    AccountStatus,
)
from app.api.v1.schemas.base import SuccessResponse, ErrorResponse
from app.services.fin.async_tool_registry import AsyncToolRegistry

logger = getLogger(__name__)
router = APIRouter()

# Initialize tool registry for account operations
tool_registry = AsyncToolRegistry()


@router.post(
    "/status",
    response_model=AccountStatusResponse,
    summary="Get user account status",
    description="Get connected bank accounts and status",
)
async def get_account_status(request: AccountStatusRequest) -> AccountStatusResponse:
    """Get user account status for Fin queries."""
    try:
        logger.info(f"Getting account status for user {request.user_id}")

        # Execute get_user_accounts tool
        result = await tool_registry.execute(
            tool_name="get_user_accounts",
            tool_args={},
            user_id=request.user_id,
            transactions=[],
            user_context=request.user_context,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Format accounts
        accounts = [
            BankAccount(
                id=acc.get("id", ""),
                name=acc.get("name", "Unknown"),
                account_type=acc.get("account_type", "other").lower(),
                balance=acc.get("balance", 0.0),
                available_balance=acc.get("available_balance"),
                institution=acc.get("institution", "Unknown Bank"),
                last_four=acc.get("last_four"),
                status=AccountStatus.ACTIVE,
                last_updated=datetime.now(),
            )
            for acc in result.get("accounts", [])
        ]

        response = AccountStatusResponse(
            has_accounts=result.get("has_accounts", False),
            account_count=result.get("account_count", 0),
            accounts=accounts,
            total_balance=result.get("total_balance", 0.0),
            status={
                "has_accounts": result.get("has_accounts", False),
                "account_count": result.get("account_count", 0),
                "last_updated": datetime.now().isoformat(),
            },
        )

        logger.info(
            f"Retrieved {response.account_count} accounts for user {request.user_id}"
        )

        return response

    except Exception as e:
        logger.error(f"Account status check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get account status: {str(e)}"
        )


@router.post(
    "/connect",
    response_model=PlaidConnectionResponse,
    summary="Initiate Plaid connection",
    description="Start the process to connect a bank account via Plaid",
)
async def initiate_plaid_connection(
    request: PlaidConnectionRequest,
) -> PlaidConnectionResponse:
    """Initiate a Plaid connection process."""
    try:
        logger.info(
            f"Initiating Plaid connection for user {request.user_id} "
            f"with preference: {request.institution_preference}"
        )

        # Execute initiate_plaid_connection tool
        result = await tool_registry.execute(
            tool_name="initiate_plaid_connection",
            tool_args={
                "institution_preference": (
                    request.institution_preference.value
                    if request.institution_preference
                    else None
                )
            },
            user_id=request.user_id,
            transactions=[],
            user_context={},
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        response = PlaidConnectionResponse(
            user_id=request.user_id,
            link_token=result.get("link_token"),
            expiration=datetime.now() + timedelta(minutes=30),
            institution_preference=result.get("institution_preference"),
            message=result.get("message", "Ready to connect your bank account"),
            next_step=result.get("next_step", "plaid_link_token"),
        )

        logger.info(f"Plaid connection initiated for user {request.user_id}")

        return response

    except Exception as e:
        logger.error(f"Plaid connection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate Plaid connection: {str(e)}"
        )


@router.post(
    "/disconnect",
    response_model=DisconnectAccountResponse,
    summary="Disconnect bank account",
    description="Disconnect a connected bank account",
)
async def disconnect_account(
    request: DisconnectAccountRequest,
) -> DisconnectAccountResponse:
    """Disconnect a bank account."""
    try:
        logger.info(
            f"Disconnecting account {request.account_id} for user {request.user_id}"
        )

        # Execute disconnect_account tool
        result = await tool_registry.execute(
            tool_name="disconnect_account",
            tool_args={"account_id": request.account_id},
            user_id=request.user_id,
            transactions=[],
            user_context={},
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        response = DisconnectAccountResponse(
            account_id=request.account_id,
            user_id=request.user_id,
            success=True,
            message=result.get("message", "Account disconnected successfully"),
            requires_confirmation=result.get("requires_confirmation", True),
            disconnected_at=(
                datetime.now() if not result.get("requires_confirmation") else None
            ),
        )

        logger.info(
            f"Account {request.account_id} disconnect initiated for user {request.user_id}"
        )

        return response

    except Exception as e:
        logger.error(f"Account disconnection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to disconnect account: {str(e)}"
        )


@router.get(
    "/{account_id}",
    response_model=AccountDetailsResponse,
    summary="Get account details",
    description="Get detailed information about a specific account",
)
async def get_account_details(
    account_id: str,
    user_id: str = Query(..., min_length=1),
    include_transactions: bool = Query(default=False),
    transaction_days: int = Query(default=30, ge=1, le=90),
) -> AccountDetailsResponse:
    """Get detailed account information."""
    try:
        # Execute get_account_details tool
        result = await tool_registry.execute(
            tool_name="get_account_details",
            tool_args={"account_id": account_id},
            user_id=user_id,
            transactions=[],
            user_context={},
        )

        if "error" in result:
            raise ResourceNotFoundError("Account", account_id)

        account_data = result.get("account", {})

        # Format account
        account = BankAccount(
            id=account_data.get("id", account_id),
            name=account_data.get("name", "Unknown"),
            official_name=account_data.get("official_name"),
            account_type=account_data.get("account_type", "other").lower(),
            balance=result.get("balance", 0.0),
            available_balance=account_data.get("available_balance"),
            institution=result.get("institution", "Unknown Bank"),
            last_four=account_data.get("last_four"),
            status=AccountStatus.ACTIVE,
            last_updated=datetime.now(),
        )

        # Add transaction summary if requested
        transaction_summary = None
        if include_transactions:
            # This would fetch actual transactions
            transaction_summary = {
                "total_income": 5000.0,
                "total_expenses": 3500.0,
                "transaction_count": 45,
                "period_days": transaction_days,
            }

        response = AccountDetailsResponse(
            account=account,
            transaction_summary=transaction_summary,
            insights=[
                {
                    "type": "spending_pattern",
                    "message": "Your spending has decreased by 10% this month",
                    "positive": True,
                }
            ],
        )

        return response

    except ResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get account details: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get account details: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[BankAccount],
    summary="List all accounts",
    description="Get all connected bank accounts for a user",
)
async def list_accounts(
    user_id: str = Query(..., min_length=1),
    status: Optional[AccountStatus] = Query(None),
) -> List[BankAccount]:
    """List all user accounts."""
    try:
        # Get user context with accounts
        user_context = {"accounts": []}  # Would fetch from database

        result = await tool_registry.execute(
            tool_name="get_user_accounts",
            tool_args={},
            user_id=user_id,
            transactions=[],
            user_context=user_context,
        )

        accounts = [
            BankAccount(
                id=acc.get("id", ""),
                name=acc.get("name", "Unknown"),
                account_type=acc.get("account_type", "other").lower(),
                balance=acc.get("balance", 0.0),
                institution=acc.get("institution", "Unknown Bank"),
                status=AccountStatus.ACTIVE,
                last_updated=datetime.now(),
            )
            for acc in result.get("accounts", [])
        ]

        # Filter by status if provided
        if status:
            accounts = [acc for acc in accounts if acc.status == status]

        return accounts

    except Exception as e:
        logger.error(f"Failed to list accounts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list accounts: {str(e)}"
        )
