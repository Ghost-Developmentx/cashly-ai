"""
Budget generation and management endpoints.
Replaces Flask BudgetController.
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from logging import getLogger

from app.core.dependencies import get_budget_service
from app.core.exceptions import ValidationError
from app.api.v1.schemas.budget import (
    BudgetRequest,
    BudgetResponse,
    BudgetSummary,
    CategoryBudget,
    SavingsPotential,
)
from app.api.v1.schemas.base import SuccessResponse, ErrorResponse
from app.services.budget import AsyncBudgetService

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate",
    response_model=BudgetResponse,
    summary="Generate personalized budget",
    description="Generate budget recommendations based on spending patterns",
)
async def generate_budget(
    request: BudgetRequest, service: AsyncBudgetService = Depends(get_budget_service)
) -> BudgetResponse:
    """Generate personalized budget recommendations."""
    try:
        # Convert Pydantic models to dicts for service
        transactions = [t.model_dump() for t in request.transactions]

        logger.info(
            f"Generating budget for user {request.user_id} "
            f"with {len(transactions)} transactions"
        )

        result = await service.generate_budget(
            user_id=request.user_id,
            transactions=transactions,
            monthly_income=request.monthly_income,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Convert to a response model with additional formatting
        category_budgets = [
            CategoryBudget(
                category=cat,
                allocated_amount=amount,
                current_spending=result["current_spending"].get(cat, 0),
                difference=amount - result["current_spending"].get(cat, 0),
                percentage_of_income=(amount / result["monthly_income"] * 100),
            )
            for cat, amount in result["budget_allocations"].items()
        ]

        response = BudgetResponse(
            monthly_income=result["monthly_income"],
            budget_allocations=result["budget_allocations"],
            category_budgets=category_budgets,
            current_spending=result["current_spending"],
            recommendations=result["recommendations"],
            savings_potential=SavingsPotential(**result["savings_potential"]),
            analysis_period=result["analysis_period"],
        )

        logger.info(
            f"Generated budget with {len(response.recommendations)} "
            f"recommendations for user {request.user_id}"
        )

        return response

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Budget generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate budget: {str(e)}"
        )


@router.post(
    "/summary",
    response_model=BudgetSummary,
    summary="Get budget summary",
    description="Get a simplified budget summary",
)
async def get_budget_summary(
    request: BudgetRequest, service: AsyncBudgetService = Depends(get_budget_service)
) -> BudgetSummary:
    """Get simplified budget summary."""
    try:
        transactions = [t.model_dump() for t in request.transactions]

        # Generate a full budget first
        result = await service.generate_budget(
            user_id=request.user_id,
            transactions=transactions,
            monthly_income=request.monthly_income,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Calculate summary metrics
        total_allocated = sum(result["budget_allocations"].values())
        total_spending = sum(result["current_spending"].values())
        savings_rate = (
            (result["monthly_income"] - total_spending) / result["monthly_income"] * 100
            if result["monthly_income"] > 0
            else 0
        )

        # Find problem categories
        problem_categories = [
            rec["category"]
            for rec in result["recommendations"]
            if rec["type"] == "reduce_spending" and rec.get("category")
        ]

        return BudgetSummary(
            total_income=result["monthly_income"],
            total_allocated=total_allocated,
            total_spending=total_spending,
            savings_rate=round(savings_rate, 1),
            is_over_budget=total_spending > total_allocated,
            problem_categories=problem_categories,
        )

    except Exception as e:
        logger.error(f"Budget summary failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate budget summary: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Validate budget data",
    description="Validate transactions and income data for budget generation",
)
async def validate_budget_data(
    request: BudgetRequest,
) -> SuccessResponse[Dict[str, Any]]:
    """Validate budget input data."""
    validation_results = {"is_valid": True, "warnings": [], "errors": []}

    # Check transaction dates
    dates = [datetime.strptime(t.date, "%Y-%m-%d") for t in request.transactions]
    date_range = (max(dates) - min(dates)).days

    if date_range < 30:
        validation_results["warnings"].append(
            "Less than 30 days of data may result in inaccurate budgets"
        )

    # Check for income transactions
    has_income = any(t.amount > 0 for t in request.transactions)
    if not has_income and not request.monthly_income:
        validation_results["errors"].append(
            "No income found and monthly_income not provided"
        )
        validation_results["is_valid"] = False

    # Check categories
    uncategorized = sum(
        1 for t in request.transactions if t.category == "Uncategorized"
    )
    if uncategorized > len(request.transactions) * 0.5:
        validation_results["warnings"].append(
            f"{uncategorized} uncategorized transactions may affect accuracy"
        )

    return SuccessResponse(data=validation_results, message="Validation complete")
