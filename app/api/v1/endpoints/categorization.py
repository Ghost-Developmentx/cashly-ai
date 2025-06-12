"""
Transaction categorization endpoints.
Replaces Flask CategorizationController.
"""

import time
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException
from logging import getLogger

from app.core.dependencies import get_categorization_service
from app.schemas.categorization import (
    TransactionInput,
    BatchTransactionInput,
    CategoryResponse,
    BatchCategoryResponse,
    CategoryFeedback,
    CategoryStatistics,
)
from app.schemas.base import SuccessResponse
from app.services.categorize import AsyncCategorizationService

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    "/transaction",
    response_model=CategoryResponse,
    summary="Categorize single transaction",
    description="Categorize a transaction based on description and amount",
)
async def categorize_transaction(
    transaction: TransactionInput,
    service: AsyncCategorizationService = Depends(get_categorization_service),
) -> CategoryResponse:
    """Categorize a single transaction."""
    try:
        result = await service.categorize_transaction(
            description=transaction.description,
            amount=transaction.amount,
            merchant=transaction.merchant,
        )

        logger.info(
            f"Categorized transaction: {result['category']} "
            f"(confidence: {result['confidence']})"
        )

        return CategoryResponse(**result)

    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchCategoryResponse,
    summary="Categorize multiple transactions",
    description="Categorize a batch of transactions",
)
async def categorize_batch(
    batch: BatchTransactionInput,
    service: AsyncCategorizationService = Depends(get_categorization_service),
) -> BatchCategoryResponse:
    """Categorize multiple transactions."""
    start_time = time.time()

    try:
        # Convert to dict format for service
        transactions = [t.model_dump() for t in batch.transactions]

        logger.info(f"Categorizing {len(transactions)} transactions")

        results = await service.categorize_batch(transactions)

        processing_time = time.time() - start_time

        logger.info(
            f"Categorized {len(results)} transactions " f"in {processing_time:.2f}s"
        )

        return BatchCategoryResponse(
            categorized_count=len(results),
            results=results,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Batch categorization failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch categorization failed: {str(e)}"
        )


@router.post(
    "/feedback",
    response_model=SuccessResponse[dict],
    summary="Submit categorization feedback",
    description="Help improve categorization with correct labels",
)
async def submit_feedback(
    feedback: CategoryFeedback,
    service: AsyncCategorizationService = Depends(get_categorization_service),
) -> SuccessResponse[dict]:
    """Submit categorization feedback for learning."""
    try:
        success = await service.learn_from_feedback(
            description=feedback.description,
            amount=feedback.amount,
            correct_category=feedback.correct_category,
        )

        if success:
            logger.info(
                f"Learned categorization: {feedback.description} -> "
                f"{feedback.correct_category}"
            )
            return SuccessResponse(
                data={"learned": True}, message="Successfully learned from feedback"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process feedback")

    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process feedback: {str(e)}"
        )


@router.post(
    "/statistics",
    response_model=CategoryStatistics,
    summary="Get categorization statistics",
    description="Analyze categorization coverage and distribution",
)
async def get_statistics(
    transactions: List[Dict],
    service: AsyncCategorizationService = Depends(get_categorization_service),
) -> CategoryStatistics:
    """Get categorization statistics for transactions."""
    try:
        stats = await service.get_category_statistics(transactions)
        return CategoryStatistics(**stats)

    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate statistics: {str(e)}"
        )
