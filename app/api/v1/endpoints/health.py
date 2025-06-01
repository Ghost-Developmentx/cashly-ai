"""
Health check endpoints.
Replaces Flask health check routes.
"""

import time
import psutil
import os
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_openai_service
from app.core.config import settings
from app.api.v1.schemas.health import (
    HealthResponse,
    DetailedHealthResponse,
    ComponentHealth,
)

router = APIRouter()

# Track startup time
STARTUP_TIME = time.time()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic service health status",
)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy", service=settings.project_name, version=settings.version
    )


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="Returns detailed health status including all components",
)
async def detailed_health_check(
    db: AsyncSession = Depends(get_db), openai_service=Depends(get_openai_service)
) -> DetailedHealthResponse:
    """Detailed health check with component status."""
    components = {}
    overall_status = "healthy"

    # Check a database
    try:
        result = await db.execute("SELECT 1")
        components["database"] = ComponentHealth(
            status="healthy", message="Database connection successful"
        )
    except Exception as e:
        components["database"] = ComponentHealth(
            status="unhealthy", message=f"Database error: {str(e)}"
        )
        overall_status = "unhealthy"

    # Check OpenAI service
    try:
        ai_health = await openai_service.health_check()
        components["openai"] = ComponentHealth(
            status=ai_health["status"],
            message="OpenAI assistants available",
            details=ai_health.get("assistants", {}),
        )
        if ai_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        components["openai"] = ComponentHealth(
            status="unhealthy", message=f"OpenAI service error: {str(e)}"
        )
        overall_status = "degraded"

    # Check system resources
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        components["system"] = ComponentHealth(
            status="healthy",
            message="System resources normal",
            details={
                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "open_files": process.num_fds(),
            },
        )
    except Exception as e:
        components["system"] = ComponentHealth(
            status="degraded", message=f"System check error: {str(e)}"
        )

    # Calculate uptime
    uptime = time.time() - STARTUP_TIME

    return DetailedHealthResponse(
        status=overall_status,
        service=settings.project_name,
        version=settings.version,
        components=components,
        uptime=round(uptime, 2),
    )
