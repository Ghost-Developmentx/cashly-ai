"""
FastAPI application entry point.
Replaces Flask's app.py
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.middleware import TimingMiddleware, ErrorHandlingMiddleware
from app.core.exceptions import CashlyException
from app.api.v1.api import api_router
from app.db.init import AsyncDatabaseInitializer
from app.db.singleton_registry import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application lifecycle events.
    """
    logger.info("🚀 Starting Cashly AI Service...")

    # Initialize database
    try:
        db_init = AsyncDatabaseInitializer()
        await db_init.initialize()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    # Initialize MLflow (non-blocking)
    try:
        from app.core.mlflow_config import mlflow_manager
        mlflow_manager.initialize()
        logger.info("✅ MLflow initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ MLflow initialization failed (non-critical): {e}")

    # Startup complete
    logger.info("✅ Cashly AI Service started successfully")

    yield

    # Cleanup on shutdown
    logger.info("🛑 Shutting down Cashly AI Service...")

    try:
        await registry.cleanup_all()
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"📥 Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response


# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }


# Global exception handler
@app.exception_handler(CashlyException)
async def cashly_exception_handler(
    request: Request, exc: CashlyException
) -> JSONResponse:
    """Handle custom Cashly exceptions."""
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
    )
