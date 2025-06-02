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
    Replaces Flask's before_first_request and teardown handlers.
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
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
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
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.api_v1_prefix)


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.project_name,
        "version": settings.version,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
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
