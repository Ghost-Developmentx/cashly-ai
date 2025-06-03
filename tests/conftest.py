"""
Test configuration and fixtures.
"""

import pytest
from typing import AsyncGenerator

import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
import sys
from pathlib import Path


from app.main import app
from app.core.dependencies import get_db

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))



@pytest_asyncio.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async for session in get_db():
        yield session
