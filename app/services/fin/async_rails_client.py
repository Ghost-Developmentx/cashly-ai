"""
Async HTTP client for Rails API communication.
"""

import os
import logging
from typing import Dict, Any, Optional
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


class AsyncRailsClient:
    """
    Client for asynchronous communication with a Rails API.

    This class provides methods to perform asynchronous HTTP interactions with a Rails
    API, including POST and DELETE requests. It manages the session lifecycle with
    aiohttp and enables secure communications using API keys and session locks. The
    class is particularly useful for scenarios where multiple asynchronous network
    requests are needed efficiently.

    Attributes
    ----------
    base_url : Optional[str]
        The base URL for the Rails API retrieved from environment variables.
    api_key : Optional[str]
        The API key for authentication. Defaults to a secure key if not provided in the
        environment variables.
    timeout : aiohttp.ClientTimeout
        The default timeout configuration for the aiohttp client.
    session : Optional[aiohttp.ClientSession]
        The aiohttp client session used for making requests.
    _session_lock : asyncio.Lock
        Lock to ensure thread-safe creation of the aiohttp session.
    """

    def __init__(self):
        self.base_url = os.getenv("RAILS_API_URL")
        self.api_key = os.getenv("INTERNAL_API_KEY", "your-secure-internal-api-key")
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            async with self._session_lock:
                if self.session is None:
                    self.session = aiohttp.ClientSession(
                        timeout=self.timeout,
                        headers={
                            "X-Internal-Api-Key": self.api_key,
                            "Content-Type": "application/json",
                        },
                    )
        return self.session

    async def post(
        self, endpoint: str, data: Dict[str, Any], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make async POST request to Rails API.

        Args:
            endpoint: API endpoint
            data: Request data
            timeout: Optional custom timeout

        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        try:
            custom_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None

            async with session.post(url, json=data, timeout=custom_timeout) as response:
                response_data = await response.json()

                if response.status == 200:
                    return response_data
                else:
                    logger.error(
                        f"Rails API error: {response.status} - {response_data}"
                    )
                    return {
                        "error": response_data.get("error", "Request failed"),
                        "status_code": response.status,
                    }

        except asyncio.TimeoutError:
            logger.error(f"Rails API timeout for {endpoint}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Rails API request failed: {e}")
            return {"error": str(e)}

    async def get(
            self, endpoint: str, params: Optional[Dict[str, Any]] = None,
            timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make async GET request to Rails API."""
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        try:
            custom_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None
            async with session.get(url, params=params, timeout=custom_timeout) as response:
                response_data = await response.json()

                if response.status == 200:
                    return response_data
                else:
                    logger.error(
                        f"Rails API error: {response.status} - {response_data}"
                    )
                    return {
                        "error": response_data.get("error", "Request failed"),
                        "status_code": response.status,
                    }

        except asyncio.TimeoutError:
            logger.error(f"Rails API timeout for {endpoint}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Rails API GET failed: {e}")
            return {"error": str(e)}

    async def delete(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make async DELETE request to Rails API.

        Args:
            endpoint: API endpoint
            data: Optional request data

        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        try:
            async with session.delete(url, json=data) as response:
                response_data = await response.json()

                if response.status == 200:
                    return response_data
                else:
                    return {
                        "error": response_data.get("error", "Delete failed"),
                        "status_code": response.status,
                    }

        except Exception as e:
            logger.error(f"Rails API delete failed: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
