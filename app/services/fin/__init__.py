"""
Financial services package.
Only the Rails client remains after tool migration.
"""

from .async_rails_client import AsyncRailsClient

__all__ = ["AsyncRailsClient"]