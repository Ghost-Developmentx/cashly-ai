"""
Connection health monitoring utilities.
Helps debug and monitor connection states.
"""

import logging
import psutil
import os
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionHealthMonitor:
    """Monitors and reports on connection health."""

    @staticmethod
    def get_connection_stats() -> Dict[str, Any]:
        """Get current connection statistics."""
        try:
            process = psutil.Process(os.getpid())

            # Get file descriptors
            fds = process.num_fds()

            # Get network connections
            connections = process.net_connections(kind="inet")

            # Categorize connections
            established = [c for c in connections if c.status == "ESTABLISHED"]
            listen = [c for c in connections if c.status == "LISTEN"]
            other = [
                c for c in connections if c.status not in ["ESTABLISHED", "LISTEN"]
            ]

            # Get specific connection types
            http_connections = []
            db_connections = []

            for conn in established:
                if conn.raddr:
                    if conn.raddr.port == 443:  # HTTPS
                        http_connections.append(conn)
                    elif conn.raddr.port in [5432, 5433]:  # PostgreSQL
                        db_connections.append(conn)

            return {
                "timestamp": datetime.now().isoformat(),
                "file_descriptors": fds,
                "total_connections": len(connections),
                "established": len(established),
                "listening": len(listen),
                "other": len(other),
                "https_connections": len(http_connections),
                "db_connections": len(db_connections),
                "details": {
                    "by_status": {
                        status: len([c for c in connections if c.status == status])
                        for status in set(c.status for c in connections)
                    }
                },
            }
        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            return {"error": str(e)}

    @staticmethod
    def log_connection_state(prefix: str = ""):
        """Log current connection state."""
        stats = ConnectionHealthMonitor.get_connection_stats()

        if "error" in stats:
            logger.error(f"{prefix} Connection stats error: {stats['error']}")
            return

        logger.info(
            f"{prefix} Connections - "
            f"FDs: {stats['file_descriptors']}, "
            f"Total: {stats['total_connections']}, "
            f"HTTPS: {stats['https_connections']}, "
            f"DB: {stats['db_connections']}"
        )

    @staticmethod
    def check_connection_leaks(
        threshold_fds: int = 100, threshold_connections: int = 50
    ) -> List[str]:
        """Check for potential connection leaks."""
        warnings = []
        stats = ConnectionHealthMonitor.get_connection_stats()

        if "error" not in stats:
            if stats["file_descriptors"] > threshold_fds:
                warnings.append(
                    f"High file descriptor count: {stats['file_descriptors']}"
                )

            if stats["total_connections"] > threshold_connections:
                warnings.append(f"High connection count: {stats['total_connections']}")

            if stats["https_connections"] > 20:
                warnings.append(f"Many HTTPS connections: {stats['https_connections']}")

        return warnings


# Convenience function for middleware
def log_connection_health(when: str):
    """Log connection health at specific points."""
    monitor = ConnectionHealthMonitor()
    monitor.log_connection_state(f"[{when}]")
