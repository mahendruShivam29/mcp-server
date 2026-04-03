"""Engineering Automation MCP Server package."""

from .client import MCPClient
from .config import DEFAULT_DB_PATH, PACKAGE_NAME
from .server import EngineeringAutomationServer

__all__ = [
    "DEFAULT_DB_PATH",
    "EngineeringAutomationServer",
    "MCPClient",
    "PACKAGE_NAME",
]
