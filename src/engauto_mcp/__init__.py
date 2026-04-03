"""Engineering Automation MCP Server package."""

from .config import DEFAULT_DB_PATH, PACKAGE_NAME

__all__ = [
    "DEFAULT_DB_PATH",
    "EngineeringAutomationServer",
    "MCPClient",
    "PACKAGE_NAME",
]


def __getattr__(name: str):
    if name == "MCPClient":
        from .client import MCPClient

        return MCPClient
    if name == "EngineeringAutomationServer":
        from .server import EngineeringAutomationServer

        return EngineeringAutomationServer
    raise AttributeError(name)
