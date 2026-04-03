from __future__ import annotations

import asyncio

from .server import EngineeringAutomationServer


def main() -> int:
    server = EngineeringAutomationServer()
    asyncio.run(server.serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
