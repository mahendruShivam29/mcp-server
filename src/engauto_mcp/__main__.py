from __future__ import annotations

import uvicorn


def main() -> int:
    uvicorn.run("engauto_mcp.server:app", host="0.0.0.0", port=8000, lifespan="on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
