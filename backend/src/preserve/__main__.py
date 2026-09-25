"""Run the Preserve API server."""

import uvicorn

from preserve.config import settings


def main():
    uvicorn.run(
        "preserve.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
