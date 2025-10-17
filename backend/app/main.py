"""Entry point for the backend web application."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(title="DL Result Analyzer API")

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Simple health endpoint to confirm the service is running."""
        return {"status": "ok"}

    return app


app = create_app()
