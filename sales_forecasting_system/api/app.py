"""
api/app.py
==========
FastAPI application factory.
Creates the app instance, registers middleware, exception handlers, and routers.
Exposes Swagger UI at /docs and ReDoc at /redoc.
"""

import time
from contextlib import asynccontextmanager
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes.forecast import router as forecast_router
from api.schemas.response import ErrorResponse, HealthResponse
from src.constants import METRICS_DIR, CONFIG_PATH
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan context manager (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("🚀 Sales Forecasting API starting up…")
    # Verify critical artifacts
    if not (METRICS_DIR / "best_model_selection.json").exists():
        logger.warning(
            "⚠️  Model selection file not found. "
            "Call POST /train before calling POST /predict."
        )
    yield
    logger.info("🛑 Sales Forecasting API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = load_yaml(CONFIG_PATH)
    api_cfg = cfg.get("api", {})

    app = FastAPI(
        title=api_cfg.get("title", "Sales Forecasting API"),
        description=api_cfg.get(
            "description",
            "Production-ready time-series forecasting service using "
            "SARIMA, Prophet, XGBoost, and LSTM.",
        ),
        version=api_cfg.get("version", "1.0.0"),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "ML Engineering Team",
            "email": "ml-team@company.com",
        },
        license_info={
            "name": "MIT",
        },
    )

    # ── Middleware ────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request timing middleware ─────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

    # ── Global exception handlers ─────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message="An unexpected error occurred.",
                detail=str(exc),
            ).model_dump(),
        )

    # ── Routes ────────────────────────────────────────────────────────
    app.include_router(forecast_router)

    # ── Health check (root) ───────────────────────────────────────────
    @app.get(
        "/",
        response_model=HealthResponse,
        summary="Health check",
        tags=["Health"],
    )
    async def health_check():
        """
        **GET /**

        Returns service health and model readiness status.
        """
        models_ready = (METRICS_DIR / "best_model_selection.json").exists()
        return HealthResponse(
            models_ready=models_ready,
            message=(
                "Models ready. Use POST /predict to generate forecasts."
                if models_ready
                else "Models NOT ready. Please run POST /train first."
            ),
        )

    logger.info(
        f"FastAPI app created — Swagger UI: http://localhost:"
        f"{api_cfg.get('port', 8000)}/docs"
    )
    return app


# Instantiate application (for uvicorn / gunicorn import)
app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_yaml(CONFIG_PATH)
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
        log_level=api_cfg.get("log_level", "info"),
    )
