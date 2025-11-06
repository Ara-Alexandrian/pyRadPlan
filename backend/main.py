"""Main FastAPI application for pyRadPlan dashboard.

This is the entry point for the backend API server.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from .core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    config = get_config()
    logger.info(f"Starting pyRadPlan Dashboard API")
    logger.info(f"Environment: {config.environment.name}")
    logger.info(f"Backend: {config.network.base_url}:{config.network.backend_port}")

    # Create necessary directories
    config.paths.patient_data.mkdir(parents=True, exist_ok=True)
    config.paths.results.mkdir(parents=True, exist_ok=True)
    config.paths.temp.mkdir(parents=True, exist_ok=True)
    logger.info("Data directories ensured")

    # Initialize database (future)
    # await init_database()

    # Initialize job queue (future)
    # await init_job_queue()

    yield

    # Shutdown
    logger.info("Shutting down pyRadPlan Dashboard API")
    # Cleanup code here


# Create FastAPI app
app = FastAPI(
    title="pyRadPlan Dashboard API",
    description="REST API for pyRadPlan radiation therapy treatment planning",
    version="0.1.0",
    lifespan=lifespan
)

# Get configuration
config = get_config()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.network.base_url,
        f"http://localhost:{config.network.frontend_port}",
        f"http://127.0.0.1:{config.network.frontend_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": config.environment.name,
        "version": "0.1.0"
    }


# Configuration info endpoint
@app.get("/api/config/info", tags=["Configuration"])
async def get_config_info():
    """Get basic configuration information (non-sensitive)."""
    return {
        "environment": {
            "name": config.environment.name,
            "location": config.environment.location
        },
        "features": {
            "dicom_import": config.features.dicom_import,
            "matlab_export": config.features.matlab_export,
            "realtime_updates": config.features.realtime_updates,
        },
        "database": {
            "type": config.database.type
        },
        "auth": {
            "enabled": config.auth.enabled
        }
    }


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle FileNotFoundError exceptions."""
    logger.error(f"FileNotFoundError: {exc}")
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)}
    )


# Import and include routers
# These will be added as we build the API
# from .api.routes import patients, planning, dose, optimization, sequencing, viz

# app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
# app.include_router(planning.router, prefix="/api/plans", tags=["Planning"])
# app.include_router(dose.router, prefix="/api/dose", tags=["Dose Calculation"])
# app.include_router(optimization.router, prefix="/api/optimize", tags=["Optimization"])
# app.include_router(sequencing.router, prefix="/api/sequencing", tags=["Sequencing"])
# app.include_router(viz.router, prefix="/api/viz", tags=["Visualization"])


if __name__ == "__main__":
    import uvicorn

    # Get config for runtime
    config = get_config()

    uvicorn.run(
        "backend.main:app",
        host=config.network.backend_host,
        port=config.network.backend_port,
        reload=True,  # Enable hot reload in development
        log_level="info"
    )
