import sys
import os
# Add parent directory to Python path to fix import issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from backend.core.config import settings
from backend.db.models import init_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AI Co-Scientist application...")
    
    # Initialize database if persistence is enabled
    if settings.ENABLE_PERSISTENCE:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
    
    # Initialize task queue
    if hasattr(app.state, 'celery_app'):
        logger.info("Task queue initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Co-Scientist application...")

# Create FastAPI application
app = FastAPI(
    title="AI Co-Scientist API",
    description="Multi-agent system for scientific hypothesis generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "persistence_enabled": settings.ENABLE_PERSISTENCE,
        "auth_enabled": settings.AUTH_ENABLED
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Co-Scientist API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Include API routers
from backend.api.rest import router as rest_router
from backend.api.websocket import router as ws_router

app.include_router(rest_router, prefix="/v1", tags=["API"])
app.include_router(ws_router, tags=["WebSocket"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=settings.WORKER_PROCESSES
    ) 