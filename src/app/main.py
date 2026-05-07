import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from lib.api import router as face_router
from lib.config import settings
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Facial Recognition Backend",
    version="0.1.0",
    description="Backend API for TP1 facial recognition system.",
)

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_router)

@app.get("/health", tags=["health"])
async def health() -> dict:
    status = "ok"
    details = []
    
    if not settings.model_name:
        status = "degraded"
        details.append("Model name is not set as environment variable")
        logger.warning("Model name is not set as environment variable")
    
    model_full_path = f"{settings.model_path}/{settings.model_name}" if settings.model_name else None
    if model_full_path and not os.path.exists(model_full_path):
        status = "degraded"
        details.append(f"Model file {settings.model_name} missing at {settings.model_path}")
        logger.warning(f"Model path {model_full_path} does not exist")
        
    return {
        "status": status, 
        "model": settings.model_name,
        "details": details
    }
