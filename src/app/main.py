import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from lib.api import router as face_router, face_service
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
async def health() -> dict[str, str]:
    if face_service.app is None:
        logger.error("InsightFace FaceAnalysis app is not initialized")
        raise HTTPException(status_code=500, detail="Face analysis model is not initialized")

    return {"status": "ok", "model": "insightface-buffalo_l"}
