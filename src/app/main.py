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
async def health() -> dict[str, str]:
    # Sin MODEL_NAME: InsightFace buffalo_l está pre-descargado en la imagen Docker.
    if not settings.model_name:
        logger.info("Usando InsightFace buffalo_l (pre-descargado en imagen Docker).")
        return {"status": "ok", "model": "InsightFace buffalo_l"}
    if not os.path.exists(f"{settings.model_path}/{settings.model_name}"):
        logger.error(f"Model path {settings.model_path}/{settings.model_name} does not exist")
        raise HTTPException(status_code=500, detail=f"Model path {settings.model_path}/{settings.model_name} does not exist")
    logger.info(f"Model name is set to {settings.model_name} located at {settings.model_path}/{settings.model_name}")
    return {"status": "ok", "model": settings.model_name}
