from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from PIL import Image

import cv2
import numpy as np
from insightface.app import FaceAnalysis

class EmbeddingRecord(BaseModel):
    id_imagen: str
    embedding: list[float]
    path: str
    etiqueta: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InsertRequest(BaseModel):
    identity: str
    image_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlignedFace(BaseModel):
    """One aligned face from insightface (bbox/kps/image may be numpy arrays)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: Any
    keypoints: Any
    image: Any
    embedding: Optional[list[float]] = None




class PredictRequest(BaseModel):
    source_path: str
    source_type: Literal["image", "video"] = "image"


class AsyncTaskCreated(BaseModel):
    status: Literal["accepted"] = "accepted"
    job_id: str


class UploadResponse(BaseModel):
    """Respuesta tras subir un archivo al servidor (rutas usadas por /predict y /insert)."""

    path: str
    download_url: str


class StatusResponse(BaseModel):
    status: Literal["done", "inProgress", "failed"]
    link: str
    reason: Optional[str] = None
    artifact_url: Optional[str] = Field(
        default=None,
        description="URL relativa al API del artefacto principal (.json o imagen de registro).",
    )
    source_image_url: Optional[str] = Field(
        default=None,
        description="URL relativa de la imagen origen (predicción con resultado JSON).",
    )

class FaceDetection(BaseModel):
    bbox: list[int]
    keypoints: dict[str, list[int]]
    label: str
    score: float


class PredictResult(BaseModel):
    source_path: str
    detections: list[FaceDetection]
    detected_people: list[str]
