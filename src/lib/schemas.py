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

def process_and_align(img_pil: Image.Image) -> list[AlignedFace]:

    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640)) 
    cv2_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    faces = app.get(cv2_img)
    
    aligned_results = []
    
    for face in faces:
        face_rgb = cv2.cvtColor(face.norm_face, cv2.COLOR_BGR2RGB)
        aligned_image_pil = Image.fromarray(face_rgb)
        

        aligned_face = AlignedFace(
            bbox=face.bbox,
            keypoints=face.kps,
            image=aligned_image_pil, 
            embedding=face.embedding.tolist()
        )
        
        aligned_results.append(aligned_face)
        
    return aligned_results


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
