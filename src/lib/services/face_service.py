from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align as insightface_align
from lib.schemas import EmbeddingRecord, FaceDetection, PredictResult, AlignedFace
from lib.storage.base import EmbeddingStoreProtocol
import os
import logging

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        face_size: int,
        model_path: Path,
        output_path: Path = Path("output"),
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.output_path = output_path

        # Intentar cargar el modelo ONNX/PTH si existe; si no, InsightFace maneja los embeddings.
        try:
            self.model = self._load_model(model_path)
        except (ValueError, Exception) as exc:
            logger.warning("No se pudo cargar el modelo en %s: %s. Se usará InsightFace directamente.", model_path, exc)
            self.model = None

        # Inicializamos InsightFace FaceAnalysis (RetinaFace + ArcFace buffalo_l).
        self._analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self._analyzer.prepare(ctx_id=-1, det_size=(320, 320))
        logger.info("InsightFace FaceAnalysis inicializado correctamente.")

        # Cache liviano: evita re-analizar la misma imagen.
        self._cache_ref: object = None
        self._cache_faces: list = []

        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    @staticmethod
    def _kps_to_keypoints_dict(kps: np.ndarray | None) -> dict[str, list[int]]:
        if kps is None or len(kps) == 0:
            return {}
        return {
            f"k{i}": [int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))]
            for i in range(len(kps))
        }


    def _load_model(self, model_path: Path) -> any:
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            return torch.load(mp, map_location="cpu", weights_only=False)
        if suf == ".onnx":
            return onnxruntime.InferenceSession(str(mp))
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image

    def _get_faces_cached(self, image: np.ndarray) -> list:
        """Analiza la imagen con InsightFace; usa cache si es el mismo objeto imagen.
        Usa 'is' en vez de id() para evitar colisiones cuando Python reutiliza direcciones."""
        if self._cache_ref is not image:
            self._cache_faces = self._analyzer.get(image)
            self._cache_ref = image  # referencia viva → el id() no se reutiliza
        return self._cache_faces

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detecta rostros usando RetinaFace (InsightFace buffalo_l).
        Filtra caras muy pequeñas (artefactos de fondo) y retorna (x1, y1, x2, y2).
        """
        faces = self._get_faces_cached(image)
        h, w = image.shape[:2]
        # Ignorar caras cuya área sea menor al 0.5% del frame (ruido de fondo).
        min_area = w * h * 0.005
        boxes = []
        for face in faces:
            x1, y1, x2, y2 = (int(c) for c in face.bbox[:4])
            x1, y1, x2, y2 = self._clip_xyxy(x1, y1, x2, y2, h, w)
            if (x2 - x1) * (y2 - y1) < min_area:
                logger.debug("Cara ignorada por tamaño pequeño: bbox=(%d,%d,%d,%d)", x1, y1, x2, y2)
                continue
            boxes.append((x1, y1, x2, y2))
        return boxes

    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> AlignedFace:
        """
        Alinea la cara usando los 5 keypoints faciales de InsightFace (norm_crop).
        Busca la cara cacheada más cercana al bounding box dado.
        """
        faces = self._get_faces_cached(image)
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]

        best_face = None
        best_dist = float("inf")
        for face in faces:
            # Aplicar el mismo clipping que detect_faces para comparar correctamente.
            fx1, fy1, fx2, fy2 = (int(c) for c in face.bbox[:4])
            fx1, fy1, fx2, fy2 = self._clip_xyxy(fx1, fy1, fx2, fy2, h, w)
            dist = abs(fx1 - x1) + abs(fy1 - y1)
            if dist < best_dist:
                best_dist = dist
                best_face = face

        if best_face is not None and best_dist < 30:
            kps = best_face.kps
            aligned_img = insightface_align.norm_crop(image, kps, image_size=self.face_size)
            emb: list[float] | None = None
            if best_face.normed_embedding is not None:
                emb = best_face.normed_embedding.tolist()
            return AlignedFace(
                bbox=best_face.bbox.tolist(),
                keypoints=kps,
                image=aligned_img,
                embedding=emb,
            )

        # Fallback: recorte simple (sin embedding de InsightFace disponible).
        logger.warning("align_face: no se encontró match para bbox %s (dist=%.1f).", box, best_dist)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image
        crop = cv2.resize(crop, (self.face_size, self.face_size))
        return AlignedFace(bbox=list(box), keypoints=None, image=crop, embedding=None)

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extrae el embedding facial.
        Prioridad:
          1. Embedding pre-calculado por InsightFace (almacenado en face.embedding).
          2. Modelo ONNX propio (self.model) si está cargado.
        """
        # 1. Usar embedding de InsightFace si ya fue calculado en align_face.
        if face.embedding is not None:
            return face.embedding

        # 2. Usar modelo ONNX (ArcFace) cargado desde MODEL_PATH.
        if self.model is not None:
            img = cv2.resize(face.image, (112, 112)).astype(np.float32)
            img = img[:, :, ::-1]           # BGR → RGB
            img = (img - 127.5) / 128.0    # normalización estándar ArcFace
            img = img.transpose(2, 0, 1)[np.newaxis]
            input_name = self.model.get_inputs()[0].name
            out = self.model.run(None, {input_name: img})[0]
            arr = np.array(out[0], dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()

        raise ValueError("No hay modelo ni embedding disponible para extract_embedding_from_face.")
        
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    def identify(self, query_embedding: list[float]) -> tuple[str, float]:
        records = self.store.all()
        if not records:
            return "unknown", 0.0

        best_label = "unknown"
        best_score = -1.0
        for record in records:
            score = self.similarity(query_embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_label = record.etiqueta

        if best_score < self.similarity_threshold:
            return "unknown", max(best_score, 0.0)
        return best_label, best_score

    def register_identity(
        self, identity: str, image_path: str, metadata: dict[str, object]
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        faces = self.detect_faces(image)

        if len(faces) != 1:
            raise ValueError("Exactly one face must be detected for identity registration.")
        
        logger.info(f"Face detected: {faces[0]}")

        box = faces[0]
        aligned = self.align_face(image, box)
        embedding = self.extract_embedding_from_face(aligned)

        img_id = str(uuid4())
        img_output_path = self.output_path / f"img_{img_id}.jpg"
        
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(img_output_path),
            etiqueta=identity,
            metadata=metadata,
        )
        self.store.append(record)

        cv2.imwrite(str(img_output_path), aligned.image)
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)
        detections: list[FaceDetection] = []
        for (x1, y1, x2, y2) in faces:
            aligned = self.align_face(image, (x1, y1, x2, y2))
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            kps = getattr(aligned, "keypoints", None)
            kps_arr = None
            if kps is not None:
                # El frontend espera keypoints en coordenadas relativas al recorte (bbox).
                # face.kps de InsightFace son coordenadas absolutas: Restamos el offset del bbox.
                kps_full = np.asarray(kps, dtype=float)
                kps_arr = kps_full.copy()
                kps_arr[:, 0] -= x1
                kps_arr[:, 1] -= y1
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=self._kps_to_keypoints_dict(kps_arr),
                    label=label,
                    score=round(float(score), 4),
                )
            )

        detected_people = sorted({item.label for item in detections if item.label != "unknown"})
        result_payload = PredictResult(
            source_path=source_path,
            detections=detections,
            detected_people=detected_people,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(result_payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
