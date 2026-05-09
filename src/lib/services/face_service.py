from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
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
        self.model: any = self._load_model(model_path)
        self.output_path = output_path

        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"])
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))

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

    # ==========================================
    # ETAPA 1: DETECCIÓN (Detección de Rostros y Keypoints)
    # ==========================================
    def detect_faces(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        """
        Return a list of tuples containing the bounding box and the keypoints.
        """
        faces = self.face_analyzer.get(image)
        result = []
        for face in faces:
            bbox = tuple(map(int, face.bbox))
            result.append((bbox, face.kps))
        return result


    # ==========================================
    # ETAPA 2: ALINEACIÓN (Recorte y Transformación Afín)
    # ==========================================
    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int], kps: np.ndarray | None = None
    ) -> AlignedFace:
        """
        Crop using box (x1, y1, x2, y2) and run FaceAnalysis on the crop.
        Return an AlignedFace object.
        """
        from insightface.utils import face_align
        
        if kps is not None:
            crop = face_align.norm_crop(image, landmark=kps, image_size=self.face_size)
        else:
            x1, y1, x2, y2 = self._clip_xyxy(box[0], box[1], box[2], box[3], image.shape[0], image.shape[1])
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                crop = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (self.face_size, self.face_size))
            
        return AlignedFace(image=crop, bbox=list(box), keypoints=kps)

    # ==========================================
    # ETAPA 3: EMBEDDINGS (Extracción de Características Vectoriales)
    # ==========================================
    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extract embedding from face.
        Return a list of floats representing the embedding of the face.
        """
        img_rgb = cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB)
        
        if isinstance(self.model, torch.nn.Module):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(img_rgb).unsqueeze(0).to(next(self.model.parameters()).device)
            
            self.model.eval()
            with torch.no_grad():
                embedding = self.model(tensor)
            
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.squeeze(0).tolist()
        else:
            # Assume ONNX
            img_resized = cv2.resize(img_rgb, (self.face_size, self.face_size))
            img_normalized = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            tensor_np = np.expand_dims(img_transposed, axis=0)
            
            input_name = self.model.get_inputs()[0].name
            embedding = self.model.run(None, {input_name: tensor_np})[0]
            
            # Normalize L2
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding[0].tolist()
        
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

    # ==========================================
    # ETAPA 4: COMPARACIÓN (Búsqueda de Similitud e Identificación)
    # ==========================================
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
        
        logger.info(f"Face detected: {faces[0][0]}")

        box, kps = faces[0]
        aligned = self.align_face(image, box, kps)
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
        for (box, kps) in faces:
            aligned = self.align_face(image, box, kps)
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            detections.append(
                FaceDetection(
                    bbox=list(box),
                    keypoints=self._kps_to_keypoints_dict(kps),
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
