from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import PIL
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
        from facenet_pytorch import MTCNN
        import torchvision.transforms as T
        
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: any = self._load_model(model_path)
        self.output_path = output_path
        self.detector = MTCNN(image_size=face_size, margin=0, keep_all=True, post_process=False, device=self.device)
        
        self._embed_tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        os.makedirs(self.output_path, exist_ok=True)

    def _load_model(self, model_path: Path):
        import timm
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            # Reconstruir arquitectura y cargar state_dict; head=Identity para extraer features 768-D
            state_dict = torch.load(mp, map_location=self.device, weights_only=True)
            # num_classes se infiere del state_dict: head.weight tiene shape (N, 768)
            num_classes = state_dict["head.weight"].shape[0] if "head.weight" in state_dict else 0
            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
            model.load_state_dict(state_dict)
            model.head = torch.nn.Identity()
            return model.to(self.device).eval()
        if suf == ".onnx":
            return onnxruntime.InferenceSession(str(mp))
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    # def _load_model(self, model_path: Path) -> any:
    #     mp = Path(model_path)
    #     if not mp.exists():
    #         raise ValueError(f"Model path does not exist: {model_path}")
    #     suf = mp.suffix.lower()
    #     if suf == ".pth":
    #         return torch.load(mp, map_location="cpu", weights_only=False)
    #     if suf == ".onnx":
    #         return onnxruntime.InferenceSession(str(mp))
    #     raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

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

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image
        # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _to_pil(image: np.ndarray) -> PIL.Image.Image:
        if np.issubdtype(image.dtype, np.floating):
            return PIL.Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))
        return PIL.Image.fromarray(image.astype(np.uint8))

    def detect_faces(self, image: np.ndarray) -> list[tuple[tuple[int,int,int,int], np.ndarray]]:
        """
        Each box is (x1, y1, x2, y2) in pixels (InsightFace convention).
        Return a list of tuples with the coordinates of the faces detected in the image
        and the landmarks found.
        """

        pil_img = self._to_pil(image)
        boxes, _, landmarks = self.detector.detect(pil_img, landmarks=True)
        if boxes is None:
            return []
        return [
            (tuple(map(int, b)), kps.astype(np.float32))
            for b, kps in zip(boxes, landmarks)
        ]

    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int],keypoints: np.ndarray,
    ) -> AlignedFace:
        """
        Crop using box (x1, y1, x2, y2) and run FaceAnalysis on the crop.
        Return an AlignedFace object.
        """
        pil = self._to_pil(image)
        box_np = np.asarray([box], dtype=np.float32)  # MTCNN.extract espera array (N, 4)
        crop = self.detector.extract(pil, box_np, save_path=None)
        if crop is None:
            raise ValueError("MTCNN no pudo extraer el crop para esa bbox.")
        if crop.ndim == 4:
            crop = crop[0]
        aligned = crop.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # RGB
        return AlignedFace(bbox=list(box), keypoints=keypoints, image=aligned)        

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extract embedding from face.
        Return a list of floats representing the embedding of the face.
        """
        import torch.nn.functional as F
        pil = PIL.Image.fromarray(face.image.astype(np.uint8))
        x = self._embed_tf(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            v = self.model(x)                       # (1, 768)
            v = F.normalize(v, dim=-1)              # L2-normalize para cosine similarity
        return v.squeeze(0).cpu().numpy().astype(np.float32).tolist()
        
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
        
        box, kps = faces[0]
        
        logger.info(f"Face detected at: {box}")

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
        # cv2.imwrite(str(img_output_path), cv2.cvtColor(aligned.image, cv2.COLOR_RGB2BGR))
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)

        detections: list[FaceDetection] = []
        for box, kps in faces:
            x1, y1, x2, y2 = box
            aligned = self.align_face(image, box, kps)
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
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
