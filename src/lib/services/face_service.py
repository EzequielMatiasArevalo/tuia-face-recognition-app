from __future__ import annotations

import json
import threading
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
from insightface.model_zoo import get_model as insightface_get_model
from insightface.utils import face_align
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
        detector_path: Path | None = None,
        det_score_threshold: float = 0.5,
        det_input_size: int = 640,
        min_face_size: int = 20,
        align_margin: float = 0.5,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.model: any = self._load_model(model_path)
        self.output_path = output_path
        self.det_score_threshold = det_score_threshold
        self.det_input_size = det_input_size
        self.min_face_size = min_face_size
        self.align_margin = align_margin
        # Ultima deteccion sobre imagen completa, por hilo (ver _detect_cached)
        self._local = threading.local()
        self.detector: any = self._load_detector(detector_path)

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

    def _load_detector(self, detector_path: Path | None) -> any:
        """
        Carga SCRFD (det_10g.onnx de InsightFace) UNA sola vez (se llama desde __init__).
        Es el mismo detector que se uso en 1-pre-processing.ipynb, con los mismos
        parametros (DET_SCORE_THRESHOLD y DET_INPUT_SIZE salen del .env).
        """
        if detector_path is None:
            raise ValueError(
                "DETECTOR_MODEL_NAME no esta configurado: indicar el archivo del detector "
                "SCRFD (por ejemplo det_10g.onnx) dentro de MODEL_PATH."
            )
        dp = Path(detector_path)
        if not dp.exists():
            raise ValueError(f"Detector path does not exist: {detector_path}")
        detector = insightface_get_model(str(dp), providers=["CPUExecutionProvider"])
        size = int(self.det_input_size)
        detector.prepare(
            ctx_id=-1,
            input_size=(size, size),
            det_thresh=float(self.det_score_threshold),
        )
        return detector

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image
    
    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """Intersection over union between two boxes (x1, y1, x2, y2)."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return float(inter / union) if union > 0 else 0.0
    
    # ------------------------------------------------------------------ #
    #  Pipeline de rostros (mismo que 1-pre-processing.ipynb)             #
    #  imagen BGR uint8 -> SCRFD -> norm_crop 5 landmarks -> ArcFace      #
    # ------------------------------------------------------------------ #

    # IoU minimo para reconocer, entre las detecciones, el rostro que corresponde a la caja dada.
    _IOU_MATCH_MIN = 0.3
    # Normalizacion de ArcFace (w600k_r50): (x - 127.5) / 127.5, canales RGB, NCHW.
    _INPUT_MEAN = 127.5
    _INPUT_STD = 127.5

    def _detect_raw(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """SCRFD sobre una imagen BGR. Devuelve (bboxes (N,5) con score, kps (N,5,2) o None)."""
        bboxes, kpss = self.detector.detect(image, max_num=0, metric="default")
        return bboxes, kpss

    def _detect_cached(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Deteccion sobre la imagen COMPLETA. Se guarda la ultima por hilo: predict() llama a
        detect_faces y luego a align_face por cada caja con el mismo objeto `image`, asi se
        detecta una sola vez por imagen y los landmarks son identicos a los de la notebook.
        """
        last = getattr(self._local, "last", None)
        if last is not None and last[0] is image:
            return last[1], last[2]
        bboxes, kpss = self._detect_raw(image)
        self._local.last = (image, bboxes, kpss)
        return bboxes, kpss

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Each box is (x1, y1, x2, y2) in pixels (InsightFace convention).
        Return a list of tuples with the coordinates of the faces detected in the image.
        Devuelve [] si no hay rostros. Descarta detecciones con score < DET_SCORE_THRESHOLD
        (lo aplica el propio detector) o con lado menor < MIN_FACE_SIZE.
        """
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        bboxes, _ = self._detect_cached(image)
        if bboxes is None or len(bboxes) == 0:
            return []

        min_side = float(self.min_face_size)
        boxes: list[tuple[int, int, int, int]] = []
        # de mayor a menor score
        for det in sorted(bboxes, key=lambda b: -float(b[4])):
            if float(det[4]) < float(self.det_score_threshold):
                continue
            x1, y1, x2, y2 = self._clip_xyxy(
                int(round(det[0])), int(round(det[1])),
                int(round(det[2])), int(round(det[3])),
                height, width,
            )
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _landmarks_for_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        """
        Landmarks (5, 2), en coordenadas de la imagen completa, del rostro que corresponde a `box`.
        1) Deteccion sobre la imagen completa (igual que en la notebook) y match por IoU con `box`.
        2) Si no hay match, se reintenta sobre un recorte con margen ALIGN_MARGIN alrededor de la caja.
        Es stateless desde afuera: no exige haber llamado antes a detect_faces.
        """
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        target = np.asarray(box, dtype=np.float32)

        def match(bboxes, kpss, ox: int, oy: int) -> np.ndarray | None:
            if bboxes is None or len(bboxes) == 0 or kpss is None:
                return None
            offset = np.array([ox, oy, ox, oy], dtype=np.float32)
            ious = [self._iou(target, b[:4].astype(np.float32) + offset) for b in bboxes]
            best = int(np.argmax(ious))
            if ious[best] < self._IOU_MATCH_MIN:
                return None
            return kpss[best].astype(np.float32) + np.array([ox, oy], dtype=np.float32)

        bboxes, kpss = self._detect_cached(image)
        kps = match(bboxes, kpss, 0, 0)
        if kps is not None:
            return kps

        mx = int(round((x2 - x1) * float(self.align_margin)))
        my = int(round((y2 - y1) * float(self.align_margin)))
        cx1, cy1, cx2, cy2 = self._clip_xyxy(x1 - mx, y1 - my, x2 + mx, y2 + my, height, width)
        bboxes, kpss = self._detect_raw(image[cy1:cy2, cx1:cx2])
        return match(bboxes, kpss, cx1, cy1)

    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> AlignedFace:
        """
        Alinea el rostro de `box` con los 5 landmarks (transformacion de similitud hacia la
        plantilla ArcFace, face_align.norm_crop) y lo deja de face_size x face_size, BGR uint8.
        keypoints: array (5, 2) RELATIVO a la esquina (x1, y1) de la caja (el frontend les
        vuelve a sumar (x1, y1)). Orden: ojo izq, ojo der, nariz, comisura izq, comisura der.
        """
        x1, y1, x2, y2 = (int(v) for v in box)
        kps = self._landmarks_for_box(image, (x1, y1, x2, y2))

        if kps is None:
            # Sin landmarks no se puede alinear: recorte simple como ultimo recurso.
            logger.warning("No se hallaron landmarks para la caja %s; recorte sin alinear", box)
            crop = image[y1:y2, x1:x2]
            face_img = cv2.resize(
                crop, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA
            )
            return AlignedFace(bbox=(x1, y1, x2, y2), keypoints=None, image=face_img)

        face_img = face_align.norm_crop(image, landmark=kps, image_size=self.face_size)
        return AlignedFace(
            bbox=(x1, y1, x2, y2),
            keypoints=kps - np.array([x1, y1], dtype=np.float32),
            image=face_img,
        )

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """BGR uint8 (H, W, 3) -> float32 (1, 3, H, W), RGB, (x - 127.5) / 127.5."""
        return cv2.dnn.blobFromImage(
            face_image,
            scalefactor=1.0 / self._INPUT_STD,
            size=(face_image.shape[1], face_image.shape[0]),
            mean=(self._INPUT_MEAN, self._INPUT_MEAN, self._INPUT_MEAN),
            swapRB=True,
        ).astype(np.float32)

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Embedding normalizado L2 del rostro alineado, como lista de floats de Python.
        Longitud = EMBEDDING_DIM (512 en ArcFace).
        """
        img = face.image
        if isinstance(self.model, onnxruntime.InferenceSession):
            inp = self.model.get_inputs()[0]
            # forma estatica esperada (1, 3, H, W); si el rostro no coincide, se redimensiona
            h, w = inp.shape[2], inp.shape[3]
            if isinstance(h, int) and isinstance(w, int) and img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            out = self.model.run(None, {inp.name: self._preprocess(img)})[0]
            emb = np.asarray(out, dtype=np.float32).reshape(-1)
        else:
            self.model.eval()
            with torch.no_grad():
                emb = self.model(torch.from_numpy(self._preprocess(img))).cpu().numpy().reshape(-1)

        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb.astype(float).tolist()

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
            kps_arr = np.asarray(kps) if kps is not None else None
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
