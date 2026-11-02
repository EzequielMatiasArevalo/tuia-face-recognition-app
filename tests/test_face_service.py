"""Tests de FaceService (detect_faces / align_face / extract_embedding_from_face).

Requieren models/det_10g.onnx y models/w600k_r50.onnx (se saltean si no estan).
Correr desde la raiz del repo:  pip install pytest && PYTHONPATH=src pytest tests -v
"""
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage import data

from lib.services.face_service import FaceService

MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    not ((MODELS / "det_10g.onnx").exists() and (MODELS / "w600k_r50.onnx").exists()),
    reason="faltan models/det_10g.onnx y models/w600k_r50.onnx",
)


class MemoryStore:
    def __init__(self):
        self.records = []

    def all(self):
        return list(self.records)

    def append(self, record):
        self.records.append(record)


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    return FaceService(
        store=MemoryStore(), similarity_metric="cosine", similarity_threshold=0.5, face_size=112,
        model_path=MODELS / "w600k_r50.onnx", output_path=tmp_path_factory.mktemp("out"),
        detector_path=MODELS / "det_10g.onnx", det_score_threshold=0.5, det_input_size=640,
        min_face_size=40, align_margin=0.5,
    )


@pytest.fixture(scope="module")
def face_image():
    """Foto con un rostro (astronauta de scikit-image), en BGR uint8 como cv2.imread."""
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


def test_detect_faces_sin_rostros_devuelve_lista_vacia(service):
    assert service.detect_faces(np.zeros((300, 300, 3), dtype=np.uint8)) == []
    assert service.detect_faces(cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)) == []


def test_detect_faces_devuelve_cajas_enteras_dentro_de_la_imagen(service, face_image):
    boxes = service.detect_faces(face_image)
    assert len(boxes) == 1
    h, w = face_image.shape[:2]
    x1, y1, x2, y2 = boxes[0]
    assert all(isinstance(v, int) for v in boxes[0])
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_detect_faces_descarta_rostros_menores_a_min_face_size(service, face_image):
    chica = cv2.resize(face_image, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    assert service.detect_faces(chica) == []


def test_align_face_forma_tipo_y_keypoints_relativos(service, face_image):
    box = service.detect_faces(face_image)[0]
    aligned = service.align_face(face_image, box)
    assert aligned.image.shape == (112, 112, 3) and aligned.image.dtype == np.uint8
    assert aligned.keypoints.shape == (5, 2)
    x1, y1, x2, y2 = box
    # relativos a la caja: caen dentro del tamano de la caja, no en coordenadas absolutas
    assert (aligned.keypoints[:, 0] > 0).all() and (aligned.keypoints[:, 0] < (x2 - x1) * 1.2).all()
    assert (aligned.keypoints[:, 1] > 0).all() and (aligned.keypoints[:, 1] < (y2 - y1) * 1.2).all()
    # ojo izquierdo a la izquierda del derecho, y nariz entre ojos y boca
    assert aligned.keypoints[0, 0] < aligned.keypoints[1, 0]
    assert aligned.keypoints[0, 1] < aligned.keypoints[2, 1] < aligned.keypoints[3, 1]


def test_embedding_dimension_norma_y_tipos(service, face_image):
    box = service.detect_faces(face_image)[0]
    emb = service.extract_embedding_from_face(service.align_face(face_image, box))
    assert isinstance(emb, list) and len(emb) == 512
    assert all(type(v) is float for v in emb)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-4)


def test_misma_persona_con_variaciones_es_mas_similar_que_el_umbral(service, face_image):
    def embed(img):
        return service.extract_embedding_from_face(service.align_face(img, service.detect_faces(img)[0]))

    h, w = face_image.shape[:2]
    rotada = cv2.warpAffine(face_image, cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0), (w, h))
    oscura = cv2.convertScaleAbs(face_image, alpha=0.6, beta=0)
    base = embed(face_image)
    assert service.similarity(base, embed(rotada)) > 0.8
    assert service.similarity(base, embed(oscura)) > 0.8


def test_registro_con_dos_rostros_falla(service, face_image, tmp_path):
    dos = np.hstack([face_image, cv2.flip(face_image, 1)])
    path = tmp_path / "dos.jpg"
    cv2.imwrite(str(path), dos)
    with pytest.raises(ValueError):
        service.register_identity("x", str(path), {})


def test_registro_y_prediccion(service, face_image, tmp_path):
    foto = tmp_path / "ok.jpg"
    cv2.imwrite(str(foto), face_image)
    service.register_identity("astronauta", str(foto), {})
    import json
    resultado = json.loads(Path(service.predict(str(foto), tmp_path / "pred")).read_text())
    assert resultado["detected_people"] == ["astronauta"]
    assert set(resultado["detections"][0]["keypoints"]) == {"k0", "k1", "k2", "k3", "k4"}
