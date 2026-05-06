import cv2
from pathlib import Path

from lib.services.face_service import FaceService
from lib.storage.embedding_store import EmbeddingStore


class TestFaceService(FaceService):
    def _load_model(self, model_path: Path):
        # Para testear solo detección no necesitamos cargar todavía
        # el modelo de embeddings.
        return None


store = EmbeddingStore(Path("data/embeddings.json"))

service = TestFaceService(
    store=store,
    similarity_metric="cosine",
    similarity_threshold=0.5,
    face_size=112,
    model_path=Path("model/dummy.onnx"),
)

image = cv2.imread("test.jpg")

if image is None:
    raise ValueError("No se pudo cargar la imagen. Revisá nombre y ubicación del archivo.")

boxes = service.detect_faces(image)

print("Faces detected:", boxes)

for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()