"""
evaluate.py — Evaluación visual del modelo de reconocimiento facial.
Genera gráficos PCA y t-SNE de los embeddings de TODAS las clases del dataset,
incluyendo a Valentino (que no fue parte del entrenamiento).

Salida:
  - Graphics/pca.png
  - Graphics/tsne.png
  - Graphics/evaluation_report.json  (distancias inter/intra clase, vecino más cercano de Valentino)
"""

import os
import glob
import json
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ──────────────────────────── Configuración ────────────────────────────
DATASET_DIR = "Dataset"
MODEL_PATH = "models/face_detection.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE = 112
OUTPUT_DIR = "Graphics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────── Carga ────────────────────────────
print("Cargando modelo...")
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval()

analyzer = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"])
analyzer.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ──────────────────────────── Extracción de embeddings ────────────────────────────
embeddings = []
labels = []
filenames = []
class_names = []

print("Extrayendo embeddings de TODO el dataset (incluyendo Valentino)...")
for cls_name in sorted(os.listdir(DATASET_DIR)):
    cls_path = os.path.join(DATASET_DIR, cls_name)
    if not os.path.isdir(cls_path):
        continue
    class_names.append(cls_name)
    cls_idx = len(class_names) - 1

    files = sorted(glob.glob(os.path.join(cls_path, "*.*")))
    for fpath in files:
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue

        faces = analyzer.get(img_bgr)
        if len(faces) > 0:
            faces_sorted = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )
            aligned_bgr = face_align.norm_crop(
                img_bgr, landmark=faces_sorted[0].kps, image_size=FACE_SIZE
            )
        else:
            aligned_bgr = cv2.resize(img_bgr, (FACE_SIZE, FACE_SIZE))

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().squeeze().numpy())
            labels.append(cls_idx)
            filenames.append(os.path.basename(fpath))

    print(f"  {cls_name}: {sum(1 for l in labels if l == cls_idx)} embeddings extraídos")

X = np.array(embeddings)
y = np.array(labels)

# ──────────────────────────── Colores y estilo ────────────────────────────
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]

def cls_color(idx):
    return PALETTE[idx % len(PALETTE)]

# ──────────────────────────── PCA ────────────────────────────
print("Generando PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(11, 8))
for idx, cls in enumerate(class_names):
    mask = y == idx
    marker = "*" if cls.lower() == "valentino" else "o"
    size = 200 if cls.lower() == "valentino" else 60
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        label=cls.capitalize(), color=cls_color(idx),
        marker=marker, s=size, alpha=0.85, edgecolors="white", linewidths=0.5,
    )
ax.set_title("PCA de Embeddings Faciales (2 componentes principales)", fontsize=14)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pca.png"), dpi=150)
plt.close(fig)

# ──────────────────────────── t-SNE ────────────────────────────
print("Generando t-SNE...")
perplexity = min(30, len(X) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(11, 8))
for idx, cls in enumerate(class_names):
    mask = y == idx
    marker = "*" if cls.lower() == "valentino" else "o"
    size = 200 if cls.lower() == "valentino" else 60
    ax.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        label=cls.capitalize(), color=cls_color(idx),
        marker=marker, s=size, alpha=0.85, edgecolors="white", linewidths=0.5,
    )
ax.set_title("t-SNE de Embeddings Faciales (Valentino marcado con ★)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "tsne.png"), dpi=150)
plt.close(fig)

# ──────────────────────────── Reporte de evaluación ────────────────────────────
print("Calculando métricas de distancia...")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# Distancias intra-clase (promedio de cosine similarity entre pares del mismo sujeto)
intra_class = {}
for idx, cls in enumerate(class_names):
    mask = y == idx
    cls_embs = X[mask]
    if len(cls_embs) < 2:
        intra_class[cls] = None
        continue
    sims = []
    for i in range(len(cls_embs)):
        for j in range(i + 1, len(cls_embs)):
            sims.append(cosine_sim(cls_embs[i], cls_embs[j]))
    intra_class[cls] = round(float(np.mean(sims)), 4)

# Vecino más cercano de cada imagen de Valentino
valentino_idx = None
for idx, cls in enumerate(class_names):
    if cls.lower() == "valentino":
        valentino_idx = idx
        break

valentino_analysis = []
if valentino_idx is not None:
    val_mask = y == valentino_idx
    val_embs = X[val_mask]
    val_files = [filenames[i] for i in range(len(filenames)) if labels[i] == valentino_idx]

    for vi, (v_emb, v_file) in enumerate(zip(val_embs, val_files)):
        best_sim = -1
        best_cls = "?"
        for idx, cls in enumerate(class_names):
            if idx == valentino_idx:
                continue
            for emb in X[y == idx]:
                s = cosine_sim(v_emb, emb)
                if s > best_sim:
                    best_sim = s
                    best_cls = cls
        valentino_analysis.append({
            "file": v_file,
            "nearest_class": best_cls,
            "similarity": round(best_sim, 4),
        })

report = {
    "class_names": class_names,
    "total_embeddings": len(X),
    "intra_class_similarity": intra_class,
    "valentino_nearest_neighbor": valentino_analysis,
    "pca_variance_explained": [round(v, 4) for v in pca.explained_variance_ratio_.tolist()],
}
with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nReporte de evaluación guardado en {OUTPUT_DIR}/evaluation_report.json")
if valentino_analysis:
    print("\n── Análisis de Valentino (unknown) ──")
    for va in valentino_analysis:
        print(f"  {va['file']} → vecino más cercano: {va['nearest_class']} (sim={va['similarity']})")

print("\n¡Evaluación completada con éxito!")
