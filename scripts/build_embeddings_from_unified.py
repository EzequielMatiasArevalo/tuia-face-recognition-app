#!/usr/bin/env python3
"""
Fase 2: embeddings a partir del dataset unificado (ImageFolder-like).

Lee:   <data-dir>/unified/<etiqueta>/*.jpg  (salida de scripts/unify_dataset.py)
Escribe: JSON lista de registros compatibles con EmbeddingStore / EmbeddingRecord.

Uso:
  cd tuia-face-recognition-app
  source .venv/bin/activate
  python scripts/build_embeddings_from_unified.py

  python scripts/build_embeddings_from_unified.py --input-dir ./data/unified --output-json ./data/embeddings_unified.json

Sin venv, el python del sistema suele fallar con: ModuleNotFoundError: No module named 'torch'.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
except ModuleNotFoundError as exc:
    venv_py = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    print(
        "No está instalado PyTorch en este intérprete.\n"
        "Activá el entorno del proyecto e intentá de nuevo:\n\n"
        "  cd tuia-face-recognition-app\n"
        "  source .venv/bin/activate\n"
        "  python scripts/build_embeddings_from_unified.py\n\n"
        f"O ejecutá directamente: {venv_py} scripts/build_embeddings_from_unified.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
from PIL import Image


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_unified = root / "data" / "unified"
    default_out = root / "data" / "embeddings_unified.json"
    p = argparse.ArgumentParser(description="Extraer embeddings desde data/unified.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=default_unified,
        help=f"Carpeta unificada (default: {default_unified})",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=default_out,
        help=f"Salida JSON array EmbeddingRecord (default: {default_out})",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Dispositivo para inferencia.",
    )
    p.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp",
        help="Extensiones a incluir (coma, minusculas).",
    )
    return p.parse_args()


def collect_images(root: Path, exts: set[str]) -> list[tuple[Path, str]]:
    """Lista (ruta_imagen, etiqueta) donde etiqueta = nombre de carpeta hija."""
    out: list[tuple[Path, str]] = []
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        label = sub.name
        for img in sorted(sub.iterdir()):
            if not img.is_file():
                continue
            if img.suffix.lower() not in exts:
                continue
            out.append((img.resolve(), label))
    return out


def build_model(device: torch.device) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    m = models.resnet18(weights=weights)
    m.fc = nn.Identity()
    m = m.to(device).eval()
    return m


def build_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def embed_path(path: Path, model: nn.Module, tfm: T.Compose, device: torch.device) -> list[float]:
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    v = model(x).squeeze(0).cpu().numpy().astype(float)
    n = float((v * v).sum() ** 0.5)
    if n > 0:
        v = v / n
    return v.tolist()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    if not exts:
        exts = {".jpg", ".jpeg", ".png", ".webp"}

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not input_dir.is_dir():
        print(
            f"Error: no existe la carpeta unificada:\n  {input_dir}\n\n"
            "Primero ejecutá:\n  python scripts/unify_dataset.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = collect_images(input_dir, exts)
    if not pairs:
        print(f"Error: no hay imágenes bajo {input_dir}", file=sys.stderr)
        sys.exit(1)

    model = build_model(device)
    tfm = build_transform()
    print(f"Device: {device} | Imágenes: {len(pairs)} | Backbone: ResNet18 (512-D, L2)")

    records: list[dict] = []
    for i, (path, etiqueta) in enumerate(pairs):
        try:
            emb = embed_path(path, model, tfm, device)
        except OSError as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        records.append(
            {
                "id_imagen": str(uuid.uuid4()),
                "embedding": emb,
                "path": str(path),
                "etiqueta": etiqueta,
                "metadata": {
                    "source": "build_embeddings_from_unified",
                    "script": "scripts/build_embeddings_from_unified.py",
                },
            }
        )
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(pairs)}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Registros escritos: {len(records)}")
    print(f"Archivo: {output_json}")
    print(
        "Para usar en el backend, apuntá EMBEDDINGS_PATH a este archivo "
        "y USE_PGVECTOR=false (o importá a pgvector en otra fase)."
    )


if __name__ == "__main__":
    main()
