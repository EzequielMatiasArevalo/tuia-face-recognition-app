#!/usr/bin/env python3
"""
Fase 1: unificar datasets locales en estructura ImageFolder.

Salida por defecto: <repo>/data/unified/<identity>/*.jpg

Fuentes por defecto (solo los 2 ZIP en <repo>/data/):
  - archive.zip                       (LFW deepfunneled)
  - Fri_May_01_18-27-27_CEST_2026.zip (FACES, id por prefijo numérico)

Opcional: --include-public-faces para mezclar también data/public_faces/ (no forma parte del flujo por defecto).

Uso:
  cd tuia-face-recognition-app
  python scripts/unificar_dataset.py

  # Carpeta con los ZIPs: la del repo (relativa), no /data del sistema
  python scripts/unificar_dataset.py --data-dir ./data

  python scripts/unificar_dataset.py --data-dir /ruta/absoluta/al/repo/data --out-dir /ruta/salida
"""
from __future__ import annotations

import argparse
import sys
import re
import shutil
import zipfile
from pathlib import Path


def repo_root() -> Path:
    # scripts/unificar_dataset.py -> parents[1] == raíz del repo
    return Path(__file__).resolve().parents[1]


def safe_id(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.strip())
    return s[:120] or "unknown"


def copy_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def unique_dst(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suf = base.stem, base.suffix
    parent = base.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"No se pudo generar nombre único para {base}")


def merge_lfw_from_archive_zip(zpath: Path, out_root: Path, prefix: str = "lfw") -> int:
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            parts = Path(name).parts
            if len(parts) >= 2:
                person = parts[-2]
            else:
                continue
            rel = f"{prefix}_{safe_id(person)}"
            data = z.read(name)
            fname = Path(name).name
            dst = unique_dst(out_root / rel / fname)
            copy_bytes(dst, data)
            n += 1
    return n


def merge_faces_zip(zpath: Path, out_root: Path, prefix: str = "faces") -> int:
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            base = Path(name).name
            m = re.match(r"^(\d{3})_", base)
            pid = m.group(1) if m else "unknown"
            rel = f"{prefix}_{pid}"
            data = z.read(name)
            dst = unique_dst(out_root / rel / base)
            copy_bytes(dst, data)
            n += 1
    return n


def merge_public_faces_dir(src: Path, out_root: Path, prefix: str = "public_faces") -> int:
    """Copia data/public_faces/<id>/*.jpg -> unified/public_faces_<id>/"""
    if not src.is_dir():
        return 0
    n = 0
    for sub in sorted(src.iterdir()):
        if not sub.is_dir():
            continue
        ident = safe_id(sub.name)
        rel = f"{prefix}_{ident}"
        for img in sorted(sub.glob("*.jpg")):
            dst = unique_dst(out_root / rel / img.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_data = root / "data"
    p = argparse.ArgumentParser(description="Unificar ZIPs locales en estructura ImageFolder.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help=(
            f"Carpeta del repo que contiene los .zip (default: {default_data}). "
            "Usá ./data desde la raíz del repo; no uses /data (raíz del sistema)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Salida unified (default: <data-dir>/unified)",
    )
    p.add_argument(
        "--archive-zip",
        type=Path,
        default=None,
        help="Ruta a archive.zip (default: <data-dir>/archive.zip)",
    )
    p.add_argument(
        "--faces-zip",
        type=Path,
        default=None,
        help="Ruta al ZIP FACES (default: <data-dir>/Fri_May_01_18-27-27_CEST_2026.zip)",
    )
    p.add_argument(
        "--include-public-faces",
        action="store_true",
        help="Además de los 2 ZIP, copiar <data-dir>/public_faces/ (opcional; desactivado por defecto).",
    )
    p.add_argument(
        "--public-faces-dir",
        type=Path,
        default=None,
        help="Solo con --include-public-faces: carpeta a copiar (default: <data-dir>/public_faces).",
    )
    p.add_argument(
        "--no-wipe",
        action="store_true",
        help="No borrar out-dir antes (añade sin limpiar; puede duplicar nombres con sufijo).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.expanduser().resolve()
    out: Path = (args.out_dir or (data_dir / "unified")).expanduser().resolve()

    if not data_dir.is_dir():
        print(
            f"Error: --data-dir no es una carpeta existente:\n  {data_dir}\n\n"
            "Suele confundirse `/data/` (raíz del sistema) con la carpeta `data/` del proyecto.\n"
            "Ejemplos correctos:\n"
            f"  python scripts/unificar_dataset.py\n"
            f"  python scripts/unificar_dataset.py --data-dir ./data\n"
            f"  python scripts/unificar_dataset.py --data-dir {repo_root() / 'data'}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    archive_zip = (args.archive_zip or (data_dir / "archive.zip")).resolve()
    faces_zip = (args.faces_zip or (data_dir / "Fri_May_01_18-27-27_CEST_2026.zip")).resolve()
    public_dir = (args.public_faces_dir or (data_dir / "public_faces")).resolve()

    if not args.no_wipe and out.exists():
        shutil.rmtree(out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"Error: no se pudo crear la carpeta de salida:\n  {out}\n"
            f"Motivo: {exc}\n\n"
            "Comprobá permisos y que --out-dir / --data-dir apunten a carpetas donde tengás escritura "
            "(por ejemplo `./data` dentro del repo, no `/data/`).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    total = 0
    if archive_zip.is_file():
        total += merge_lfw_from_archive_zip(archive_zip, out)
    else:
        print(f"[skip] no existe: {archive_zip}")

    if faces_zip.is_file():
        total += merge_faces_zip(faces_zip, out)
    else:
        print(f"[skip] no existe: {faces_zip}")

    if args.include_public_faces:
        if public_dir.is_dir():
            total += merge_public_faces_dir(public_dir, out)
        else:
            print(f"[skip] --include-public-faces pero no existe: {public_dir}", file=sys.stderr)

    print(f"Imágenes escritas: {total}")
    print(f"Salida: {out}")


if __name__ == "__main__":
    main()
