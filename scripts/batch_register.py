"""Registra y testea en batch todas las fotos bajo una carpeta.

Estructura esperada:

    dataset/
      Alejandro/
        foto1.jpg
        foto2.jpg
      Maria/
        ...
      desconocido/        # opcional: fotos de gente NO registrada

El script:
1. Divide las fotos de cada persona conocida en train/test (default 70/30).
2. Registra el set de train via /upload + /insert.
3. Predice el set de test via /upload + /predict y verifica el match.
4. Si existe la subcarpeta 'desconocido/', predice cada foto y verifica que
   el sistema la marque como `unknown` (test fuera del dataset).
5. Imprime un reporte final con accuracy y lista de errores.

Uso:
    python scripts/batch_register.py dataset/
    python scripts/batch_register.py dataset/ --split 0.7 --seed 42
    python scripts/batch_register.py dataset/ --backend http://localhost:8000

Sin dependencias externas: solo stdlib.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import sys
import time
import uuid
from pathlib import Path
from urllib import request as urlrequest
from urllib import error as urlerror

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_DIR_NAMES = {"desconocido", "desconocidos", "unknown", "unknowns"}


def _build_multipart(image_path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    mime, _ = mimetypes.guess_type(image_path.name)
    mime = mime or "image/jpeg"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = head + image_path.read_bytes() + tail
    return body, f"multipart/form-data; boundary={boundary}"


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urlrequest.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 15.0) -> dict:
    with urlrequest.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def upload_image(backend: str, image_path: Path) -> str:
    body, ctype = _build_multipart(image_path)
    req = urlrequest.Request(
        f"{backend}/upload", data=body, headers={"Content-Type": ctype}, method="POST"
    )
    with urlrequest.urlopen(req, timeout=60.0) as r:
        body_json = json.loads(r.read().decode("utf-8"))
    return str(body_json["path"])


def insert_identity(backend: str, identity: str, server_path: str, source_file: str) -> str:
    body = _post_json(
        f"{backend}/insert",
        {
            "identity": identity,
            "image_path": server_path,
            "metadata": {"source": "batch_register", "file": source_file},
        },
    )
    return str(body["job_id"])


def request_predict(backend: str, server_path: str) -> str:
    body = _post_json(
        f"{backend}/predict",
        {"source_path": server_path, "source_type": "image"},
    )
    return str(body["job_id"])


def wait_for_job(backend: str, job_id: str, timeout_s: float = 60.0) -> tuple[str, str, str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = _get_json(f"{backend}/status/{job_id}", timeout=10.0)
        status = body.get("status")
        if status == "done":
            return "done", body.get("link") or "", str(body.get("artifact_url") or "")
        if status == "failed":
            return "failed", body.get("reason") or "sin detalle", ""
        time.sleep(0.4)
    return "timeout", f"superado el limite de {timeout_s}s", ""


def fetch_predict_result(backend: str, artifact_url: str) -> dict | None:
    if not artifact_url:
        return None
    url = artifact_url if artifact_url.startswith("http") else f"{backend}{artifact_url}"
    try:
        with urlrequest.urlopen(url, timeout=15.0) as r:
            content_type = r.headers.get("content-type", "")
            content = r.read()
    except urlerror.URLError:
        return None
    if "json" not in content_type and not url.lower().endswith(".json"):
        return None
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def best_label_from_result(result: dict) -> tuple[str, float]:
    detections = result.get("detections") or []
    if not detections:
        return "unknown", 0.0
    best = max(detections, key=lambda d: float(d.get("score", 0.0)))
    return str(best.get("label", "unknown")), float(best.get("score", 0.0))


def split_train_test(
    images: list[Path], split: float, rng: random.Random
) -> tuple[list[Path], list[Path]]:
    imgs = images.copy()
    rng.shuffle(imgs)
    n_train = max(1, int(round(len(imgs) * split)))
    return imgs[:n_train], imgs[n_train:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Carpeta con subcarpetas por persona")
    parser.add_argument("--backend", default="http://localhost:8000", help="URL del backend")
    parser.add_argument("--split", type=float, default=0.7, help="Fraccion para train (default 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para el split (default 42)")
    parser.add_argument("--no-test", action="store_true", help="Solo registrar, no predecir")
    args = parser.parse_args()

    root: Path = args.folder
    if not root.is_dir():
        print(f"ERROR: {root} no es una carpeta valida", file=sys.stderr)
        return 1

    backend = args.backend.rstrip("/")
    rng = random.Random(args.seed)

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        print(f"ERROR: no hay subcarpetas en {root}", file=sys.stderr)
        return 1

    known_dirs = [d for d in subdirs if d.name.lower() not in UNKNOWN_DIR_NAMES]
    unknown_dirs = [d for d in subdirs if d.name.lower() in UNKNOWN_DIR_NAMES]

    print(f"Backend: {backend}")
    print(f"Carpeta: {root.resolve()}")
    print(f"Identidades conocidas: {[d.name for d in known_dirs]}")
    print(f"Carpeta(s) unknown: {[d.name for d in unknown_dirs]}")
    print(f"Split train/test: {args.split:.0%} / {1 - args.split:.0%}")
    print()

    try:
        _get_json(f"{backend}/health", timeout=5.0)
    except urlerror.URLError as exc:
        print(f"ERROR: backend no responde en {backend}/health: {exc}", file=sys.stderr)
        return 2

    train_set: dict[str, list[Path]] = {}
    test_set: dict[str, list[Path]] = {}

    for person_dir in known_dirs:
        imgs = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not imgs:
            print(f"  (sin imagenes en {person_dir.name}, salteado)")
            continue
        tr, te = split_train_test(imgs, args.split, rng)
        train_set[person_dir.name] = tr
        test_set[person_dir.name] = te

    print("== REGISTRO (train) ==")
    n_reg_ok = 0
    n_reg_fail = 0
    for identity, imgs in train_set.items():
        print(f"-- {identity} ({len(imgs)} imagenes) --")
        for img in imgs:
            try:
                server_path = upload_image(backend, img)
                job_id = insert_identity(backend, identity, server_path, img.name)
                status, detail, _ = wait_for_job(backend, job_id)
                if status == "done":
                    print(f"  [OK]   {img.name}")
                    n_reg_ok += 1
                else:
                    print(f"  [FAIL] {img.name}  {status}: {detail}")
                    n_reg_fail += 1
            except Exception as exc:
                print(f"  [FAIL] {img.name}  {exc}")
                n_reg_fail += 1

    print(f"\nRegistro: OK={n_reg_ok}  FAIL={n_reg_fail}")

    if args.no_test:
        return 0 if n_reg_fail == 0 else 3

    print("\n== TEST (held-out) ==")
    n_test = 0
    n_correct = 0
    errores: list[str] = []
    for identity, imgs in test_set.items():
        print(f"-- {identity} ({len(imgs)} imagenes) --")
        for img in imgs:
            try:
                server_path = upload_image(backend, img)
                job_id = request_predict(backend, server_path)
                status, _, artifact_url = wait_for_job(backend, job_id)
                if status != "done":
                    errores.append(f"{identity}/{img.name}: job {status}")
                    continue
                result = fetch_predict_result(backend, artifact_url)
                if result is None:
                    errores.append(f"{identity}/{img.name}: sin resultado JSON")
                    continue
                pred_label, pred_score = best_label_from_result(result)
                n_test += 1
                ok = pred_label == identity
                if ok:
                    n_correct += 1
                    print(f"  [OK]   {img.name}  -> {pred_label} ({pred_score:.3f})")
                else:
                    print(f"  [MISS] {img.name}  esperado={identity}, got={pred_label} ({pred_score:.3f})")
                    errores.append(f"{identity}/{img.name}: predijo {pred_label} ({pred_score:.3f})")
            except Exception as exc:
                errores.append(f"{identity}/{img.name}: {exc}")

    if n_test > 0:
        acc = n_correct / n_test
        print(f"\nAccuracy en test held-out: {n_correct}/{n_test} = {acc:.3f}")

    if unknown_dirs:
        print("\n== UNKNOWN (fuera del dataset) ==")
        n_unk = 0
        n_unk_ok = 0
        for d in unknown_dirs:
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            for img in imgs:
                try:
                    server_path = upload_image(backend, img)
                    job_id = request_predict(backend, server_path)
                    status, _, artifact_url = wait_for_job(backend, job_id)
                    if status != "done":
                        errores.append(f"unknown/{img.name}: job {status}")
                        continue
                    result = fetch_predict_result(backend, artifact_url)
                    if result is None:
                        errores.append(f"unknown/{img.name}: sin resultado JSON")
                        continue
                    pred_label, pred_score = best_label_from_result(result)
                    n_unk += 1
                    if pred_label == "unknown":
                        n_unk_ok += 1
                        print(f"  [OK]   {img.name}  -> unknown")
                    else:
                        print(f"  [MISS] {img.name}  esperado=unknown, got={pred_label} ({pred_score:.3f})")
                        errores.append(f"unknown/{img.name}: predijo {pred_label} ({pred_score:.3f})")
                except Exception as exc:
                    errores.append(f"unknown/{img.name}: {exc}")
        if n_unk > 0:
            print(f"\nAccuracy en unknown: {n_unk_ok}/{n_unk} = {n_unk_ok / n_unk:.3f}")

    if errores:
        print("\nErrores / mismatches:")
        for e in errores:
            print(f"  - {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
