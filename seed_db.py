"""
seed_db.py — Carga inicial de identidades en PostgreSQL (pgvector).
Registra 2 imágenes por persona usando la API del backend.
Valentino queda EXCLUIDO intencionalmente.

Uso:
  docker compose run --rm --entrypoint python jupyter seed_db.py
"""

import os
import glob
import time
import requests

API_URL = "http://backend:8000"
DATASET_DIR = "Dataset"
IMAGES_PER_PERSON = 2   # cuántas imágenes registrar por persona


def wait_for_api(timeout=120):
    """Espera a que el backend responda."""
    print("Esperando a que la API esté disponible...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_URL}/docs", timeout=5)
            if r.status_code == 200:
                print("API disponible.")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def wait_for_job(job_id, timeout=60):
    """Espera a que un job asíncrono termine (done o failed)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_URL}/status/{job_id}", timeout=5)
            if r.status_code == 200:
                status = r.json().get("status")
                if status == "done":
                    return True
                if status == "failed":
                    reason = r.json().get("reason", "desconocido")
                    print(f"    ⚠ Job {job_id} falló: {reason}")
                    return False
        except Exception:
            pass
        time.sleep(1)
    print(f"    ⚠ Timeout esperando job {job_id}")
    return False


def seed_db():
    if not wait_for_api():
        print("La API no está respondiendo. Abortando seed.")
        return

    print(f"\nIniciando carga de base de datos ({IMAGES_PER_PERSON} imgs/persona)...\n")
    total_ok = 0

    for cls_name in sorted(os.listdir(DATASET_DIR)):
        cls_path = os.path.join(DATASET_DIR, cls_name)
        if not os.path.isdir(cls_path):
            continue

        if cls_name.lower() == "valentino":
            print(f"  ⏭ Saltando a {cls_name} (excluido intencionalmente)")
            continue

        files = sorted(glob.glob(os.path.join(cls_path, "*.*")))[:IMAGES_PER_PERSON]

        for file_path in files:
            fname = os.path.basename(file_path)
            print(f"  Registrando {cls_name} — {fname}...", end=" ")

            # 1) Subir archivo
            with open(file_path, "rb") as f:
                upload_res = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (fname, f, "image/jpeg")},
                )
            if upload_res.status_code != 200:
                print(f"ERROR al subir ({upload_res.status_code})")
                continue

            server_path = upload_res.json()["path"]

            # 2) Registrar identidad (la API devuelve 202 + job_id)
            payload = {
                "identity": cls_name.capitalize(),
                "image_path": server_path,
                "metadata": {"source": "seed_db.py"},
            }
            insert_res = requests.post(f"{API_URL}/insert", json=payload)

            if insert_res.status_code in (200, 202):
                job_id = insert_res.json().get("job_id", "")
                if job_id:
                    ok = wait_for_job(job_id)
                    if ok:
                        print("✓")
                        total_ok += 1
                    else:
                        print("✗")
                else:
                    print("✓ (sin job_id)")
                    total_ok += 1
            else:
                print(f"ERROR ({insert_res.status_code})")

    print(f"\nSeed finalizado: {total_ok} identidades registradas con éxito.")


if __name__ == "__main__":
    seed_db()
