"""API ligera: TestClient + mocks de FaceService para no cargar modelo pesado."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import lib.api as api_module
from lib.schemas import EmbeddingRecord


def test_health_ok(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "dummy.pth"


def test_status_unknown_job(client) -> None:
    r = client.get("/status/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_upload_returns_path_and_download_url(client) -> None:
    files = {"file": ("t.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "path" in data and "download_url" in data
    assert data["download_url"].startswith("/files/output/")


def test_register_accepts_and_completes_with_mock_face_service(client, monkeypatch) -> None:
    out_img = Path(api_module.settings.output_path) / "reg_out.jpg"
    out_img.write_bytes(b"fakejpg")

    rec = MagicMock(spec=EmbeddingRecord)
    rec.path = str(out_img.resolve())

    mock_fs = MagicMock()
    mock_fs.register_identity = MagicMock(return_value=rec)
    monkeypatch.setattr(api_module, "face_service", mock_fs)

    r = client.post(
        "/register",
        json={"identity": "test_id", "image_path": str(out_img), "metadata": {"src": "pytest"}},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert job_id

    for _ in range(50):
        st = client.get(f"/status/{job_id}")
        if st.json().get("status") == "done":
            break
        time.sleep(0.05)
    assert st.status_code == 200
    assert st.json()["status"] == "done"
    mock_fs.register_identity.assert_called_once()


def test_inference_accepts_with_mock_predict(client, monkeypatch) -> None:
    out_dir = Path(api_module.settings.output_path)
    result_json = out_dir / "pytest-result.json"
    result_json.write_text(
        json.dumps(
            {
                "source_path": "/tmp/in.jpg",
                "detections": [],
                "detected_people": [],
            }
        ),
        encoding="utf-8",
    )

    mock_fs = MagicMock()
    mock_fs.predict = MagicMock(return_value=str(result_json.resolve()))
    monkeypatch.setattr(api_module, "face_service", mock_fs)

    r = client.post("/inference", json={"source_path": "/tmp/in.jpg", "source_type": "image"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    for _ in range(50):
        st = client.get(f"/status/{job_id}")
        if st.json().get("status") == "done":
            break
        time.sleep(0.05)
    assert st.status_code == 200
    assert st.json()["status"] == "done"
    mock_fs.predict.assert_called_once()
