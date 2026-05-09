"""Contratos Pydantic: serialización / deserialización JSON."""
from __future__ import annotations

import json

from lib.schemas import (
    AsyncTaskCreated,
    EmbeddingRecord,
    InsertRequest,
    PredictRequest,
    StatusResponse,
    UploadResponse,
)


def test_insert_request_roundtrip() -> None:
    obj = InsertRequest(identity="Ana", image_path="/tmp/x.jpg", metadata={"k": 1})
    raw = obj.model_dump()
    back = InsertRequest.model_validate(raw)
    assert back == obj
    assert json.loads(json.dumps(raw))["identity"] == "Ana"


def test_predict_request_defaults() -> None:
    obj = PredictRequest(source_path="/a/b.jpg")
    assert obj.source_type == "image"
    d = json.loads(obj.model_dump_json())
    assert d["source_type"] == "image"


def test_async_task_created() -> None:
    obj = AsyncTaskCreated(job_id="abc-123")
    assert obj.status == "accepted"
    StatusResponse.model_validate(
        {"status": "inProgress", "link": "none", "reason": None, "artifact_url": None, "source_image_url": None}
    )


def test_status_response_done_with_urls() -> None:
    obj = StatusResponse(
        status="done",
        link="/tmp/result.json",
        reason=None,
        artifact_url="/files/output/result.json",
        source_image_url="/files/data/x.jpg",
    )
    payload = json.loads(obj.model_dump_json())
    restored = StatusResponse.model_validate(payload)
    assert restored.status == "done"
    assert restored.artifact_url is not None


def test_upload_response() -> None:
    obj = UploadResponse(path="/abs/file.jpg", download_url="/files/output/uploads/x.jpg")
    assert "path" in obj.model_dump()


def test_embedding_record() -> None:
    obj = EmbeddingRecord(
        id_imagen="id1",
        embedding=[0.1, 0.2],
        path="/p.jpg",
        etiqueta="004",
        metadata={"src": "test"},
    )
    EmbeddingRecord.model_validate(json.loads(obj.model_dump_json()))
