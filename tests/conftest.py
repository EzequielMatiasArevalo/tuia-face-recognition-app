"""
Configuración de tests: variables de entorno y artefactos mínimos ANTES de importar la app.

Así /health puede pasar sin modelo real y FaceService carga un .pth mínimo vía torch.load.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRATCH = _ROOT / ".pytest_ci"
_SCRATCH.mkdir(exist_ok=True)
(_SCRATCH / "models").mkdir(exist_ok=True)
(_SCRATCH / "output" / "uploads").mkdir(parents=True, exist_ok=True)
(_SCRATCH / "data").mkdir(exist_ok=True)

_EMB = _SCRATCH / "embeddings.json"
_EMB.write_text("[]", encoding="utf-8")

_DUMMY_PTH = _SCRATCH / "models" / "dummy.pth"
if not _DUMMY_PTH.exists():
    import torch

    torch.save({"_pytest_dummy": True}, _DUMMY_PTH)

os.environ.setdefault("USE_PGVECTOR", "false")
os.environ.setdefault("EMBEDDINGS_PATH", str(_EMB))
os.environ.setdefault("DATA_PATH", str(_SCRATCH / "data"))
os.environ.setdefault("OUTPUT_PATH", str(_SCRATCH / "output"))
os.environ.setdefault("MODEL_PATH", str(_SCRATCH / "models"))
os.environ.setdefault("MODEL_NAME", "dummy.pth")
os.environ.setdefault("SIMILARITY_THRESHOLD", "0.55")
os.environ.setdefault("CORS_ORIGINS", "*")


@pytest.fixture
def client():
    """TestClient sobre app.main (import diferido para respetar os.environ)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
