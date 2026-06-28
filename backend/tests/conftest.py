"""Shared fixtures configure an isolated storage directory before app import."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("RAG_STORAGE_DIR", str(storage))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_DIM", "64")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.15")
    monkeypatch.setenv("RERANK_ENABLED", "false")

    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

    main = importlib.import_module("app.main")
    with TestClient(main.app) as test_client:
        yield test_client
