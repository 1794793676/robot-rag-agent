from __future__ import annotations

import numpy as np

from app.rag.vector_store import VectorRecord, VectorStore


def test_search_applies_allowed_chunk_ids_before_top_k(tmp_path):
    store = VectorStore(2, tmp_path)
    records = [
        VectorRecord(f"other-{index}", "doc-other", [1.0, 0.0])
        for index in range(100)
    ]
    records.extend(
        [
            VectorRecord("allowed-low", "doc-a", [0.6, 0.8]),
            VectorRecord("allowed-high", "doc-a", [0.8, 0.6]),
        ]
    )
    store.rebuild(records)

    hits = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
        allowed_chunk_ids={"allowed-low", "allowed-high"},
    )

    assert [chunk_id for chunk_id, _ in hits] == [
        "allowed-high",
        "allowed-low",
    ]
    assert len(hits) == 2


def test_search_returns_only_available_allowed_records(tmp_path):
    store = VectorStore(2, tmp_path)
    store.rebuild(
        [
            VectorRecord("a", "doc-a", [1.0, 0.0]),
            VectorRecord("b", "doc-b", [0.0, 1.0]),
        ]
    )

    assert store.search([1.0, 0.0], 20, allowed_chunk_ids={"b", "missing"}) == [
        ("b", 0.0)
    ]
