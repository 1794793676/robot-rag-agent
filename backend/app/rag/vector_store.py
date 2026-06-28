"""Rebuildable local vector index with optional hnswlib acceleration."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent optional path
    hnswlib = None

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorRecord:
    chunk_id: str
    doc_id: str
    vector: list[float] | np.ndarray


class VectorStore:
    """In-memory search state persisted as an optional HNSW cache."""

    def __init__(self, dimension: int, index_dir: Path):
        self.dimension = dimension
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: list[VectorRecord] = []
        self._matrix = np.empty((0, dimension), dtype=np.float32)
        self._hnsw = None

    @property
    def backend_name(self) -> str:
        return "hnswlib" if self._hnsw is not None else "numpy"

    @property
    def record_count(self) -> int:
        return len(self._records)

    def add_vectors(self, records: list[VectorRecord]) -> None:
        self.rebuild([*self._records, *records])

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.rebuild([record for record in self._records if record.doc_id != doc_id])

    def rebuild(self, records: list[VectorRecord]) -> None:
        """Replace all search state; callers source records from SQLite."""

        with self._lock:
            valid: list[VectorRecord] = []
            vectors: list[np.ndarray] = []
            for record in records:
                vector = np.asarray(record.vector, dtype=np.float32)
                if vector.shape != (self.dimension,):
                    logger.warning("Skipping vector %s with invalid dimension", record.chunk_id)
                    continue
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
                valid.append(VectorRecord(record.chunk_id, record.doc_id, vector))
                vectors.append(vector)
            self._records = valid
            self._matrix = (
                np.vstack(vectors)
                if vectors
                else np.empty((0, self.dimension), dtype=np.float32)
            )
            self._hnsw = None
            if hnswlib is not None and valid:
                try:
                    index = hnswlib.Index(space="cosine", dim=self.dimension)
                    index.init_index(max_elements=len(valid), ef_construction=100, M=16)
                    index.add_items(self._matrix, np.arange(len(valid)))
                    index.set_ef(min(max(20, len(valid)), 100))
                    self._hnsw = index
                except Exception:
                    logger.exception("HNSW initialization failed; using NumPy fallback")
            self.persist()

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._records or top_k <= 0:
                return []
            query = np.asarray(query_vector, dtype=np.float32)
            if query.shape != (self.dimension,):
                raise ValueError("查询向量维度与索引不一致")
            norm = np.linalg.norm(query)
            if norm:
                query = query / norm
            if allowed_chunk_ids is not None:
                allowed_indices = np.fromiter(
                    (
                        index
                        for index, record in enumerate(self._records)
                        if record.chunk_id in allowed_chunk_ids
                    ),
                    dtype=np.intp,
                )
                if not len(allowed_indices):
                    return []
                scoped_scores = self._matrix[allowed_indices] @ query
                count = min(top_k, len(allowed_indices))
                scoped_order = np.argsort(-scoped_scores)[:count]
                return [
                    (
                        self._records[int(allowed_indices[index])].chunk_id,
                        float(scoped_scores[index]),
                    )
                    for index in scoped_order
                ]

            count = min(top_k, len(self._records))
            if self._hnsw is not None:
                labels, distances = self._hnsw.knn_query(query, k=count)
                return [
                    (self._records[int(label)].chunk_id, float(1.0 - distance))
                    for label, distance in zip(labels[0], distances[0])
                ]
            scores = self._matrix @ query
            indices = np.argsort(-scores)[:count]
            return [
                (self._records[int(index)].chunk_id, float(scores[index]))
                for index in indices
            ]

    def persist(self) -> None:
        """Persist HNSW and mapping caches; both may be safely rebuilt."""

        metadata_path = self.index_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                [
                    {"chunk_id": record.chunk_id, "doc_id": record.doc_id}
                    for record in self._records
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        hnsw_path = self.index_dir / "vectors.hnsw"
        if self._hnsw is not None:
            self._hnsw.save_index(str(hnsw_path))
        elif hnsw_path.exists():
            hnsw_path.unlink()

    def load(self, records: list[VectorRecord]) -> None:
        """Load safely by rebuilding from canonical SQLite records."""

        try:
            self.rebuild(records)
        except Exception:
            logger.exception("Vector cache was invalid; rebuilding from SQLite")
            for path in self.index_dir.glob("*"):
                if path.is_file():
                    path.unlink()
            self.rebuild(records)
