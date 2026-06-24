"""Unified DashScope and deterministic development embedding interface."""

from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

from app.core.config import Settings


class EmbeddingError(RuntimeError):
    """Raised when the remote embedding service cannot complete a request."""


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _fake_embedding(self, text: str) -> list[float]:
        """Generate stable lexical feature vectors for offline development."""

        normalized = re.sub(r"\s+", "", text.lower())
        features = list(normalized)
        features += [normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))]
        features += re.findall(r"[a-z0-9_]+", text.lower())
        vector = np.zeros(self.settings.embedding_dim, dtype=np.float32)
        for feature in features or [text]:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.settings.embedding_dim
            vector[index] += 1.0 if value & 1 else -1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.dashscope_api_key:
            return [self._fake_embedding(text) for text in texts]

        results: list[list[float]] = []
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        batch_size = max(1, self.settings.embedding_batch_size)
        try:
            with httpx.Client(timeout=60) as client:
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    payload = {
                        "model": self.settings.embedding_model,
                        "input": batch,
                        "dimensions": self.settings.embedding_dim,
                        "encoding_format": "float",
                    }
                    response = client.post(
                        self.settings.dashscope_base_url, headers=headers, json=payload
                    )
                    response.raise_for_status()
                    data = sorted(response.json()["data"], key=lambda item: item["index"])
                    results.extend(item["embedding"] for item in data)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"DashScope embedding 调用失败：{exc}") from exc
        return results

