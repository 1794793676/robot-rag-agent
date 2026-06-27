"""Environment-driven application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from the project .env and process environment."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Local RAG Document Knowledge Base"
    rag_storage_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "storage"
    )
    dashscope_api_key: str = ""
    dashscope_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    )
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 512
    embedding_batch_size: int = 10
    chat_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    chat_model: str = "qwen3.6-flash"
    chat_max_tokens: int = 800
    chat_temperature: float = 0.2
    qwen_realtime_model: str = "qwen3.5-omni-flash-realtime"
    qwen_realtime_region: str = "singapore"
    qwen_realtime_url: str = ""
    qwen_realtime_workspace_id: str = ""
    qwen_realtime_voice: str = "Tina"
    rag_base_url: str = "http://127.0.0.1:8000"
    web_search_provider: str = "tavily"
    session_ttl_seconds: int = 1800
    log_level: str = "INFO"
    chunk_size_chars: int = 800
    chunk_overlap_chars: int = 120
    similarity_threshold: float = 0.35
    rerank_enabled: bool = True
    rerank_model: str = "qwen3-rerank"
    rerank_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks"
    )
    rerank_candidate_k: int = Field(default=30, ge=20, le=50)
    rerank_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    rerank_timeout_seconds: float = Field(default=2.0, gt=0.0, le=10.0)
    max_upload_mb: int = 30
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def rerank_is_enabled(self) -> bool:
        return self.rerank_enabled and bool(self.dashscope_api_key)

    @property
    def files_dir(self) -> Path:
        return self.rag_storage_dir / "files"

    @property
    def index_dir(self) -> Path:
        return self.rag_storage_dir / "index"

    @property
    def database_path(self) -> Path:
        return self.rag_storage_dir / "rag.db"

    def ensure_directories(self) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings object."""

    return Settings()
