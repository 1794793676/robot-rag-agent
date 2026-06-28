"""Transactional document ingestion, replacement, and deletion."""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi import UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Chunk, Document, RagDatabase, utc_now
from app.rag.chunker import chunk_sections
from app.rag.embedder import Embedder
from app.rag.parsers import parse_document
from app.rag.vector_store import VectorRecord, VectorStore


ALLOWED_EXTENSIONS = {"txt", "docx", "xls", "xlsx", "pdf"}


class DocumentService:
    def __init__(
        self, settings: Settings, embedder: Embedder, vector_store: VectorStore
    ):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename).name
        return re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]", "_", name)[:240] or "document"

    async def _read_upload(self, upload: UploadFile) -> tuple[str, str, bytes, str]:
        filename = self._safe_filename(upload.filename or "")
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("仅支持 txt、docx、xls、xlsx、pdf 文件")
        content = await upload.read()
        if not content:
            raise ValueError("上传文件为空")
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"文件不能超过 {self.settings.max_upload_mb} MB")
        return filename, extension, content, hashlib.sha256(content).hexdigest()

    def _prepare_chunks(
        self, temp_path: Path, extension: str
    ) -> list[tuple[object, list[float]]]:
        sections = parse_document(temp_path, extension)
        chunks = chunk_sections(
            sections,
            self.settings.chunk_size_chars,
            self.settings.chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError("文档解析后没有可用 chunk")
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding 返回数量与 chunk 数量不一致")
        return list(zip(chunks, vectors))

    def _rebuild_index(self, session: Session) -> None:
        rows = session.execute(select(Chunk)).scalars().all()
        records = [
            VectorRecord(
                chunk_id=chunk.id,
                doc_id=chunk.document_id,
                vector=np.frombuffer(chunk.embedding, dtype=np.float32).copy(),
            )
            for chunk in rows
        ]
        self.vector_store.load(records)

    async def create(
        self, session: Session, upload: UploadFile, rag_database: RagDatabase
    ) -> tuple[Document, bool]:
        filename, extension, content, digest = await self._read_upload(upload)
        existing = session.scalar(
            select(Document).where(
                Document.file_hash == digest,
                Document.rag_database_id == rag_database.id,
            )
        )
        if existing:
            return existing, True

        doc_id = str(uuid.uuid4())
        temp_path = self.settings.files_dir / f".{doc_id}.upload.{extension}"
        final_name = f"{doc_id}.{extension}"
        final_path = self.settings.files_dir / final_name
        temp_path.write_bytes(content)
        try:
            prepared = self._prepare_chunks(temp_path, extension)
            document = Document(
                id=doc_id,
                rag_database_id=rag_database.id,
                filename=filename,
                stored_filename=final_name,
                file_type=extension,
                file_size=len(content),
                file_hash=digest,
                status="ready",
                parse_message="解析和索引完成",
            )
            session.add(document)
            for index, (chunk, vector) in enumerate(prepared):
                session.add(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        chunk_index=index,
                        text=chunk.text,
                        page=chunk.page,
                        char_count=chunk.char_count,
                        embedding=np.asarray(vector, dtype=np.float32).tobytes(),
                    )
                )
            session.commit()
            temp_path.replace(final_path)
            self._rebuild_index(session)
            return document, False
        except Exception:
            session.rollback()
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    async def replace(
        self, session: Session, doc_id: str, upload: UploadFile, rag_database: RagDatabase
    ) -> Document:
        document = session.scalar(
            select(Document).where(
                Document.id == doc_id,
                Document.rag_database_id == rag_database.id,
            )
        )
        if not document:
            raise LookupError("文档不存在")
        filename, extension, content, digest = await self._read_upload(upload)
        conflict = session.scalar(
            select(Document).where(Document.file_hash == digest, Document.id != doc_id)
            .where(Document.rag_database_id == rag_database.id)
        )
        if conflict:
            raise FileExistsError("相同内容已作为其他文档存在")

        temp_path = self.settings.files_dir / f".{doc_id}.replacement.{extension}"
        new_name = f"{doc_id}.{extension}"
        final_path = self.settings.files_dir / new_name
        old_path = self.settings.files_dir / document.stored_filename
        temp_path.write_bytes(content)
        try:
            prepared = self._prepare_chunks(temp_path, extension)
            session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            document.filename = filename
            document.stored_filename = new_name
            document.file_type = extension
            document.file_size = len(content)
            document.file_hash = digest
            document.status = "ready"
            document.parse_message = "替换、解析和索引完成"
            document.updated_at = utc_now()
            for index, (chunk, vector) in enumerate(prepared):
                session.add(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        chunk_index=index,
                        text=chunk.text,
                        page=chunk.page,
                        char_count=chunk.char_count,
                        embedding=np.asarray(vector, dtype=np.float32).tobytes(),
                    )
                )
            session.commit()
            if old_path != final_path:
                old_path.unlink(missing_ok=True)
            temp_path.replace(final_path)
            self._rebuild_index(session)
            return document
        except Exception:
            session.rollback()
            temp_path.unlink(missing_ok=True)
            raise

    def delete(self, session: Session, doc_id: str, rag_database: RagDatabase) -> None:
        document = session.scalar(
            select(Document).where(
                Document.id == doc_id,
                Document.rag_database_id == rag_database.id,
            )
        )
        if not document:
            raise LookupError("文档不存在")
        path = self.settings.files_dir / document.stored_filename
        session.delete(document)
        session.commit()
        path.unlink(missing_ok=True)
        self._rebuild_index(session)

    def delete_database(self, session: Session, rag_database: RagDatabase) -> None:
        if rag_database.is_default:
            raise ValueError("默认 RAG 数据库不能删除")

        documents = session.scalars(
            select(Document).where(Document.rag_database_id == rag_database.id)
        ).all()
        doc_ids = [document.id for document in documents]
        paths = [self.settings.files_dir / document.stored_filename for document in documents]
        if doc_ids:
            session.execute(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
            session.execute(delete(Document).where(Document.id.in_(doc_ids)))
        session.delete(rag_database)
        session.commit()

        for path in paths:
            path.unlink(missing_ok=True)
        self._rebuild_index(session)

    @staticmethod
    def to_dict(session: Session, document: Document, detail: bool = False) -> dict:
        chunk_count = session.scalar(
            select(func.count(Chunk.id)).where(Chunk.document_id == document.id)
        )
        payload = {
            "doc_id": document.id,
            "rag_database_id": document.rag_database_id,
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "chunk_count": chunk_count or 0,
            "status": document.status,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
        }
        if detail:
            payload.update(
                file_hash=document.file_hash,
                parse_message=document.parse_message,
            )
        return payload
