"""Shared serialization boundary for untrusted retrieved evidence."""

from __future__ import annotations

from html import escape


EVIDENCE_CHAR_LIMIT = 12_000
EVIDENCE_METADATA_CHAR_LIMIT = 500
_OPEN_SET = "<evidence_set>\n"
_CLOSE_SET = "\n</evidence_set>"


def escape_metadata(value: object, default: str = "") -> str:
    normalized = str(value) if value is not None else default
    return escape(normalized, quote=True)[:EVIDENCE_METADATA_CHAR_LIMIT]


def serialize_evidence(results: list[dict]) -> str:
    """Return deterministic XML-like data with a hard total-size envelope."""

    blocks: list[str] = []
    used = len(_OPEN_SET) + len(_CLOSE_SET)
    for index, result in enumerate(results, start=1):
        raw_text = str(result.get("text") or "")
        if not raw_text:
            continue

        filename = escape_metadata(
            result.get("filename") or result.get("source"), "未知"
        )
        page = escape_metadata(result.get("page"), "未知")
        text = escape(raw_text, quote=True)
        prefix = "\n".join(
            [
                f'<evidence index="{index}">',
                f"<filename>{filename}</filename>",
                f"<page>{page}</page>",
                "<text>",
            ]
        )
        suffix = "</text>\n</evidence>"
        separator = "\n\n" if blocks else ""
        available_text = (
            EVIDENCE_CHAR_LIMIT
            - used
            - len(separator)
            - len(prefix)
            - len(suffix)
        )
        if available_text <= 0:
            break

        block = f"{prefix}{text[:available_text]}{suffix}"
        blocks.append(f"{separator}{block}")
        used += len(separator) + len(block)
        if len(text) > available_text:
            break

    inner = "".join(blocks) or "无"
    serialized = f"{_OPEN_SET}{inner}{_CLOSE_SET}"
    assert len(serialized) <= EVIDENCE_CHAR_LIMIT
    return serialized
