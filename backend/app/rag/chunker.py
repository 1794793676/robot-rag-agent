"""Sentence-aware character chunking with heading context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.parsers import ParsedSection


@dataclass(slots=True)
class TextChunk:
    chunk_index: int
    text: str
    page: int | None
    char_count: int


@dataclass(slots=True)
class _SemanticUnit:
    text: str
    separator: str = ""


_CLOSING_PUNCTUATION = r"""["'”’）)\]】}》〉」』]"""
_SENTENCE_BOUNDARY = re.compile(
    rf"[。！？!?；;]+{_CLOSING_PUNCTUATION}*"
    rf"|\.(?:{_CLOSING_PUNCTUATION})*(?=\s)"
)


def _sliding_windows(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - min(overlap, size - 1))
    return [
        text[start : start + size].strip()
        for start in range(0, len(text), step)
        if text[start : start + size].strip()
    ]


def _sentence_units(paragraph: str, first_separator: str) -> list[_SemanticUnit]:
    units: list[_SemanticUnit] = []
    start = 0
    separator = first_separator

    for boundary in _SENTENCE_BOUNDARY.finditer(paragraph):
        if boundary.start() > start and boundary.group().isspace():
            end = boundary.start()
            next_separator = boundary.group()
        else:
            end = boundary.end()
            whitespace = re.match(r"\s+", paragraph[end:])
            next_separator = whitespace.group() if whitespace else ""
            if whitespace:
                end += len(next_separator)

        sentence = paragraph[start:end].strip()
        if sentence:
            units.append(_SemanticUnit(sentence, separator))
            separator = next_separator
        start = end

    remainder = paragraph[start:].strip()
    if remainder:
        units.append(_SemanticUnit(remainder, separator))
    return units


def _semantic_units(text: str) -> list[_SemanticUnit]:
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()
    ]
    units: list[_SemanticUnit] = []
    for paragraph in paragraphs:
        units.extend(_sentence_units(paragraph, "\n\n" if units else ""))
    return units


def _join_units(units: list[_SemanticUnit]) -> str:
    if not units:
        return ""
    return units[0].text + "".join(
        f"{unit.separator}{unit.text}" for unit in units[1:]
    )


def _overlap_suffix(units: list[_SemanticUnit], overlap: int) -> list[_SemanticUnit]:
    suffix: list[_SemanticUnit] = []
    for unit in reversed(units):
        candidate = [unit, *suffix]
        if len(_join_units(candidate)) > overlap:
            break
        suffix = candidate
    return suffix


def _heading_prefix(heading: str | None, chunk_size: int) -> str:
    if not heading:
        return ""
    heading = heading.strip()
    if not heading:
        return ""
    if len(heading) + 1 < chunk_size:
        return f"{heading}\n"
    if chunk_size < 2:
        return ""
    return f"{heading[: chunk_size - 2]}\n"


def chunk_sections(
    sections: list[ParsedSection], chunk_size: int = 800, overlap: int = 120
) -> list[TextChunk]:
    """Split sections into bounded chunks while preserving page and heading."""

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 overlap，且 overlap 不能为负数")

    output: list[TextChunk] = []

    def append_chunk(body: str, prefix: str, page: int | None) -> None:
        text = f"{prefix}{body}"
        output.append(TextChunk(len(output), text, page, len(text)))

    for section in sections:
        prefix = _heading_prefix(section.heading, chunk_size)
        body_size = max(chunk_size - len(prefix), max(1, chunk_size // 2))
        current: list[_SemanticUnit] = []

        for unit in _semantic_units(section.text):
            if len(unit.text) > body_size:
                if current:
                    append_chunk(_join_units(current), prefix, section.page)
                    current = []
                for window in _sliding_windows(unit.text, body_size, overlap):
                    append_chunk(window, prefix, section.page)
                continue

            candidate = [*current, unit]
            if len(_join_units(candidate)) <= body_size:
                current = candidate
                continue

            append_chunk(_join_units(current), prefix, section.page)
            current = _overlap_suffix(current, min(overlap, body_size))
            while current and len(_join_units([*current, unit])) > body_size:
                current.pop(0)
            current.append(unit)

        if current:
            append_chunk(_join_units(current), prefix, section.page)

    return output
