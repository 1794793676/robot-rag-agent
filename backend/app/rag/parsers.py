"""Text extraction for txt, docx, and text-layer PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


class DocumentParseError(ValueError):
    """Raised when a supported file cannot provide usable text."""


@dataclass(slots=True)
class ParsedSection:
    text: str
    page: int | None = None
    heading: str | None = None


def _parse_txt(path: Path) -> list[ParsedSection]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentParseError("无法识别 TXT 文件编码，请使用 UTF-8、GBK 或 GB18030")
    if not text.strip():
        raise DocumentParseError("文档没有可用文本")
    return [ParsedSection(text=text.strip())]


def _table_to_markdown(table) -> str:
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header, separator, *body])


_CHINESE_NUMBERED_HEADING = re.compile(
    r"^(?:[一二三四五六七八九十百]+[、．.]|（[一二三四五六七八九十百]+）)\s*\S+$"
)
_WORD_HEADING_STYLE = re.compile(r"^(?:heading|标题)\s*(\d+)$")


def _heading_level(paragraph: Paragraph) -> int | None:
    style_name = (paragraph.style.name if paragraph.style else "").strip().lower()
    match = _WORD_HEADING_STYLE.match(style_name)
    if not match:
        return None
    return int(match.group(1))


def _is_heading(paragraph: Paragraph, text: str) -> bool:
    if _heading_level(paragraph) is not None:
        return True
    return (
        len(text) <= 20
        and text[-1] not in "。！？!?；;"
        and bool(_CHINESE_NUMBERED_HEADING.fullmatch(text))
    )


def _parse_docx(path: Path) -> list[ParsedSection]:
    try:
        document = DocxDocument(path)
    except Exception as exc:
        raise DocumentParseError(f"DOCX 解析失败：{exc}") from exc

    sections: list[ParsedSection] = []
    heading_stack: list[str] = []
    current_heading: str | None = None
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            if _is_heading(paragraph, text):
                level = _heading_level(paragraph)
                if level is None:
                    heading_stack = [text]
                    current_heading = text
                else:
                    heading_stack = heading_stack[: max(0, level - 1)]
                    heading_stack.append(text)
                    current_heading = " / ".join(heading_stack)
                continue
        elif isinstance(child, CT_Tbl):
            text = _table_to_markdown(Table(child, document))
            if not text:
                continue
        else:
            continue
        sections.append(ParsedSection(text=text, heading=current_heading))

    if not sections:
        raise DocumentParseError("DOCX 文档没有可用文本")
    return sections


def _parse_pdf(path: Path) -> list[ParsedSection]:
    try:
        reader = PdfReader(str(path))
        sections = [
            ParsedSection(text=text.strip(), page=index)
            for index, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "")).strip()
        ]
    except Exception as exc:
        raise DocumentParseError(f"PDF 解析失败：{exc}") from exc
    total_chars = sum(len(section.text) for section in sections)
    if total_chars < 20:
        raise DocumentParseError(
            "当前版本不支持 OCR，请上传可复制文本的 PDF、docx 或 txt 文件。"
        )
    return sections


def parse_document(path: Path, file_type: str) -> list[ParsedSection]:
    """Parse a document into page-aware text sections."""

    parsers = {"txt": _parse_txt, "docx": _parse_docx, "pdf": _parse_pdf}
    try:
        parser = parsers[file_type]
    except KeyError as exc:
        raise DocumentParseError("仅支持 txt、docx、pdf 文件") from exc
    return parser(path)
