import pytest

from app.rag.chunker import chunk_sections
from app.rag.parsers import ParsedSection


def test_heading_is_prefixed_and_chinese_sentences_stay_whole():
    sections = [
        ParsedSection(
            text="操作前关闭电源。确认设备停止。佩戴绝缘手套。",
            page=3,
            heading="安全规范",
        )
    ]

    chunks = chunk_sections(sections, chunk_size=13, overlap=0)

    assert [chunk.text for chunk in chunks] == [
        "安全规范\n操作前关闭电源。",
        "安全规范\n确认设备停止。",
        "安全规范\n佩戴绝缘手套。",
    ]
    assert all(chunk.text.removeprefix("安全规范\n").endswith("。") for chunk in chunks)
    assert [chunk.page for chunk in chunks] == [3, 3, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.char_count for chunk in chunks] == [
        len(chunk.text) for chunk in chunks
    ]


def test_overlong_single_sentence_falls_back_to_bounded_character_windows():
    sentence = "这" * 50 + "。"

    chunks = chunk_sections(
        [ParsedSection(text=sentence)], chunk_size=20, overlap=5
    )

    assert len(chunks) > 1
    assert all(0 < chunk.char_count <= 20 for chunk in chunks)
    assert chunks[0].text == sentence[:20]
    assert chunks[1].text == sentence[15:35]


def test_overlap_reuses_only_complete_trailing_semantic_units():
    chunks = chunk_sections(
        [ParsedSection(text="甲句完整。乙句完整。丙句完整。")],
        chunk_size=10,
        overlap=5,
    )

    assert [chunk.text for chunk in chunks] == [
        "甲句完整。乙句完整。",
        "乙句完整。丙句完整。",
    ]


def test_english_period_followed_by_newline_is_a_sentence_boundary():
    chunks = chunk_sections(
        [ParsedSection(text="First sentence.\nSecond sentence.")],
        chunk_size=20,
        overlap=0,
    )

    assert [chunk.text for chunk in chunks] == [
        "First sentence.",
        "Second sentence.",
    ]


@pytest.mark.parametrize(
    ("text", "chunk_size", "expected"),
    [
        (
            "他说：“你好！” 我走了。",
            10,
            ["他说：“你好！”", "我走了。"],
        ),
        (
            '"Hello." Next sentence.',
            16,
            ['"Hello."', "Next sentence."],
        ),
        (
            "注意（立即停止！） 然后撤离。",
            12,
            ["注意（立即停止！）", "然后撤离。"],
        ),
    ],
)
def test_sentence_boundary_keeps_closing_quote_or_parenthesis(text, chunk_size, expected):
    chunks = chunk_sections(
        [ParsedSection(text=text)],
        chunk_size=chunk_size,
        overlap=0,
    )

    assert [chunk.text for chunk in chunks] == expected


def test_simple_unheaded_paragraphs_keep_existing_joining_behavior():
    chunks = chunk_sections(
        [ParsedSection(text="first paragraph\n\nsecond paragraph", page=7)],
        chunk_size=50,
        overlap=5,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "first paragraph\n\nsecond paragraph"
    assert chunks[0].page == 7
    assert chunks[0].chunk_index == 0
    assert chunks[0].char_count == len(chunks[0].text)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_parameters_raise_value_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_sections([], chunk_size=chunk_size, overlap=overlap)


def test_long_heading_keeps_body_whole_even_when_chunk_exceeds_chunk_size():
    chunks = chunk_sections(
        [ParsedSection(text="正文。", heading="这是一个远远超过块大小的标题")],
        chunk_size=8,
        overlap=0,
    )

    assert chunks
    assert len(chunks) == 1
    assert chunks[0].text.endswith("\n正文。")
    assert chunks[0].text.split("\n", 1)[1] == "正文。"
    assert chunks[0].char_count > 8
