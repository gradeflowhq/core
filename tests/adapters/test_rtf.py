import time

import pytest

from gradeflow_core.adapters.rtf import MAX_GROUP_DEPTH, ParagraphPart, paragraphs


@pytest.mark.parametrize(
    ("rtf", "expected"),
    [
        pytest.param(r"{\rtf1 One\par Two\par}", [["One"], ["Two"]], id="paragraphs"),
        pytest.param(
            r"{\rtf1 Caf\'e9 \u8722?1 \uc0\u8722 2}", [["Caf\xe9 \u22121 \u22122"]], id="unicode"
        ),
        pytest.param(r"{\rtf1 {\f3\u10003?}A. x\par}", [["✓A. x"]], id="fallback ends with group"),
        pytest.param(r"{\rtf1 \u-10179?\u-8704?\par}", [["😀"]], id="surrogates"),
        pytest.param(
            r"{\rtf1 {\fonttbl{\f0 Arial;}}{\headerr Page 1\par}{\*\bkmkstart x}Body\par}",
            [["Body"]],
            id="skipped",
        ),
        pytest.param(
            r"{\rtf1 \trowd A\cell B\cell\row \trowd C\cell\cell D\cell\row}",
            [["A | B"], ["C |  | D"]],
            id="table",
        ),
        pytest.param(
            r"{\rtf1 don\rquote t \ldblquote go\rdblquote\emdash now\par}",
            [["don\u2019t \u201cgo\u201d\u2014now"]],
            id="symbols",
        ),
        pytest.param(r"{\rtf1 {\pict\pngblip 89504e47}text\par}", [["text"]], id="hex picture"),
        pytest.param(r"{\rtf1 a\\b\{c\}\~d\par}", [["a\\b{c}\xa0d"]], id="escapes"),
    ],
)
def test_text_reads_as_written(rtf: str, expected: list[list[ParagraphPart]]) -> None:
    assert paragraphs(rtf) == expected


def test_binary_pictures_keep_their_bytes() -> None:
    assert paragraphs("{\\rtf1 x{\\*\\shppict{\\pict\\bin3 \x89}{}y\\par}") == [
        ["x", b"\x89}{", "y"]
    ]


def test_a_negative_binary_size_reads_nothing() -> None:
    assert paragraphs(r"{\rtf1 \bin-7 x}") == [["x"]]


def test_groups_nest_only_so_deep() -> None:
    with pytest.raises(ValueError, match="nested too deeply"):
        paragraphs("{" * (MAX_GROUP_DEPTH + 1))


def test_large_documents_read_in_linear_time() -> None:
    def seconds_to_read(size: int) -> float:
        rtf = "{\\rtf1 " + "\\'41\\cell " * size + "\\par}"
        start = time.perf_counter()
        paragraphs(rtf)
        return time.perf_counter() - start

    assert seconds_to_read(200_000) < 10 * seconds_to_read(20_000) + 0.5
