import struct
import zlib

import pytest

from gradeflow_core import GradeFlowError, Question, QuestionSet, read_question_document
from gradeflow_core.adapters.rtf import MAX_GROUP_DEPTH
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import ListReading
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import BLANK_SEPARATOR, ValuesQuestion

HEADER = (
    r"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\froman Times New Roman;}{\f3 ZapfDingbats;}}"
    r"{\colortbl\red0\green0\blue0;}{\info{\title Midterm}}"
    r"\pard\plain\f1 Midterm - Confidential\par"
    r"\pard\plain\f1 # of Questions: 3\par"
    r"\pard\plain\f1 1.\tThis assessment contains 3 questions.\par"
)
SEPARATOR = (
    r"\pard\plain\f1 ____________________________________________________________\par\f0 \par "
)


def _formula_picture(mathml: str) -> str:
    """A formula as ExamSoft writes it: a PNG whose MathML text chunk holds it."""

    def chunk(chunk_type: bytes, content: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + content)
        return struct.pack(">I", len(content)) + chunk_type + content + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
        + chunk(b"tEXt", b"MathML\x00" + mathml.encode())
        + chunk(b"IEND", b"")
    )
    return (
        r"{\*\shppict{\pict\pngblip\picw33\pich28"
        + f"\n\\bin{len(png)} {png.decode('latin-1')}}}}}"
    )


def _formula(*tokens: str, inline: bool = False) -> str:
    """A formula in a paragraph of its own, as ExamSoft writes formulas in text, or ``inline``."""
    picture = _formula_picture(
        f'<math xmlns="http://www.w3.org/1998/Math/MathML">{"".join(tokens)}</math>'
    )
    return picture if inline else rf"\par\pard\ql{picture}\par\par" + "\n"


def _item(number: int, body: str, weight: str = "2.0") -> str:
    fields = "".join(
        rf"\pard\plain\b Item {name}: \b0 {value}\par"
        for name, value in (
            ("ID", f"{399000 + number} / 3"),
            ("Description", rf"{number}. [2 marks] Truncated\'85 [abc123]"),
            ("Group", "Part 1: Search"),
            ("Weight", weight),
            ("Creator", "author@example.edu"),
        )
    )
    psychometrics = (
        r"\pard\plain\b Item Psychometrics: \b0\par\par\trowd\cellx0"
        r"Difficulty\cell Upper 27%\cell\row 0.98\cell 99.07%\cell\row\pard"
    )
    return (
        rf"\pard\plain\b\cf2 Question #: \b0\cf0 {number}\par\f0 \par "
        + body
        + rf"\f0 \par {fields}{psychometrics}"
        + SEPARATOR
    )


def _choice(letter: str, text: str, correct: bool = False) -> str:
    tick = r"\f3 \u10003?\f1 " if correct else ""
    return rf"\pard\plain\fi-268\li720 {tick}{letter}. \f1 {text}\par"


def _document(*items: str) -> bytes:
    return (HEADER + SEPARATOR + "".join(items) + "}").encode("latin-1")


CHOICE_ITEM = _item(
    1,
    r"\pard\plain\b 1A. [2 marks]\b0  Which state "
    + _formula("<mo>(</mo><mi>r</mi><mo>,</mo><mi>k</mi><mo>)</mo>")
    + r"\f0 \par \pard\plain is reached first?\par\pard\plain\f0 \par \par"
    + _choice("A", "Caf\\'e9", correct=True)
    + _choice("B", "")
    + _formula("<msub><mi>h</mi><mn>1</mn></msub>")
    + _choice("C", "None of the above", correct=True),
)
BLANKS_ITEM = _item(
    2,
    r"\pard\plain 2. The cost of "
    + _formula(
        "<msup><mi>h</mi><mo>*</mo></msup><mo>(</mo><msub><mi>s</mi><mn>0</mn></msub><mo>)</mo>"
    )
    + r"is\par\pard\plain\ul \u160?\u160?1\u160?\u160?\ul0\par"
    + r"\pard\plain and its name is\par\pard\plain\ul \u160?\u160?2\u160?\u160?\ul0\par"
    + r"\pard\plain .\par\pard\plain\f0 \par \par"
    + r"\pard\plain\fi-244\li720 1. \ul 2|2.0|\ul0\par"
    + r"\pard\plain\fi-244\li720 2. \ul Terminal|terminal|\ul0\par",
    weight="3.0",
)
TABLE_ITEM = _item(
    3,
    r"\pard\plain 3A. Using the table\par\trowd\cellx1\cellx2 State\cell Cost\cell\row"
    r"\trowd\cellx1\cellx2 A\cell "
    + _formula("<mfrac><mn>1</mn><mn>2</mn></mfrac>", inline=True)
    + r"\cell\row\pard\plain, explain.\par",
    weight="4.5",
)
DOCUMENT = _document(CHOICE_ITEM, BLANKS_ITEM, TABLE_ITEM)
QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(max_marks=1),
        "Q2": TextQuestion(),
        "Q3": NumberQuestion(),
        "Q4": NumberQuestion(max_marks=5),
    }
)


def test_choices_make_a_choice_question_with_every_option() -> None:
    questions_read = read_question_document(DOCUMENT, QUESTIONS)

    assert questions_read["Q1"] == ChoiceQuestion(
        options=frozenset({"a", "b", "c"}),
        allow_multiple=True,
        max_marks=2,
        description="1A. [2 marks] Which state $(r,k)$ is reached first?\n\n"
        "A. Café\n\nB. $h_{1}$\n\nC. None of the above",
    )


@pytest.mark.parametrize(
    ("existing", "allow_multiple"),
    [
        (QUESTIONS["Q1"], False),
        (ChoiceQuestion(options=frozenset({"a", "e"}), allow_multiple=True), True),
    ],
)
def test_one_correct_choice_keeps_whether_several_are_allowed(
    existing: Question, allow_multiple: bool
) -> None:
    one_correct_item = _item(
        1, r"\pard\plain Pick\par" + _choice("A", "x", correct=True) + _choice("B", "y")
    )
    questions_read = read_question_document(
        _document(one_correct_item), QuestionSet({"Q1": existing})
    )
    assert questions_read["Q1"] == ChoiceQuestion(
        options=frozenset({"a", "b"}),
        allow_multiple=allow_multiple,
        max_marks=2,
        description="Pick\n\nA. x\n\nB. y",
    )


def test_blanks_make_a_question_of_their_values_and_hide_the_accepted_answers() -> None:
    questions_read = read_question_document(DOCUMENT, QUESTIONS)

    assert questions_read["Q2"] == ValuesQuestion(
        reading=ListReading(separator=BLANK_SEPARATOR),
        value_types=["number", "text"],
        max_marks=3,
        description=r"2\. The cost of $h^{*}(s_{0})$ is (1) and its name is (2).",
    )


@pytest.mark.parametrize(
    ("accepted_answers", "question_type"),
    [
        ("0.63|0.625|", NumberQuestion),
        ("\\u8722?1|-1|", NumberQuestion),
        ("2|2. from s0, transpose", TextQuestion),
        ("chlorophyll|Chlorophyll|", TextQuestion),
    ],
)
def test_one_blank_makes_a_number_question_when_every_accepted_answer_is_one(
    accepted_answers: str, question_type: type[NumberQuestion | TextQuestion]
) -> None:
    one_blank_item = _item(
        4,
        r"\pard\plain Which?\par\pard\plain\ul \u160?1\u160?\ul0\par"
        + rf"\pard\plain\fi-244\li720 1. {accepted_answers}\par",
    )

    question = read_question_document(_document(one_blank_item), QUESTIONS)["Q4"]

    assert question == question_type(max_marks=2, description="Which? (1)")


def test_without_choices_or_blanks_a_question_keeps_its_type() -> None:
    questions_read = read_question_document(DOCUMENT, QUESTIONS)

    assert questions_read["Q3"] == NumberQuestion(
        max_marks=4.5,
        description="3A. Using the table\n\nState | Cost\n\nA | $\\frac{1}{2}$\n\n, explain.",
    )


def test_a_stem_may_start_with_a_formula() -> None:
    formula_first_item = _item(
        3, _formula("<mi>x</mi>") + r"\pard\plain is what?\par", weight="7.0"
    )

    questions_read = read_question_document(_document(CHOICE_ITEM, formula_first_item), QUESTIONS)

    assert (questions_read["Q1"].max_marks, questions_read["Q3"].max_marks) == (2, 7)
    assert questions_read["Q3"].description == "$x$ is what?"


def test_a_stem_with_lettered_parts_is_still_a_blank() -> None:
    lettered_parts_item = _item(
        3,
        r"\pard\plain Two parts:\par\pard\plain A. The cost is\par"
        r"\pard\plain\ul \u160?1\u160?\ul0\par\pard\plain\fi-244\li720 1. 2|\par",
    )

    questions_read = read_question_document(_document(lettered_parts_item), QUESTIONS)

    assert questions_read["Q3"] == NumberQuestion(
        max_marks=2, description="Two parts:\n\nA. The cost is (1)"
    )


def test_text_that_reads_as_markdown_is_escaped() -> None:
    markdown_item = _item(
        3,
        r"\pard\plain # 1 - not *bold* or `code`, [a](b) costs $5 & &amp; more, \\ \par"
        r"\pard\plain - a list?\par\pard\plain 10) or this\par"
        r"\pard\plain [a [b] c](d), <https://x.y> but a < b > c\par",
    )

    questions_read = read_question_document(_document(markdown_item), QUESTIONS)

    assert questions_read["Q3"].description == (
        r"\# 1 - not \*bold\* or \`code\`, \[a](b) costs \$5 & \&amp; more, \\"
        "\n\n"
        r"\- a list?"
        "\n\n"
        r"10\) or this"
        "\n\n"
        r"\[a \[b] c](d), \<https://x.y> but a < b > c"
    )


def test_a_weight_that_isnt_a_number_is_left_out() -> None:
    questions_read = read_question_document(
        _document(_item(4, r"\pard\plain How many?\par", "N/A")), QUESTIONS
    )

    assert questions_read["Q4"].max_marks == 5


def test_only_questions_in_both_are_described() -> None:
    questions_read = read_question_document(
        _document(CHOICE_ITEM, _item(9, r"\pard\plain Unasked\par")), QUESTIONS
    )

    assert list(questions_read) == ["Q1"]


@pytest.mark.parametrize(
    "file",
    [b"StudentID,Q1\n", _document(), b"{\\rtf1 {unclosed"],
    ids=["csv", "no questions", "unclosed"],
)
def test_other_files_are_refused(file: bytes) -> None:
    with pytest.raises(
        GradeFlowError,
        match=r"^Couldn't read the file: it isn't an ExamSoft question document\.$",
    ):
        read_question_document(file, QUESTIONS)


def test_a_document_that_cant_be_read_says_why() -> None:
    with pytest.raises(
        GradeFlowError, match=r"^Couldn't read the file: its groups are nested too deeply\.$"
    ):
        read_question_document(b"{\\rtf1 " + b"{" * MAX_GROUP_DEPTH, QUESTIONS)
