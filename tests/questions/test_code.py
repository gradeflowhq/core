import pytest

from gradeflow_core.editing import HINT_KEY
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.inference import infer_question

UNINDENTED_CODE = "def first_positive(xs):\nfor x in xs:\nif x > 0:\nreturn x\nreturn None"
INDENTED_CODE = (
    "def first_positive(xs):\n"
    "    for x in xs:\n"
    "        if x > 0:\n"
    "            return x\n"
    "    return None"
)


def test_code_reads_as_typed_without_surrounding_blank_lines_or_shared_indentation() -> None:
    assert CodeQuestion().read("\r\n    x = 1  \r\n    if x:\r\n        y = X\n\n") == (
        "x = 1\nif x:\n    y = X"
    )


@pytest.mark.parametrize("answer", ["", "  \n", "N/A"])
def test_blank_code_reads_as_blank(answer: str) -> None:
    assert CodeQuestion().read(answer) is None


def test_code_is_written_as_typed() -> None:
    assert CodeQuestion().normalize(UNINDENTED_CODE) == UNINDENTED_CODE


def test_code_says_its_answers_are_code() -> None:
    assert CodeQuestion().answer_schema() == {
        "type": "string",
        HINT_KEY: {"kind": "code", "language": "python"},
    }


def test_custom_code_is_told_how_answers_reach_it() -> None:
    assert CodeQuestion().python_type() == "str | None"


def test_answers_that_are_programs_make_a_code_question() -> None:
    assert isinstance(
        infer_question([UNINDENTED_CODE, INDENTED_CODE, "def f():\nreturn 1", ""]), CodeQuestion
    )


@pytest.mark.parametrize(
    "answers",
    [
        [
            "photosynthesis",
            "respiration",
            "osmosis",
            "diffusion",
            "mitosis",
            "meiosis",
            "enzymes",
            "x",
        ],
        ["Light is absorbed.\nWater is split.", "It makes sugar.\nIt needs light."],
    ],
)
def test_words_and_prose_are_not_code(answers: list[str]) -> None:
    assert not isinstance(infer_question(answers), CodeQuestion)
