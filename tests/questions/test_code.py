import pytest

from gradeflow_core.editing import HINT_KEY
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.inference import infer_question
from gradeflow_core.questions.reading import CodeReading

GUESSING = CodeQuestion(reading=CodeReading(guess_indentation=True))
FLAT = "def first_positive(xs):\nfor x in xs:\nif x > 0:\nreturn x\nreturn None"
LAID_OUT = (
    "def first_positive(xs):\n"
    "    for x in xs:\n"
    "        if x > 0:\n"
    "            return x\n"
    "    return None"
)


def test_code_reads_as_typed_without_its_edges_or_shared_indentation() -> None:
    assert CodeQuestion().read("\r\n    x = 1  \r\n    if x:\r\n        y = X\n\n") == (
        "x = 1\nif x:\n    y = X"
    )


@pytest.mark.parametrize("answer", ["", "  \n", "N/A"])
def test_blank_code_reads_as_blank(answer: str) -> None:
    assert CodeQuestion().read(answer) is None
    assert GUESSING.read(answer) is None


LONGEST_RUN = (
    "def longest_run(xs):\nbest = 0\nrun = 0\nprev = 0\nfor x in xs:\nif x == prev:\nrun += 1\n"
    "else:\nrun = 1\nbest = max(best, run)\nprev = x\nreturn best"
)


def test_guessing_reads_code_as_its_likeliest_layouts() -> None:
    assert GUESSING.read(FLAT) == [LAID_OUT]
    assert len(GUESSING.read(LONGEST_RUN) or []) == 3
    assert (
        len(
            CodeQuestion(reading=CodeReading(guess_indentation=True, most_guesses=1)).read(
                LONGEST_RUN
            )
            or []
        )
        == 1
    )


def test_code_that_compiles_or_that_no_layout_compiles_reads_as_itself() -> None:
    assert GUESSING.read(LAID_OUT) == [LAID_OUT]
    assert GUESSING.read("def f(:\nreturn 1") == ["def f(:\nreturn 1"]


def test_only_a_guessing_question_reads_answers_as_several() -> None:
    assert GUESSING.answer_shape == "multiple"
    assert CodeQuestion().answer_shape == "single"
    assert GUESSING.single() == CodeQuestion()
    assert CodeQuestion().single() == CodeQuestion()


def test_code_is_written_as_its_likeliest_layout() -> None:
    assert GUESSING.normalize(FLAT) == LAID_OUT
    assert CodeQuestion().normalize(FLAT) == FLAT
    assert GUESSING.first([LAID_OUT, FLAT]) == LAID_OUT


def test_code_says_its_answers_are_code() -> None:
    assert CodeQuestion().answer_schema() == {
        "type": "string",
        HINT_KEY: {"kind": "code", "language": "python"},
    }


def test_custom_code_is_told_how_answers_reach_it() -> None:
    assert CodeQuestion().python_type() == "str | None"
    assert GUESSING.python_type().startswith("list[str] | None")


def test_answers_that_are_programs_make_a_code_question() -> None:
    assert isinstance(infer_question([FLAT, LAID_OUT, "def f():\nreturn 1", ""]), CodeQuestion)


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
