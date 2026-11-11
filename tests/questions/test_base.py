import pytest
from pydantic import ValidationError

from gradeflow_core import InvalidError, Question
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import ListReading, TextReading
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion


def test_base_question_is_abstract() -> None:
    with pytest.raises(TypeError):
        Question()  # type: ignore[abstract, call-arg]


@pytest.mark.parametrize("max_marks", [-5, "nan", "inf"])
def test_max_marks_is_a_finite_number_of_zero_or_more(max_marks: object) -> None:
    with pytest.raises(InvalidError) as exc_info:
        TextQuestion.load({"max_marks": max_marks})
    assert [problem.field for problem in exc_info.value.problems] == ["/max_marks"]


def test_multiple_values_need_at_least_one_value() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ValuesQuestion(value_types=[])
    assert exc_info.value.errors()[0]["loc"] == ("value_types",)


def test_separator_is_not_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ListReading(separator="")
    assert exc_info.value.errors()[0]["loc"] == ("separator",)


@pytest.mark.parametrize(
    ("question", "answers", "normalized"),
    [
        (NumberQuestion(), ["42", "42.0", " 42 ", "4.2e1", "42 kg"], "42"),
        (NumberQuestion(), ["9.81 m/s^2", "9.81"], "9.81"),
        (
            ChoiceQuestion(options=("b", "d")),
            ["B,D", "d, b"],
            "b, d",
        ),
        (
            ValuesQuestion(value_types=["number", "text"]),
            ["3.0, x", "(3, x)"],
            "3, x",
        ),
        (TextQuestion(), ["", "N/A", "  "], ""),
        (TextQuestion(), ["Canberra", " canberra "], "canberra"),
        (NumberQuestion(), ["twelve"], "twelve"),
    ],
)
def test_answers_read_alike_normalize_alike(
    question: Question, answers: list[str], normalized: str
) -> None:
    assert {question.normalize(answer) for answer in answers} == {normalized}


def test_answers_read_differently_normalize_differently() -> None:
    question = TextQuestion(reading=TextReading(ignore_case=False))
    assert question.normalize("AA") != question.normalize("Aa")


@pytest.mark.parametrize(
    ("question", "python_type"),
    [
        (TextQuestion(), "str | None"),
        (NumberQuestion(), "int | float | None"),
        (ChoiceQuestion(options=("a",)), "set[str]"),
        (
            ValuesQuestion(value_types=["text", "number"]),
            "list[str | int | float | None], in order: text, number",
        ),
    ],
)
def test_a_question_says_how_custom_code_gets_its_answers(
    question: Question, python_type: str
) -> None:
    assert question.python_type() == python_type


def test_only_a_question_of_several_values_has_a_question_for_each() -> None:
    values = ValuesQuestion(value_types=["text", "number"])
    assert (values.value_count, TextQuestion().value_count) == (2, 0)
    assert [type(values.value_question(value)) for value in (1, 2)] == [
        TextQuestion,
        NumberQuestion,
    ]
    assert (values.value_question(3), TextQuestion().value_question(1)) == (None, None)


@pytest.mark.parametrize(
    ("question", "answer", "normalized"),
    [
        (
            ValuesQuestion(value_types=["number", "text"], reading=ListReading(separator="~")),
            "3.0~East",
            "3~ east",
        ),
        (
            ChoiceQuestion(
                options=("a", "b"),
                allow_multiple=True,
                reading=ListReading(separator=";"),
            ),
            "B;A",
            "a; b",
        ),
        (
            ChoiceQuestion(
                options=("a", "b"),
                allow_multiple=True,
                reading=ListReading(trim_spaces=False),
            ),
            "B,A",
            "a,b",
        ),
    ],
)
def test_a_normalized_answer_reads_as_the_answer_does(
    question: Question, answer: str, normalized: str
) -> None:
    assert question.normalize(answer) == normalized
    assert question.read(question.normalize(answer)) == question.read(answer)
