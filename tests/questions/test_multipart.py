import pytest

from gradeflow_core.common.problems import InvalidError
from gradeflow_core.questions.multipart import MultipartQuestion
from gradeflow_core.questions.parts import AnswerPart
from gradeflow_core.questions.reading import ListReading


def multipart(*types: str, separator: str = ",") -> MultipartQuestion:
    return MultipartQuestion.load(
        {
            "reading": {"separator": separator},
            "parts": [
                {"name": f"part_{number}", "type": part_type}
                for number, part_type in enumerate(types, 1)
            ],
        }
    )


def test_a_multipart_answer_reads_each_part_as_its_type() -> None:
    question = MultipartQuestion.load(
        {"parts": [{"name": "place"}, {"name": "year", "type": "number"}]}
    )
    assert question.read("Waterloo, 1815") == {"place": "waterloo", "year": 1815}


def test_parts_with_a_custom_separator() -> None:
    question = multipart("text", "text", "number", "text", separator=";")
    assert question.read("alpha; beta ; 3; gamma") == {
        "part_1": "alpha",
        "part_2": "beta",
        "part_3": 3,
        "part_4": "gamma",
    }


def test_blank_parts_read_as_none() -> None:
    question = multipart("number", "text", "number", separator=";")
    assert question.read("1; N/A; ") == {"part_1": 1, "part_2": None, "part_3": None}
    assert question.read("; N/A; ") is None, "every part blank is a blank answer"


def test_parts_read_fractions_and_scientific_notation() -> None:
    question = multipart("number", "number", "text")
    assert question.read("3/2, 1e2, hello") == {"part_1": 1.5, "part_2": 100.0, "part_3": "hello"}


def test_an_answer_with_the_wrong_number_of_parts_raises() -> None:
    question = multipart("text", "number", "text", separator="|")
    with pytest.raises(ValueError, match="Expected 3 parts"):
        question.read("only_one|2")


def test_text_parts_are_read_as_text() -> None:
    question = MultipartQuestion(
        reading=ListReading(separator="~"),
        parts=[AnswerPart(name="agree"), AnswerPart(name="count", type="number")],
    )
    assert question.read(" Yes ~ 2 ") == {"agree": "yes", "count": 2}


def test_parts_in_brackets_are_read_inside_them() -> None:
    coordinates = multipart("number", "number")
    assert coordinates.read("(3, -2)") == coordinates.read("[3,-2]") == {"part_1": 3, "part_2": -2}
    with pytest.raises(ValueError, match="Couldn't read"):
        coordinates.read("(3, -2")


def test_a_multipart_answer_is_written_part_by_part() -> None:
    question = MultipartQuestion.load(
        {"parts": [{"name": "x", "type": "number"}, {"name": "atomic_number"}]}
    )
    titles = [schema["title"] for schema in question.answer_schema()["properties"].values()]
    assert titles == ["X", "Atomic number"]
    assert question.answer_values("(3, x)") == {"x": "3", "atomic_number": "x"}
    assert question.answer_values(" ") == {"x": "", "atomic_number": ""}
    with pytest.raises(ValueError, match="Expected 2 parts, but got 1"):
        question.answer_values("3")
    assert question.answer_from({"x": " 3", "atomic_number": "x "}) == "3,x"
    assert question.answer_from({"x": "", "atomic_number": " "}) == ""


def test_a_part_written_with_the_minus_sign_reads_as_a_number() -> None:
    question = multipart("text", "number", "number", separator="~")
    assert question.read("terminal~\u22121~0") == {"part_1": "terminal", "part_2": -1, "part_3": 0}


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ([{"name": "a"}, {"name": "a"}], "Give each part its own name."),
        ([{"name": "a"}], "Add at least 2."),
        ([{"name": "a"}, {"name": "2b"}], "Use letters, digits and underscores"),
    ],
)
def test_parts_need_two_or_more_distinct_identifier_names(
    parts: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(InvalidError, match=message):
        MultipartQuestion.load({"parts": parts})
