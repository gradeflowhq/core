from typing import get_args

import pytest

from gradeflow_core import (
    ChoiceQuestion,
    InvalidError,
    NumberQuestion,
    Problem,
    Question,
    TextQuestion,
)
from gradeflow_core.editing import HINT_KEY, QUESTION_CLASSES, QuestionType, new_question


def test_catalogue_lists_every_question_type_with_text_first() -> None:
    assert list(QUESTION_CLASSES) == ["text", "number", "choice", "values", "code"]
    assert set(QUESTION_CLASSES) == set(get_args(QuestionType))
    assert [question_type.label for question_type in QUESTION_CLASSES.values()] == [
        "Text",
        "Number",
        "Choice",
        "Multiple values",
        "Code",
    ]
    assert all(question_type.summary.endswith(".") for question_type in QUESTION_CLASSES.values())


@pytest.mark.parametrize("question_type", QUESTION_CLASSES)
def test_a_question_type_is_fixed_in_its_schema(question_type: QuestionType) -> None:
    type_schema = QUESTION_CLASSES[question_type].json_schema()["properties"]["type"]
    assert (type_schema["const"], type_schema["readOnly"]) == (question_type, True)


def test_catalogue_schemas_describe_editable_fields() -> None:
    schemas = {
        name: question_type.json_schema() for name, question_type in QUESTION_CLASSES.items()
    }
    assert schemas["choice"]["properties"]["options"]["minItems"] == 1
    assert schemas["text"]["properties"]["description"]["type"] == ["string", "null"]


def test_catalogue_matches_each_types_own_type() -> None:
    assert all(
        question_type.model_fields["type"].default == name
        for name, question_type in QUESTION_CLASSES.items()
    )


def test_catalogue_fields_read_like_a_form() -> None:
    schemas = {
        name: question_type.json_schema() for name, question_type in QUESTION_CLASSES.items()
    }
    text, choice, values_schema = schemas["text"], schemas["choice"], schemas["values"]

    assert text["properties"]["max_marks"]["title"] == "Maximum marks"
    assert text["properties"]["description"][HINT_KEY] == {"kind": "markdown"}
    assert choice["properties"]["options"]["title"] == "Options"
    assert values_schema["properties"]["value_types"][HINT_KEY]["labels"] == {
        "text": "Text",
        "number": "Number",
    }
    for schema in schemas.values():
        for config in schema["$defs"].values():
            for name, property_schema in config["properties"].items():
                assert property_schema[HINT_KEY]["advanced"] is True, name
                assert property_schema["title"], name
                assert property_schema["description"].endswith("."), name


@pytest.mark.parametrize("question_type", QUESTION_CLASSES)
def test_each_type_offers_a_question_to_start_from(question_type: QuestionType) -> None:
    assert isinstance(new_question(question_type), QUESTION_CLASSES[question_type])


def test_a_question_of_a_type_is_set_up_as_its_schema_describes() -> None:
    choice = QUESTION_CLASSES["choice"].load({"options": ["A", "B", "C"]})
    assert choice.model_dump()["options"] == ("a", "b", "c")
    with pytest.raises(InvalidError) as exc_info:
        QUESTION_CLASSES["choice"].load({"options": []})
    assert exc_info.value.problems == [
        Problem("Add at least one.", field=("options",), is_incomplete=True)
    ]


def test_a_question_loads_as_the_type_it_names_or_the_type_it_is_loaded_as() -> None:
    choice = Question.load({"type": "choice", "options": ["a"]})
    assert choice == ChoiceQuestion(options=("a",))
    assert Question.load(choice.dump()) == choice
    assert choice.dump() == {"type": "choice", "options": ["a"]}
    assert NumberQuestion.load({"max_marks": 2}) == NumberQuestion(max_marks=2)
    with pytest.raises(InvalidError, match="type: Choose 'number'"):
        NumberQuestion.load({"type": "text"})
    with pytest.raises(InvalidError, match="Must be an object"):
        Question.load("text")
    assert TextQuestion.load({}).dump() == {"type": "text"}
