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
from gradeflow_core.editing import HINT_KEY, QUESTION_TYPES, QuestionType, new_question


def test_catalogue_lists_every_question_type_with_text_first() -> None:
    assert list(QUESTION_TYPES) == ["text", "number", "choice", "values", "code"]
    assert set(QUESTION_TYPES) == set(get_args(QuestionType))
    assert [kind.label for kind in QUESTION_TYPES.values()] == [
        "Text",
        "Number",
        "Choice",
        "Multiple values",
        "Code",
    ]
    assert all(kind.summary.endswith(".") for kind in QUESTION_TYPES.values())


def test_catalogue_schemas_describe_editable_fields() -> None:
    schemas = {name: kind.json_schema() for name, kind in QUESTION_TYPES.items()}

    for question_type, schema in schemas.items():
        assert schema["properties"]["type"]["const"] == question_type
        assert schema["properties"]["type"]["readOnly"] is True
    assert schemas["choice"]["properties"]["options"]["minItems"] == 1
    assert schemas["text"]["properties"]["description"]["type"] == ["string", "null"]


def test_catalogue_matches_each_types_own_type() -> None:
    assert all(kind.model_fields["type"].default == name for name, kind in QUESTION_TYPES.items())


def test_catalogue_fields_read_like_a_form() -> None:
    schemas = {name: kind.json_schema() for name, kind in QUESTION_TYPES.items()}
    text, choice, multi = schemas["text"], schemas["choice"], schemas["values"]

    assert text["properties"]["max_marks"]["title"] == "Maximum marks"
    assert text["properties"]["description"][HINT_KEY] == {"kind": "markdown"}
    assert choice["properties"]["options"]["title"] == "Options"
    assert multi["properties"]["value_types"][HINT_KEY]["labels"] == {
        "text": "Text",
        "number": "Number",
    }
    for schema in schemas.values():
        for config in schema["$defs"].values():
            for name, prop in config["properties"].items():
                assert prop[HINT_KEY]["advanced"] is True, name
                assert prop["title"], name
                assert prop["description"].endswith("."), name


@pytest.mark.parametrize("question_type", QUESTION_TYPES)
def test_each_type_offers_a_question_to_start_from(question_type: QuestionType) -> None:
    assert isinstance(new_question(question_type), QUESTION_TYPES[question_type])


def test_a_question_of_a_type_is_set_up_as_its_schema_describes() -> None:
    choice = new_question("choice", {"options": ["A", "B", "C"]})
    assert choice.model_dump()["options"] == {"a", "b", "c"}
    with pytest.raises(InvalidError) as error:
        new_question("choice", {"options": []})
    assert error.value.problems == [
        Problem("Add at least one.", field=("options",), incomplete=True)
    ]


def test_a_question_loads_as_the_type_it_names_or_the_type_it_is_loaded_as() -> None:
    choice = Question.load({"type": "choice", "options": ["a"]})
    assert choice == ChoiceQuestion(options=frozenset({"a"}))
    assert Question.load(choice.dump()) == choice
    assert choice.dump() == {"type": "choice", "options": ["a"]}
    assert NumberQuestion.load({"max_marks": 2}) == NumberQuestion(max_marks=2)
    with pytest.raises(InvalidError, match="type: Choose 'number'"):
        NumberQuestion.load({"type": "text"})
    with pytest.raises(InvalidError, match="Must be an object"):
        Question.load("text")
    assert TextQuestion.load({}).dump() == {"type": "text"}
