import re
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from gradeflow_core import BUILTINS, AnswersCsv, Check, QuestionSet, ResultsCsv
from gradeflow_core.editing import HINT_KEY, QUESTION_CLASSES, Position, compatible_methods
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from tests.schemas import schema_nodes

METHODS = {method.name: method for method in BUILTINS}

FORM_MODELS: list[type[BaseModel]] = [AnswersCsv, ResultsCsv, *QUESTION_CLASSES.values()]
CAMEL_CASE_TITLE = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+$")
CAPITALISED_ID_WORD = re.compile(r"\bId\b")


class _SchemaWithoutDefaultTitles(GenerateJsonSchema):
    def field_title_should_be_set(self, schema: Any) -> bool:
        return False


@pytest.mark.parametrize("model", FORM_MODELS, ids=lambda model: model.__name__)
def test_form_fields_have_plain_titles_help_and_enum_labels(model: type[BaseModel]) -> None:
    schema = model.model_json_schema(
        mode="validation", schema_generator=_SchemaWithoutDefaultTitles
    )
    properties = (
        item for node in schema_nodes(schema) for item in node.get("properties", {}).items()
    )
    for name, property_schema in properties:
        if property_schema.get("readOnly") or "const" in property_schema:
            continue
        field_path = f"{model.__name__}.{name}"
        # Pydantic leaves out a $ref field's title when it equals the model's title.
        assert property_schema.get("title") or "$ref" in property_schema, field_path
        assert property_schema.get("description", "").endswith("."), field_path
        assert not CAPITALISED_ID_WORD.search(
            property_schema.get("title", "") + " " + property_schema["description"]
        ), field_path
        enum = property_schema.get("enum") or property_schema.get("items", {}).get("enum")
        if enum:
            labels = property_schema.get(HINT_KEY, {}).get("labels", {})
            assert set(labels) == set(enum), field_path


def _schemas() -> Iterator[dict[str, Any]]:
    yield from (model.model_json_schema() for model in FORM_MODELS)
    yield Check.model_json_schema()
    yield from (question_type.json_schema() for question_type in QUESTION_CLASSES.values())
    question_set = QuestionSet(
        {
            "Q1": TextQuestion(),
            "Q2": ChoiceQuestion(options=("a",)),
            "Q3": ValuesQuestion(value_types=["text", "number"]),
        }
    )
    for question_id in (None, "Q1", "Q2", "Q3"):
        for is_in_list in (False, True):
            position = Position(
                question_set, question_id, is_in_list=is_in_list, is_in_group=question_id is None
            )
            for method in compatible_methods(position):
                yield method.settings_schema(position)


def test_no_schema_title_is_a_class_name() -> None:
    for schema in _schemas():
        for node in schema_nodes(schema):
            if isinstance(title := node.get("title"), str):
                assert not CAMEL_CASE_TITLE.match(title), title
                assert not CAPITALISED_ID_WORD.search(title), title


def test_form_titles_read_like_labels() -> None:
    results_csv_properties = ResultsCsv.model_json_schema()["properties"]
    assert results_csv_properties["student_id_column"]["title"] == "Student ID column"
    question_set = QuestionSet({"Q1": TextQuestion()})
    schema = METHODS["gradeflow/custom-code-for-a-group"].settings_schema(
        Position(question_set, is_in_group=True)
    )
    assert schema["properties"]["questions"]["title"] == "Questions"
    text = TextQuestion.json_schema()
    assert text["$defs"]["TextReading"]["title"] == "Reading answers"
