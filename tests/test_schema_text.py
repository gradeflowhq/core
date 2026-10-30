import re
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from gradeflow_core import BUILTINS, AnswersCsv, Check, QuestionSet, ResultsCsv
from gradeflow_core.editing import HINT_KEY, QUESTION_TYPES, Position, compatible_methods
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

METHODS = {method.name: method for method in BUILTINS}

CONFIGS: list[type[BaseModel]] = [AnswersCsv, ResultsCsv, *QUESTION_TYPES.values()]
CLASS_NAME = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+$")
ID_CASE = re.compile(r"\bId\b")


class _ExplicitTitles(GenerateJsonSchema):
    def field_title_should_be_set(self, schema: Any) -> bool:
        return False


def _properties(schema: object) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(schema, dict):
        yield from schema.get("properties", {}).items()
        for value in schema.values():
            yield from _properties(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _properties(value)


def _titles(schema: object) -> Iterator[str]:
    if isinstance(schema, dict):
        if isinstance(schema.get("title"), str):
            yield schema["title"]
        for value in schema.values():
            yield from _titles(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _titles(value)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.__name__)
def test_config_fields_have_plain_titles_help_and_enum_labels(config: type[BaseModel]) -> None:
    schema = config.model_json_schema(mode="validation", schema_generator=_ExplicitTitles)
    for name, prop in _properties(schema):
        if prop.get("readOnly") or "const" in prop:
            continue
        where = f"{config.__name__}.{name}"
        # Pydantic leaves out a $ref field's title when it equals the model's title.
        assert prop.get("title") or "$ref" in prop, where
        assert prop.get("description", "").endswith("."), where
        assert not ID_CASE.search(prop.get("title", "") + " " + prop["description"]), where
        enum = prop.get("enum") or prop.get("items", {}).get("enum")
        if enum:
            labels = prop.get(HINT_KEY, {}).get("labels", {})
            assert set(labels) == set(enum), where


def _schemas() -> Iterator[dict[str, Any]]:
    yield from (config.model_json_schema() for config in CONFIGS)
    yield Check.model_json_schema()
    yield from (kind.json_schema() for kind in QUESTION_TYPES.values())
    question_set = QuestionSet(
        {
            "Q1": TextQuestion(),
            "Q2": ChoiceQuestion(options=frozenset({"a"})),
            "Q3": ValuesQuestion(value_types=["text", "number"]),
        }
    )
    for question in (None, "Q1", "Q2", "Q3"):
        for in_list in (False, True):
            position = Position(question_set, question, in_list=in_list, in_group=question is None)
            for info in compatible_methods(position):
                yield info.settings_schema(position)


def test_no_schema_title_is_a_class_name() -> None:
    for schema in _schemas():
        for title in _titles(schema):
            assert not CLASS_NAME.match(title), title
            assert not ID_CASE.search(title), title


def test_config_titles_read_like_labels() -> None:
    csv = ResultsCsv.model_json_schema()["properties"]
    assert csv["student_id_column"]["title"] == "Student ID column"
    question_set = QuestionSet({"Q1": TextQuestion()})
    schema = METHODS["gradeflow/custom-code-for-a-group"].settings_schema(
        Position(question_set, in_group=True)
    )
    assert schema["properties"]["questions"]["title"] == "Questions"
    text = TextQuestion.json_schema()
    assert text["$defs"]["TextReading"]["title"] == "Reading answers"
