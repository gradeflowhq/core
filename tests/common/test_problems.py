import math
from typing import Any

import pytest
from pydantic import ValidationError

from gradeflow_core import (
    Check,
    InvalidError,
)
from gradeflow_core.common.error_formatting import error_message
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.reading import ListReading

RAW_PYDANTIC = (
    "Input should be",
    "validation error for",
    "Unable to extract tag",
    "type=",
    "String should",
)


def errors(check: dict[str, Any]) -> list[tuple[str, tuple[str | int, ...], str]]:
    with pytest.raises(InvalidError) as caught:
        Check.load(check)
    found = [(p.check, p.field, p.message) for p in caught.value.problems]
    assert not any(fragment in message for *_, message in found for fragment in RAW_PYDANTIC)
    return found


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        ({}, [("", ("uses",), "Choose a method.")]),
        ({"uses": "gradeflow/nope@1"}, [("", ("uses",), "There's no method gradeflow/nope@1.")]),
        ({"uses": "gradeflow/not-allowed@1"}, [("", ("with",), "Required.")]),
        ({"uses": "gradeflow/pattern@1", "with": {}}, [("", ("pattern",), "Required.")]),
        (
            {"uses": "gradeflow/pattern@1", "with": {"pattern": "("}},
            [
                (
                    "",
                    ("pattern",),
                    "Not a valid pattern: missing ), unterminated subpattern at position 0.",
                )
            ],
        ),
        (
            {"uses": "gradeflow/number-range@1", "with": {"min_value": 5, "max_value": 1}},
            [("", (), "The highest number is below the lowest.")],
        ),
        (
            {"uses": "gradeflow/length@1", "with": {"min_length": 8, "max_length": 2}},
            [("", (), "The highest count is below the lowest.")],
        ),
        (
            {"uses": "gradeflow/number-range@1", "with": {"min_value": math.nan}},
            [("", ("min_value",), "Must be a number.")],
        ),
        ({"uses": "gradeflow/accepted-answers@1", "with": {}}, [("", ("answers",), "Required.")]),
        (
            {"uses": "gradeflow/accepted-answers@1", "with": {"answers": [" "]}},
            [("", ("answers", 0), "Can't be blank.")],
        ),
        (
            {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["1"], "tolerance": -1}},
            [("", ("tolerance",), "Must be 0 or more.")],
        ),
        (
            {"uses": "gradeflow/key-points@1", "with": {"points": ["light", " "]}},
            [("", ("points", 1), "Can't be blank.")],
        ),
        (
            {"uses": "gradeflow/length@1", "with": {"min_length": -1}},
            [("", ("min_length",), "Must be 0 or more.")],
        ),
        (
            {"uses": "gradeflow/length@1", "with": {"min_length": "x"}},
            [("", ("min_length",), "Must be a whole number.")],
        ),
        (
            {"uses": "gradeflow/length@1", "with": {"count": "lines"}},
            [("", ("count",), "Choose 'words' or 'characters'.")],
        ),
        (
            {"uses": "gradeflow/multiple-choice@1", "with": {"correct": [1]}},
            [("", ("correct", 0), "Must be text.")],
        ),
        (
            {"uses": "gradeflow/code-tests@1", "with": {"tests": [{}]}},
            [
                ("", ("tests", 0, "expression"), "Required."),
                ("", ("tests", 0, "expected"), "Required."),
            ],
        ),
        (
            {
                "uses": "gradeflow/combination@1",
                "with": {"checks": [{"weight": math.nan, "uses": "gradeflow/full-marks@1"}]},
            },
            [("with.checks.0", ("weight",), "Must be a number.")],
        ),
        (
            {"uses": "gradeflow/combination@1", "with": {"checks": [{}]}},
            [("with.checks.0", ("uses",), "Choose a method.")],
        ),
        (
            {"uses": "gradeflow/combination@1", "with": {"checks": ["bad"]}},
            [("", ("checks", 0), "A check needs uses: and a method's name.")],
        ),
        (
            {
                "uses": "gradeflow/custom-code@1",
                "with": {"parameters": {"p": {"type": "integer", "value": "x"}}},
            },
            [("", ("parameters", "p", "value"), "Must be a whole number.")],
        ),
        (
            {
                "uses": "gradeflow/custom-code@1",
                "with": {
                    "code": "proportion = 1",
                    "parameters": {
                        "t": {"type": "table", "value": {"uses": {"type": "integer", "value": "x"}}}
                    },
                },
            },
            [("", ("parameters", "t", "value", "uses", "value"), "Must be a whole number.")],
        ),
        (
            {"uses": "gradeflow/custom-code@1", "with": {"parameters": {"p": {"type": "matrix"}}}},
            [("", ("parameters", "p"), "There's no type matrix.")],
        ),
        (
            {
                "uses": "gradeflow/custom-code@1",
                "with": {"parameters": {"answer": {"type": "text", "value": "x"}}},
            },
            [
                (
                    "",
                    ("parameters", "answer", "[key]"),
                    "Your code already uses answer; choose another name.",
                )
            ],
        ),
        (
            {"uses": "gradeflow/key-point@1", "with": {"point": "a"}, "id": "1st"},
            [("", ("id",), "Use letters, digits and underscores, not starting with a digit.")],
        ),
    ],
)
def test_checks_refuse_what_would_mark_wrongly_in_plain_words(
    check: dict[str, Any], expected: list[tuple[str, tuple[str | int, ...], str]]
) -> None:
    assert errors(check) == expected


@pytest.mark.parametrize(
    "check",
    [
        {"uses": "gradeflow/number-range@1", "with": {"min_value": 1, "max_value": 1}},
        {"uses": "gradeflow/length@1", "with": {"min_length": 0, "max_length": 0}},
        {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["1"], "tolerance": 0}},
        {
            "uses": "gradeflow/combination@1",
            "with": {"checks": [{"weight": -1, "uses": "gradeflow/full-marks@1"}]},
        },
    ],
)
def test_checks_accept_settings_at_their_limits(check: dict[str, Any]) -> None:
    Check.load(check)


def test_messages_keep_the_reason_a_question_gives() -> None:
    with pytest.raises(ValidationError) as caught:
        ChoiceQuestion.model_validate({"options": ["a", "A"], "reading": {"ignore_case": True}})
    [error] = caught.value.errors()
    assert error_message(error) == "Options must be different: “a” appears twice."
    case_sensitive = ChoiceQuestion(
        options=frozenset({"a", "A"}), reading=ListReading(ignore_case=False)
    )
    assert case_sensitive.options == {"a", "A"}
