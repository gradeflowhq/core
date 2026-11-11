import json
import re
from typing import Any

import pytest

from gradeflow_core import BUILTINS, InvalidError, Methods, QuestionSet
from gradeflow_core.common.hints import Hint
from gradeflow_core.editing import (
    HINT_KEY,
    EditingError,
    MarkingDraft,
    Position,
    compatible_methods,
)
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from gradeflow_core.steps.step import STEP_OWN_PROPERTIES
from tests.methods.helpers import with_value_inputs
from tests.schemas import schema_nodes

METHODS = {method.name: method for method in BUILTINS}
WORKFLOW = "gradeflow/workflow@1"

QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(),
        "Q2": ChoiceQuestion(options=("a", "b")),
        "Q3": ValuesQuestion(value_types=["text", "number"]),
        "Q4": NumberQuestion(),
    }
)
PLACES: dict[str, tuple[str | None, dict[str, Any] | None]] = {
    "question group": (None, None),
    "text": ("Q1", None),
    "choice": ("Q2", None),
    "values": ("Q3", None),
    "number": ("Q4", None),
    "text in a list": ("Q1", {}),
    "value": ("Q3", {"input": "{{ values[1] }}"}),
    "in question group": (None, {"question": "Q1"}),
}
"""Each place's question, and the check's keys in a workflow's steps (``None`` at the top level)."""


def draft_at(place: str, check: dict[str, Any]) -> tuple[MarkingDraft, str]:
    """A draft with ``check`` at ``place``, and the check's path."""
    question, list_keys = PLACES[place]
    marking = (
        check
        if list_keys is None
        else {"uses": WORKFLOW, "with": {"steps": [{**list_keys, **check}]}}
    )
    if question:
        marking = {"question": question, **marking}
    return MarkingDraft(marking, QUESTIONS, question), "" if list_keys is None else "/with/steps/0"


def schema_at(method: str, place: str, **check_keys: Any) -> dict[str, Any]:
    draft, path = draft_at(place, {"uses": METHODS[method].ref, **check_keys})
    return draft.schema(path)


def test_a_draft_offers_the_methods_compatible_with_its_check() -> None:
    methods = Methods(reversed(BUILTINS))
    draft = MarkingDraft({}, QUESTIONS, "Q1", methods=methods)
    assert draft.methods("") == compatible_methods(Position(QUESTIONS, "Q1"), methods=methods)


def test_no_setting_is_named_like_a_property_a_step_has() -> None:
    assert not [
        (method.name, key)
        for method in BUILTINS.by_ref.values()
        for key in method.settings.model_fields
        if key in STEP_OWN_PROPERTIES
    ]


def check_keys(place: str) -> list[str]:
    properties = schema_at("gradeflow/key-point", place, **{"if": "{{ true }}"})["properties"]
    return [key for key in properties if key in STEP_OWN_PROPERTIES]


def test_a_checks_fields_are_its_settings_and_the_keys_a_check_there_may_have() -> None:
    assert check_keys("text") == ["if", "feedback"]
    assert check_keys("text in a list") == ["name", "weight", "if", "feedback"]
    assert check_keys("in question group") == [
        "question",
        "input",
        "name",
        "weight",
        "if",
        "feedback",
    ]
    assert check_keys("question group") == ["if", "feedback"], (
        "a question group's marking reads no question"
    )
    group_check_schema = schema_at("gradeflow/key-point", "in question group")
    assert list(group_check_schema["properties"])[:3] == ["question", "input", "point"]
    assert group_check_schema["properties"]["question"]["enum"] == ["Q1", "Q2", "Q3", "Q4"]
    assert "question" in group_check_schema["required"], (
        "a check in a question group needs its question"
    )
    value_check_schema = schema_at("gradeflow/accepted-answers", "value")
    assert value_check_schema["properties"]["input"]["enum"] == [
        "{{ answer }}",
        "{{ values[0] }}",
        "{{ values[1] }}",
    ]
    multiple_values = {
        "question": "Q3",
        "uses": METHODS["gradeflow/multiple-values"].ref,
        "with": {"checks": with_value_inputs({"uses": METHODS["gradeflow/accepted-answers"].ref})},
    }
    value_schema = MarkingDraft(multiple_values, QUESTIONS, "Q3").schema("/with/checks/0")
    assert "input" not in value_schema["properties"], "each check's value is fixed"


def test_a_checks_fields_are_in_the_order_the_editor_shows_them() -> None:
    def keys(place: str) -> list[str]:
        return list(schema_at("gradeflow/key-point", place, **{"if": "{{ true }}"})["properties"])

    settings = ["point", "match", "strictness", "whole_words"]
    assert keys("in question group") == [
        "question",
        "input",
        *settings,
        "name",
        "weight",
        "if",
        "feedback",
    ]
    assert keys("text") == [*settings, "if", "feedback"]


def test_fields_offer_the_questions_options_and_students_answers() -> None:
    choice_properties = schema_at("gradeflow/multiple-choice", "choice")["properties"]
    assert (
        choice_properties["correct"]["items"]["enum"],
        choice_properties["correct"]["uniqueItems"],
    ) == (
        ["a", "b"],
        True,
    ), "options are chosen, each once"
    answers_properties = schema_at("gradeflow/accepted-answers", "text")["properties"]
    assert answers_properties["answers"]["x-gradeflow"] == {"lists_answers": True}
    group_properties = schema_at("gradeflow/custom-code-for-a-group", "question group")[
        "properties"
    ]
    assert group_properties["questions"]["items"]["enum"] == ["Q1", "Q2", "Q3", "Q4"]


@pytest.mark.parametrize(
    ("method", "field", "lists_answers"),
    [
        ("gradeflow/accepted-answers", "answers", True),
        ("gradeflow/key-points", "points", False),
        ("gradeflow/key-point", "point", False),
    ],
)
def test_only_fields_of_whole_answers_take_them(
    method: str, field: str, lists_answers: bool
) -> None:
    properties = schema_at(method, "text")["properties"]
    assert properties[field].get("x-gradeflow", {}).get("lists_answers", False) == lists_answers


def new_check(method: str, place: str) -> dict[str, Any]:
    """A new check of ``method`` at ``place``: a new marking, a check added to a list, or, as
    adding can't choose a value, a check whose method changed."""
    question, list_keys = PLACES[place]
    reference = METHODS[method].ref
    if list_keys is None:
        return MarkingDraft.new(reference, QUESTIONS, question).data
    draft, path = draft_at(place, {"uses": "gradeflow/unknown@1"})
    if "input" in list_keys:
        draft.change_method(path, reference)
    else:
        draft.replace({**draft.data, "uses": "gradeflow/workflow@1", "with": {"steps": []}})
        draft.add(path, reference, list_keys.get("question"))
    check: dict[str, Any] = draft.data["with"]["steps"][0]
    return check


def methods_at(place: str) -> Methods:
    draft, path = draft_at(place, {"uses": WORKFLOW})
    return draft.methods(path)


@pytest.mark.parametrize("place", PLACES)
def test_new_checks_are_plain_json_of_their_method(place: str) -> None:
    for method in methods_at(place):
        check = new_check(method.name, place)
        assert json.loads(json.dumps(check)) == check
        assert check["uses"] == method.ref


def test_multiple_values_has_a_check_for_each_value_no_more() -> None:
    checks = schema_at("gradeflow/multiple-values", "values")["properties"]["checks"]
    assert (checks["minItems"], checks["maxItems"]) == (2, 2)


def test_a_new_composed_check_starts_with_its_lists() -> None:
    assert new_check("gradeflow/answer-sets", "question group")["with"] == {
        "sets": [{"name": "Expected", "uses": "gradeflow/workflow@1", "with": {"steps": []}}]
    }


def test_a_new_multiple_values_check_checks_each_value() -> None:
    check = new_check("gradeflow/multiple-values", "values")
    assert check["question"] == "Q3"
    assert [inner["uses"] for inner in check["with"]["checks"]] == [
        "gradeflow/accepted-answers@1",
        "gradeflow/accepted-answers@1",
    ]


@pytest.mark.parametrize("place", PLACES)
def test_every_hint_uses_the_typed_vocabulary(place: str) -> None:
    for method in methods_at(place):
        for node in schema_nodes(schema_at(method.name, place)):
            if HINT_KEY in node:
                Hint.model_validate(node[HINT_KEY], strict=True)


def test_unknown_methods_are_refused_by_the_draft() -> None:
    draft = MarkingDraft({"uses": "gradeflow/nope@1"}, QUESTIONS, "Q1")
    with pytest.raises(EditingError, match=re.escape("There's no method gradeflow/nope@1.")):
        draft.schema("")


def test_a_checks_schema_and_new_settings_need_no_methods() -> None:
    assert "points" in schema_at("gradeflow/key-points", "text")["properties"]
    assert new_check("gradeflow/key-points", "text") == {
        "question": "Q1",
        "uses": "gradeflow/key-points@1",
        "with": {},
    }


MULTIPLE_VALUES_MARKING = {
    "question": "Q3",
    "uses": "gradeflow/multiple-values@1",
    "with": {
        "checks": with_value_inputs(
            {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["x"]}},
            {
                "uses": "gradeflow/not-allowed@1",
                "with": {"check": {"uses": "gradeflow/number-range@1"}},
            },
        )
    },
}


@pytest.mark.parametrize(
    ("path", "expected_position"),
    [
        ("", ("Q3", None, False)),
        ("/with/checks/0", ("Q3", 1, True)),
        ("/with/checks/1/with/check", ("Q3", 2, False)),
    ],
)
def test_checks_nest_at_any_depth(path: str, expected_position: tuple[object, ...]) -> None:
    position = MarkingDraft(MULTIPLE_VALUES_MARKING, QUESTIONS, "Q3").position(path)
    assert (
        position.question,
        position.value,
        position.is_in_list,
    ) == expected_position


def test_a_check_in_a_question_group_reads_its_own_question() -> None:
    group_marking = {
        "uses": "gradeflow/workflow@1",
        "with": {"steps": [{"question": "Q4", "uses": "gradeflow/number-range@1"}]},
    }
    draft = MarkingDraft(group_marking, QUESTIONS, None)
    assert (draft.position("").is_in_group, draft.position("/with/steps/0").question) == (
        True,
        "Q4",
    )


def test_the_schema_at_a_path_changes_when_steps_move() -> None:
    steps = [
        {"uses": "gradeflow/length@1"},
        {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["x"]}},
    ]
    marking = {"question": "Q1", "uses": WORKFLOW, "with": {"steps": steps}}
    draft = MarkingDraft(marking, QUESTIONS, "Q1")
    assert draft.schema("/with/steps/0")["title"] == "Length"
    draft.move("/with/steps/0", 1)
    assert draft.schema("/with/steps/0")["title"] == "Accepted Answers"


def test_an_empty_place_for_a_check_is_a_position_too() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/not-allowed@1", "with": {}}
    position = MarkingDraft(marking, QUESTIONS, "Q1").position("/with/check")
    assert (position.question, position.is_in_list) == ("Q1", False)


@pytest.mark.parametrize(
    "path", ["/with/nope", "/with/checks/5", "/with/checks/5/x", "/checks/0", "/with/checks/0/with"]
)
def test_paths_that_arent_checks_are_refused(path: str) -> None:
    with pytest.raises(EditingError, match=re.escape("isn't a step of this marking")):
        MarkingDraft(MULTIPLE_VALUES_MARKING, QUESTIONS, "Q3").position(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("uses: [\n", ["Line 2: expected the node content, but found '<stream end>'."]),
        (
            "- uses: gradeflow/key-point@1\n",
            ["Write one check, starting with uses: and its method's name."],
        ),
        (
            "question: Q2\nuses: gradeflow/key-point@1\nwith: {point: a}\n",
            ["Line 1: Set question to Q1."],
        ),
        (
            "question: Q1\nuses: gradeflow/accepted-answers@1\nwith:\n  answers: [' ']\n",
            ["Line 4: answers: Can't be blank."],
        ),
        (
            "question: Q1\nuses: gradeflow/workflow@1\nwith:\n  steps: []\n",
            ["Line 4: steps: Add at least one step."],
        ),
        (
            "question: Q1\nuses: gradeflow/workflow@1\nwith:\n  steps:\n"
            "  - uses: gradeflow/key-point@1\n    with: {point: a}\n    if: '{{ garbage(( }}'\n",
            ["Line 7: if: Can't read this condition."],
        ),
        (
            "question: Q1\nuses: gradeflow/key-point@1\nwith: {point: a}\nbogus: 1\n",
            ["Line 4: bogus: Unknown setting."],
        ),
        ("question: Q1\nuses: gradeflow/key-point@1\nwith: 5\n", ["Line 3: Must be an object."]),
        (
            "question: Q1\nuses: gradeflow/length@1\nwith: {min_length: 3, max_length: 2}\n",
            ["Line 2: The highest count is below the lowest."],
        ),
        (
            "question: Q1\nuses: gradeflow/workflow@1\nwith:\n  steps:\n"
            "  - uses: gradeflow/key-point@1\n    with: {point: a}\n    weigth: 2\n",
            ["Line 7: weigth: Unknown setting."],
        ),
        (
            "question: Q1\nuses: gradeflow/workflow@1\nwith:\n  steps:\n"
            "  - uses: gradeflow/multiple-choice@1\n    with: {correct: [a]}\n",
            ["Line 5: Multiple Choice can't be used on a text question."],
        ),
        (
            "question: Q1\nuses: gradeflow/workflow@1\nwith:\n  steps:\n"
            "  - uses: gradeflow/key-point@1\n    with: {point: a}\n"
            "    if: '{{ steps.later.output.correct }}'\n"
            "  - id: later\n    uses: gradeflow/key-point@1\n    with: {point: b}\n",
            ["Line 7: if: Key Point must come before this step."],
        ),
    ],
)
def test_a_marking_written_as_yaml_says_whats_wrong_on_its_lines(
    text: str, expected: list[str]
) -> None:
    try:
        problems = MarkingDraft.from_yaml(text, QUESTIONS, "Q1").problems
    except InvalidError as error:
        problems = error.problems
    assert [str(problem) for problem in problems] == expected


def test_a_marking_written_as_yaml_reads_as_its_check() -> None:
    draft = MarkingDraft.from_yaml(
        "question: Q1\nuses: gradeflow/key-point@1\nwith: {point: a}\n", QUESTIONS, "Q1"
    )
    assert draft.data == {"question": "Q1", "uses": "gradeflow/key-point@1", "with": {"point": "a"}}
    assert (draft.finalized().describe(), draft.problems) == ("Includes `a`.", [])


def test_a_drafts_schemas_differ_where_the_checks_positions_do() -> None:
    custom_code = METHODS["gradeflow/custom-code"]
    draft = MarkingDraft({"question": "Q3", "uses": custom_code.ref, "with": {}}, QUESTIONS, "Q3")
    whole_answer_code_schema = draft.schema("")["properties"]["code"]
    multiple_values_ref = METHODS["gradeflow/multiple-values"].ref
    draft.replace(
        {
            "question": "Q3",
            "uses": multiple_values_ref,
            "with": {"checks": with_value_inputs({"uses": custom_code.ref})},
        }
    )
    one_value_code_schema = draft.schema("/with/checks/0")["properties"]["code"]
    assert one_value_code_schema != whole_answer_code_schema
    fresh_draft = MarkingDraft(draft.data, QUESTIONS, "Q3")
    assert one_value_code_schema == fresh_draft.schema("/with/checks/0")["properties"]["code"]
