import json
import re
from collections.abc import Iterator

import pytest

from gradeflow_core import BUILTINS, InvalidError, Methods, QuestionSet
from gradeflow_core.common.hints import Hint
from gradeflow_core.editing import EditingError, MarkingDraft, Position, compatible_methods
from gradeflow_core.editing.drafts import _check_schema, _new_check
from gradeflow_core.methods.check import CHECK_FIELDS
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import CodeReading
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

METHODS = {method.name: method for method in BUILTINS}

QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(),
        "Q2": ChoiceQuestion(options=frozenset({"a", "b"})),
        "Q3": ValuesQuestion(value_types=["text", "number"]),
        "Q4": NumberQuestion(),
    }
)
POSITIONS = [
    Position(QUESTIONS, in_group=True),
    Position(QUESTIONS, "Q1"),
    Position(QUESTIONS, "Q2"),
    Position(QUESTIONS, "Q3"),
    Position(QUESTIONS, "Q4"),
    Position(QUESTIONS, "Q3", 2, in_list=True),
    Position(QUESTIONS, "Q1", in_list=True, in_group=True),
]
IDS = ["group", "text", "choice", "values", "number", "value", "in group"]


def test_a_draft_offers_the_methods_compatible_with_its_check() -> None:
    methods = Methods(reversed(BUILTINS))
    draft = MarkingDraft({}, QUESTIONS, "Q1", methods=methods)
    assert draft.methods("") == compatible_methods(Position(QUESTIONS, "Q1"), methods=methods)


def test_no_setting_is_named_like_a_key_a_check_has() -> None:
    assert not [
        (method.name, key)
        for method in BUILTINS
        for key in method.settings.model_fields
        if key in CHECK_FIELDS
    ]


def keys(position: Position) -> list[str]:
    properties = _check_schema(METHODS["gradeflow/key-point"], position)["properties"]
    return [key for key in properties if key in CHECK_FIELDS]


def test_a_checks_fields_are_its_settings_and_the_keys_a_check_there_may_have() -> None:
    assert keys(Position(QUESTIONS, "Q1")) == ["if", "feedback"]
    assert keys(Position(QUESTIONS, "Q1", in_list=True)) == [
        "name",
        "weight",
        "if",
        "feedback",
    ]
    assert keys(POSITIONS[-1]) == [
        "question",
        "read_only",
        "name",
        "weight",
        "if",
        "feedback",
    ]
    assert keys(POSITIONS[0]) == ["if", "feedback"], "a group's marking reads no question"
    in_group = _check_schema(METHODS["gradeflow/key-point"], POSITIONS[-1])
    assert list(in_group["properties"])[:3] == ["question", "read_only", "point"]
    assert in_group["properties"]["question"]["enum"] == ["Q1", "Q2", "Q3", "Q4"]
    assert "question" in in_group["required"], "a check in a question group needs its question"
    value = _check_schema(
        METHODS["gradeflow/accepted-answers"], Position(QUESTIONS, "Q3", 2, in_list=True)
    )
    assert value["properties"]["value"]["anyOf"][0]["enum"] == [1, 2]
    fixed = Position(QUESTIONS, "Q3", 2, in_list=True, value_fixed=True)
    assert "value" not in _check_schema(METHODS["gradeflow/accepted-answers"], fixed)["properties"]


def test_fields_offer_the_questions_options_and_students_answers() -> None:
    choice = _check_schema(METHODS["gradeflow/multiple-choice"], Position(QUESTIONS, "Q2"))[
        "properties"
    ]
    assert (choice["correct"]["items"]["enum"], choice["correct"]["uniqueItems"]) == (
        ["a", "b"],
        True,
    ), "options are chosen, each once"
    answers = _check_schema(METHODS["gradeflow/accepted-answers"], Position(QUESTIONS, "Q1"))[
        "properties"
    ]
    assert answers["answers"]["x-gradeflow"] == {"accepts_answers": True}
    group = _check_schema(
        METHODS["gradeflow/custom-code-for-a-group"], Position(QUESTIONS, in_group=True)
    )
    assert group["properties"]["questions"]["items"]["enum"] == ["Q1", "Q2", "Q3", "Q4"]


@pytest.mark.parametrize(
    ("method", "field", "takes_answers"),
    [
        ("gradeflow/accepted-answers", "answers", True),
        ("gradeflow/key-points", "points", False),
        ("gradeflow/key-point", "point", False),
    ],
)
def test_only_fields_of_whole_answers_take_them(
    method: str, field: str, takes_answers: bool
) -> None:
    properties = _check_schema(METHODS[method], Position(QUESTIONS, "Q1"))["properties"]
    assert properties[field].get("x-gradeflow", {}).get("accepts_answers", False) == takes_answers


@pytest.mark.parametrize("position", POSITIONS, ids=IDS)
def test_new_checks_are_plain_json_of_their_method(position: Position) -> None:
    for info in compatible_methods(position):
        new = _new_check(info, position, named=False)
        assert json.loads(json.dumps(new)) == new
        assert new["uses"] == info.ref


def test_multiple_values_has_a_check_for_each_value_no_more() -> None:
    checks = _check_schema(METHODS["gradeflow/multiple-values"], Position(QUESTIONS, "Q3"))[
        "properties"
    ]
    assert (checks["checks"]["minItems"], checks["checks"]["maxItems"]) == (2, 2)


def test_a_new_composed_check_starts_with_its_lists() -> None:
    new = _new_check(
        METHODS["gradeflow/answer-sets"], Position(QUESTIONS, in_group=True), named=False
    )
    assert new["with"] == {
        "sets": [{"name": "Expected", "uses": "gradeflow/combination@1", "with": {"checks": []}}]
    }


def test_a_new_multiple_values_check_checks_each_value() -> None:
    new = _new_check(METHODS["gradeflow/multiple-values"], Position(QUESTIONS, "Q3"), named=True)
    assert new["question"] == "Q3"
    assert [check["uses"] for check in new["with"]["checks"]] == [
        "gradeflow/accepted-answers@1",
        "gradeflow/accepted-answers@1",
    ]


def _hints(schema: object) -> Iterator[object]:
    if isinstance(schema, dict):
        if "x-gradeflow" in schema:
            yield schema["x-gradeflow"]
        for value in schema.values():
            yield from _hints(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _hints(value)


@pytest.mark.parametrize("position", POSITIONS, ids=IDS)
def test_every_hint_uses_the_typed_vocabulary(position: Position) -> None:
    for info in compatible_methods(position):
        for hint in _hints(_check_schema(info, position)):
            Hint.model_validate(hint, strict=True)


def test_unknown_methods_are_refused_by_the_draft() -> None:
    draft = MarkingDraft({"uses": "gradeflow/nope@1"}, QUESTIONS, "Q1")
    with pytest.raises(EditingError, match=re.escape("There's no method gradeflow/nope@1.")):
        draft.schema("")


def test_resolved_schema_and_initial_settings_need_no_collection() -> None:
    method = METHODS["gradeflow/key-points"]
    position = Position(QUESTIONS, "Q1")
    assert "points" in _check_schema(method, position)["properties"]
    assert _new_check(method, position, named=True) == {
        "question": "Q1",
        "uses": "gradeflow/key-points@1",
        "with": {},
    }


VALUES = {
    "question": "Q3",
    "uses": "gradeflow/multiple-values@1",
    "with": {
        "checks": [
            {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["x"]}},
            {
                "uses": "gradeflow/not-allowed@1",
                "with": {"check": {"uses": "gradeflow/number-range@1"}},
            },
        ]
    },
}


@pytest.mark.parametrize(
    ("path", "where"),
    [
        ("", ("Q3", None, False, False)),
        ("with.checks.0", ("Q3", 1, True, True)),
        ("with.checks.1.with.check", ("Q3", 2, False, False)),
    ],
)
def test_checks_nest_at_any_depth(path: str, where: tuple[object, ...]) -> None:
    position = MarkingDraft(VALUES, QUESTIONS, "Q3").position(path)
    assert (position.question, position.value, position.in_list, position.value_fixed) == where


def test_a_check_in_a_question_group_reads_its_own_question() -> None:
    group = {
        "uses": "gradeflow/combination@1",
        "with": {"checks": [{"question": "Q4", "uses": "gradeflow/number-range@1"}]},
    }
    draft = MarkingDraft(group, QUESTIONS, None)
    assert (draft.position("").in_group, draft.position("with.checks.0").question) == (True, "Q4")


def test_an_empty_place_for_a_check_is_a_position_too() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/not-allowed@1", "with": {}}
    position = MarkingDraft(marking, QUESTIONS, "Q1").position("with.check")
    assert (position.question, position.in_list) == ("Q1", False)


@pytest.mark.parametrize("path", ["with.nope", "with.checks.5.x", "checks.0", "with.checks.0.with"])
def test_paths_that_arent_checks_are_refused(path: str) -> None:
    with pytest.raises(EditingError, match=re.escape("isn't a check of this marking")):
        MarkingDraft(VALUES, QUESTIONS, "Q3").position(path)


@pytest.mark.parametrize(
    ("text", "problems"),
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
            "question: Q1\nuses: gradeflow/combination@1\nwith:\n  checks: []\n",
            ["Line 4: checks: Add at least one check."],
        ),
        (
            "question: Q1\nuses: gradeflow/combination@1\nwith:\n  checks:\n"
            "  - uses: gradeflow/key-point@1\n    with: {point: a}\n    if: '{{ garbage(( }}'\n",
            ["Line 7: if: Can't read this condition."],
        ),
        (
            "question: Q1\nuses: gradeflow/key-point@1\nwith: {point: a}\nbogus: 1\n",
            ["Line 4: bogus: Not allowed here."],
        ),
        ("question: Q1\nuses: gradeflow/key-point@1\nwith: 5\n", ["Line 3: Must be an object."]),
        (
            "question: Q1\nuses: gradeflow/length@1\nwith: {min_length: 3, max_length: 2}\n",
            ["Line 2: The highest count is below the lowest."],
        ),
        (
            "question: Q1\nuses: gradeflow/combination@1\nwith:\n  checks:\n"
            "  - uses: gradeflow/key-point@1\n    with: {point: a}\n    weigth: 2\n",
            ["Line 7: weigth: Not allowed here."],
        ),
        (
            "question: Q1\nuses: gradeflow/combination@1\nwith:\n  checks:\n"
            "  - uses: gradeflow/multiple-choice@1\n    with: {correct: [a]}\n",
            ["Line 5: Multiple Choice can't mark a text question."],
        ),
        (
            "question: Q1\nuses: gradeflow/combination@1\nwith:\n  checks:\n"
            "  - uses: gradeflow/key-point@1\n    with: {point: a}\n"
            "    if: '{{ checks.later.correct }}'\n"
            "  - id: later\n    uses: gradeflow/key-point@1\n    with: {point: b}\n",
            ["Line 7: if: Key Point must come before this check."],
        ),
    ],
)
def test_a_marking_written_as_yaml_says_whats_wrong_on_its_lines(
    text: str, problems: list[str]
) -> None:
    try:
        found = MarkingDraft.from_yaml(text, QUESTIONS, "Q1").problems
    except InvalidError as error:
        found = error.problems
    assert [str(problem) for problem in found] == problems


def test_a_marking_written_as_yaml_reads_as_its_check() -> None:
    draft = MarkingDraft.from_yaml(
        "question: Q1\nuses: gradeflow/key-point@1\nwith: {point: a}\n", QUESTIONS, "Q1"
    )
    assert draft.data == {"question": "Q1", "uses": "gradeflow/key-point@1", "with": {"point": "a"}}
    assert (draft.finalized().describe(), draft.problems) == ("Includes `a`.", [])


def test_a_drafts_schemas_differ_where_the_checks_positions_do() -> None:
    code = QuestionSet({"Q1": CodeQuestion(reading=CodeReading(guess_indentation=True))})
    custom = METHODS["gradeflow/custom-code"]
    draft = MarkingDraft({"question": "Q1", "uses": custom.ref, "with": {}}, code, "Q1")
    whole = draft.schema("")["properties"]["code"]
    best = METHODS["gradeflow/best-answer"].ref
    draft.replace({"question": "Q1", "uses": best, "with": {"check": {"uses": custom.ref}}})
    one = draft.schema("with.check")["properties"]["code"]
    assert one != whole
    assert one == _check_schema(custom, draft.position("with.check"))["properties"]["code"]
