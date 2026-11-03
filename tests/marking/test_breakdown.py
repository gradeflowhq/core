from typing import Any

from gradeflow_core import BUILTINS, CheckResult, Detail, QuestionResult
from gradeflow_core.editing import dump_yaml
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from tests.fake_models import ENDPOINT, InstallFakeModels

from ..methods.helpers import (
    CODE_QUESTION,
    accepted_answer,
    code_tests,
    key_point,
    mark,
    mark_all,
    mark_group,
)

DECODE = "def decode(pairs):\n    return ''.join(c * n for c, n in zip(pairs[::2], pairs[1::2]))"


def code_test(expression: str, expected: str) -> dict[str, str]:
    return {"expression": expression, "expected": expected}


def breakdown_rows(result: QuestionResult) -> list[tuple[str, str, float | None]]:
    assert result.breakdown is not None
    return [(child.title, child.effect, child.marks) for child in result.breakdown.children]


def breakdown_details(result: QuestionResult) -> list[dict[str, str]]:
    assert result.breakdown is not None
    return [
        {detail.title: detail.value for detail in child.details}
        for child in result.breakdown.children
    ]


def combination(combine: str, *checks: dict[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/combination@1", "with": {"combine": combine, "checks": list(checks)}}


def test_code_tests_break_down_into_their_tests_with_what_each_gave() -> None:
    tests = code_tests(
        code_test('decode(["a", 2])', "'aa'"),
        code_test("decode([])", "'x'"),
        combine="proportional",
    )
    result = mark(tests, DECODE, CODE_QUESTION)
    assert result.breakdown is not None
    assert result.breakdown.title == "Passes these tests, a share for each"
    assert breakdown_rows(result) == [
        ("Makes `decode([\"a\", 2])` give `'aa'`.", "earned", 0.5),
        ("Makes `decode([])` give `'x'`.", "no_marks", 0.0),
    ]
    assert breakdown_details(result) == [
        {"Expression": 'decode(["a", 2])', "Expected": "'aa'", "Output": "'aa'"},
        {"Expression": "decode([])", "Expected": "'x'", "Output": "''"},
    ]


def test_code_that_python_cant_read_is_said_once_not_per_test() -> None:
    tests = code_tests(code_test("f(1)", "1"), code_test("f(2)", "2"), combine="proportional")
    result = mark(tests, "def f(x)\n    return x", CODE_QUESTION)
    assert result.feedback == "Couldn't run the code: SyntaxError: expected ':' (line 1)."
    assert result.breakdown is not None
    assert [(detail.value, detail.kind) for detail in result.breakdown.children[0].details][-1] == (
        "SyntaxError: expected ':' (line 1)",
        "error",
    )


def test_a_test_says_what_the_code_printed_when_it_gave_the_wrong_value() -> None:
    tests = code_tests(code_test("f()", "'hi'"), code_test("len('ab')", "2"))
    result = mark(tests, "def f():\n    print('hi')", CODE_QUESTION)
    assert result.feedback == "Passed 1 of 2 tests.\nf() gave None. Expected 'hi'. It printed “hi”."
    assert breakdown_details(result)[0]["Printed"] == "hi"
    assert "Printed" not in breakdown_details(result)[1]


def test_long_values_are_shortened() -> None:
    result = mark(
        code_tests(code_test("f()", "[]")), "def f():\n    return list(range(5000))", CODE_QUESTION
    )
    assert result.feedback.startswith("f() gave [0, 1, 2,")
    assert result.feedback.endswith("…. Expected [].")
    assert len(result.feedback) < 250


def test_a_marking_of_one_check_has_no_breakdown() -> None:
    assert mark(accepted_answer("Paris"), "Rome").breakdown is None
    assert (
        mark(code_tests(code_test("f()", "1")), "def f():\n    return 2", CODE_QUESTION).breakdown
        is None
    )


def test_each_check_says_what_it_did_to_the_marks() -> None:
    marking = combination(
        "all",
        {"uses": "gradeflow/key-point@1", "with": {"point": "mitochondria", "match": "spelling"}},
        {"uses": "gradeflow/length@1", "with": {"max_length": 5, "unit": "words"}},
        {"uses": "gradeflow/not-allowed@1", "with": {"check": key_point("nucleus")}},
    )
    result = mark(marking, "Energy is released in the mitochondrion of the cell")
    assert breakdown_rows(result) == [
        ("Includes `mitochondria` in similar spelling.", "earned", None),
        ("Has at most `5` words.", "no_marks", None),
        ("**Not allowed:** Includes `nucleus`.", "earned", None),
    ]


def test_a_penalty_that_finds_what_it_checks_loses_marks() -> None:
    marking = combination(
        "total",
        key_point("16", name="Total"),
        key_point("$", name="Wrong unit", weight=-1),
        key_point("each", name="Working", weight=0.5),
    )
    result = mark(marking, "$16", TextQuestion(max_marks=2))
    assert breakdown_rows(result) == [
        ("**Total:** Includes `16`.", "earned", 1.0),
        ("**Wrong unit:** Includes `$`.", "penalty", -1.0),
        ("**Working:** Includes `each`.", "no_marks", 0.0),
    ]
    without_penalty = mark(marking, "16 each", TextQuestion(max_marks=2))
    assert [effect for _, effect, _ in breakdown_rows(without_penalty)] == [
        "earned",
        "uncounted",
        "earned",
    ]


def test_only_the_best_of_any_one_counts_unless_none_gives_marks() -> None:
    marking = combination(
        "any",
        accepted_answer("20", name="Expected"),
        accepted_answer("10", name="Perimeter", weight=0.5),
    )
    assert breakdown_rows(mark(marking, "10")) == [
        ("**Expected:** Matches `20`.", "uncounted", None),
        ("**Perimeter:** Matches `10`.", "earned", None),
    ]
    assert [effect for _, effect, _ in breakdown_rows(mark(marking, "7"))] == [
        "no_marks",
        "no_marks",
    ]


def test_a_conditional_says_what_its_condition_decided() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [accepted_answer("a", question="Q1")],
            "then": [accepted_answer("Paris", question="Q2")],
            "otherwise": [accepted_answer("Rome", question="Q2")],
        },
    }
    questions = {"Q1": TextQuestion(), "Q2": TextQuestion()}
    results = mark_group(marking, {"Q1": "a", "Q2": "Paris"}, questions)
    assert breakdown_rows(results["Q2"]) == [
        ("**Q1:** Matches `a`.", "met", None),
        ("Matches `Paris`.", "earned", None),
    ]
    otherwise = mark_group(marking, {"Q1": "b", "Q2": "Paris"}, questions)
    assert breakdown_rows(otherwise["Q2"]) == [
        ("**Q1:** Matches `a`.", "not_met", None),
        ("Matches `Rome`.", "no_marks", None),
    ]


def test_fixed_code_shows_in_the_breakdown() -> None:
    tests = code_tests(code_test("f()", "1"), code_test("f() + 1", "2"), fixes={"colons": True})
    result = mark(tests, "def f()\n    return 1", CODE_QUESTION)
    assert result.breakdown is not None
    assert result.breakdown.details == (Detail("Marked as", "def f():\n    return 1", "code"),)
    assert breakdown_rows(result) == [
        ("Makes `f()` give `1`.", "earned", None),
        ("Makes `f() + 1` give `2`.", "earned", None),
    ]


def test_multiple_values_name_each_value() -> None:
    marking = {
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "combine": "proportional",
            "checks": [accepted_answer("2"), accepted_answer("-3")],
        },
    }
    result = mark(marking, "2, 5", ValuesQuestion(value_types=["number", "number"]))
    assert breakdown_rows(result) == [
        ("**Value 1:** Matches `2`.", "earned", 0.5),
        ("**Value 2:** Matches `-3`.", "no_marks", 0.0),
    ]


def test_a_breakdown_round_trips() -> None:
    tests = code_tests(code_test("f()", "1"), code_test("f() + 1", "2"))
    result = mark(tests, "def f():\n    return 1", CODE_QUESTION)
    assert QuestionResult.model_validate_json(result.model_dump_json()) == result


def test_a_method_file_without_a_description_is_titled_by_what_it_does() -> None:
    method_file = {
        "version": 1,
        "label": "Positive",
        "summary": "Tests first_positive.",
        "question_types": ["code"],
        "checks": [code_tests(code_test("f()", "1"), code_test("f() + 1", "2"))],
    }
    methods = BUILTINS.with_files([("me/positive", dump_yaml(method_file))])
    [result] = mark_all(
        {"uses": "me/positive@1"}, ["def f():\n    return 1"], CODE_QUESTION, methods
    )
    assert result.breakdown is not None
    assert result.breakdown.title == "Passes both tests"


type BreakdownTree = list[tuple[str, str, float | None, BreakdownTree]]


def breakdown_tree(result: QuestionResult) -> BreakdownTree:
    """The breakdown's checks: each one's title, effect, marks to two places, and its parts."""

    def subtree(check: CheckResult) -> BreakdownTree:
        return [
            (
                child.title,
                child.effect,
                None if child.marks is None else round(child.marks, 2),
                subtree(child),
            )
            for child in check.children
        ]

    assert result.breakdown is not None
    return subtree(result.breakdown)


REBUILD_INDENTATION = {"indentation": True}
COUNT_MARKING = combination(
    "total",
    {
        "weight": 2,
        **code_tests(
            code_test("count([1, 2])", "2"),
            code_test("count([])", "0"),
            code_test("count([5])", "1"),
            combine="proportional",
            fixes={**REBUILD_INDENTATION, "colons": True, "colons_penalty": 0.25},
        ),
    },
    {
        "uses": "gradeflow/code-uses@1",
        "with": {"looks_for": "recursion", "fixes": REBUILD_INDENTATION},
    },
    {
        "uses": "gradeflow/not-allowed@1",
        "with": {
            "check": {
                "uses": "gradeflow/code-uses@1",
                "with": {"looks_for": "loop", "fixes": REBUILD_INDENTATION},
            }
        },
    },
)
"""A marking of a recursive function, its indentation lost: tests worth 2 marks, recursion 1 and
no loop 1."""
COUNT_QUESTION = CodeQuestion(max_marks=4)


def test_a_marking_of_unindented_code_gives_each_checks_marks() -> None:
    unindented = "def count(xs):\nif not xs:\nreturn 0\nreturn 1 + count(xs[1:])"
    result = mark(COUNT_MARKING, unindented, COUNT_QUESTION)
    assert result.breakdown is not None
    assert result.breakdown.title == "Total of these"
    [(title, effect, marks, tests), *others] = breakdown_tree(result)
    assert (title, effect, marks) == (
        "Passes these tests, a share for each",
        "earned",
        2.0,
    )
    assert tests[0] == ("Makes `count([1, 2])` give `2`.", "earned", None, []), "only the top"
    assert others == [
        ("Uses recursion.", "earned", 1.0, []),
        ("**Not allowed:** Uses a loop.", "earned", 1.0, []),
    ]
    [marked_as] = result.breakdown.children[0].details
    assert marked_as.value.splitlines()[1] == "    if not xs:"


def test_a_marking_shows_the_code_as_written_when_fixing_it_gives_no_more() -> None:
    missing_colon = (
        "def count(xs)\n    if xs == []:\n        return None\n    return 1 + count(xs[1:])"
    )
    result = mark(COUNT_MARKING, missing_colon, COUNT_QUESTION)
    [(_, _, _, tests), recursion, loop] = breakdown_tree(result)
    assert {effect for _, effect, _, _ in tests} == {"no_marks"}
    assert result.breakdown is not None
    assert result.breakdown.children[0].details == (), "nothing fixed"
    assert recursion[:3] == ("Uses recursion.", "unmarked", None)
    assert {detail.title: detail.value for detail in result.breakdown.children[1].details} == {
        "Reason": "Couldn't read the code: expected ':' on line 1."
    }
    assert loop[:3] == ("**Not allowed:** Uses a loop.", "unmarked", None)
    assert result.breakdown.details == (), "the result's feedback says why"


def number_range(question: str, min_value: float, max_value: float) -> dict[str, Any]:
    return {
        "question": question,
        "uses": "gradeflow/number-range@1",
        "with": {"min_value": min_value, "max_value": max_value},
    }


def answer_set(name: str, *checks: dict[str, Any], weight: float = 1) -> dict[str, Any]:
    return {"name": name, "weight": weight, **combination("all", *checks)}


ANSWER_SETS_WITH_CARRIED_ERROR = {
    "uses": "gradeflow/answer-sets@1",
    "with": {
        "sets": [
            answer_set("Expected", number_range("Q3", 23.9, 24.1), number_range("Q4", 19.9, 20.1)),
            answer_set(
                "Carried from Q3",
                {
                    "uses": "gradeflow/formula-of-answers@1",
                    "with": {"question_to_mark": "Q4", "formula": "Q3 - 4"},
                },
                weight=0.5,
            ),
            answer_set(
                "Swapped",
                number_range("Q3", 19.9, 20.1),
                number_range("Q4", 23.9, 24.1),
                weight=0.5,
            ),
            answer_set(
                "Rounded early", number_range("Q3", 23, 25), number_range("Q4", 19, 21), weight=0.5
            ),
            answer_set(
                "Units mixed",
                number_range("Q3", 2390, 2410),
                number_range("Q4", 199, 201),
                weight=0.25,
            ),
        ]
    },
}
"""Five consistent sets of answers to two numbers, one carrying an error from Q3 to Q4."""


def test_answer_sets_show_each_set_with_its_checks_of_this_question() -> None:
    questions = {"Q3": NumberQuestion(), "Q4": NumberQuestion()}
    carried = mark_group(ANSWER_SETS_WITH_CARRIED_ERROR, {"Q3": "30", "Q4": "26"}, questions)["Q4"]
    assert breakdown_tree(carried)[:2] == [
        (
            "**Expected**",
            "uncounted",
            None,
            [("Is between `19.9` and `20.1`.", "no_marks", None, [])],
        ),
        (
            "**Carried from Q3:** **Q4** equals `Q3 - 4`, from the student's own answers.",
            "earned",
            None,
            [],
        ),
    ]
    wrong = mark_group(ANSWER_SETS_WITH_CARRIED_ERROR, {"Q3": "7", "Q4": "8"}, questions)["Q4"]
    assert {effect for _, effect, _, _ in breakdown_tree(wrong)} == {"no_marks"}


PART_A_OR_B = {
    "uses": "gradeflow/conditional@1",
    "with": {
        "condition": [
            {"question": "Q1", "uses": "gradeflow/multiple-choice@1", "with": {"correct": ["a"]}}
        ],
        "then": [
            {
                "question": "Q2",
                **combination(
                    "total",
                    {
                        "uses": "gradeflow/key-points@1",
                        "with": {"points": ["oxygen", "glucose", "chlorophyll"]},
                    },
                    {"uses": "gradeflow/length@1", "with": {"max_length": 30, "unit": "words"}},
                    key_point("animals", name="Says animals photosynthesise", weight=-1),
                ),
            }
        ],
        "otherwise": [accepted_answer("respiration", question="Q2")],
    },
}
"""Q1 says which part the student answered; Q2 is marked for that part."""


def test_a_conditional_shows_its_branch_with_marks_only_at_the_top() -> None:
    questions = {
        "Q1": ChoiceQuestion(options=frozenset({"a", "b"})),
        "Q2": TextQuestion(max_marks=2),
    }
    result = mark_group(PART_A_OR_B, {"Q1": "a", "Q2": "animals make oxygen"}, questions)["Q2"]
    assert breakdown_tree(result) == [
        ("**Q1:** Chooses exactly `a`.", "met", None, []),
        (
            "Total of these",
            "partial",
            None,
            [
                (
                    "Includes `oxygen`, `glucose` and `chlorophyll`, a share for each.",
                    "partial",
                    None,
                    [
                        ("Includes `oxygen`.", "earned", None, []),
                        ("Includes `glucose`.", "no_marks", None, []),
                        ("Includes `chlorophyll`.", "no_marks", None, []),
                    ],
                ),
                ("Has at most `30` words.", "earned", None, []),
                ("**Says animals photosynthesise:** Includes `animals`.", "penalty", None, []),
            ],
        ),
    ]
    otherwise = mark_group(PART_A_OR_B, {"Q1": "b", "Q2": "respiration"}, questions)["Q2"]
    assert breakdown_tree(otherwise) == [
        ("**Q1:** Chooses exactly `a`.", "not_met", None, []),
        ("Matches `respiration`.", "earned", None, []),
    ]


def test_answer_criteria_are_named_by_what_they_look_for(fake_models: InstallFakeModels) -> None:
    fake_models(lambda model, answer, criterion: "no" if criterion == "Forgot to sum" else "yes")
    criteria = [
        {"criterion": "Correct total", "marks": 2},
        {"criterion": "Shows the working", "marks": 1},
        {"criterion": "Forgot to sum", "marks": -1},
    ]
    marking = {"uses": "gradeflow/answer-criteria@1", "with": {"criteria": criteria}}
    [result] = mark_all(
        marking,
        ["6 + 10 = 16"],
        TextQuestion(max_marks=3),
        model_endpoint=ENDPOINT.model_copy(update={"fast_model": "decider"}),
    )
    assert breakdown_rows(result) == [
        ("**Correct total**", "earned", 2.0),
        ("**Shows the working**", "earned", 1.0),
        ("**Forgot to sum**", "uncounted", None),
    ]
