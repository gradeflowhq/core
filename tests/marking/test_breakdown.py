from typing import Any

from gradeflow_core import BUILTINS, CheckResult, QuestionResult
from gradeflow_core.editing import dump_yaml
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import CodeReading
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from tests.fake_models import ENDPOINT

from ..methods.helpers import CODE, accepted, code_tests, key_point, mark, mark_all, mark_group
from ..methods.test_best_answer import GUESSING, LONGEST_RUN

DECODE = "def decode(pairs):\n    return ''.join(c * n for c, n in zip(pairs[::2], pairs[1::2]))"


def case(expression: str, expected: str) -> dict[str, str]:
    return {"expression": expression, "expected": expected}


def rows(result: QuestionResult) -> list[tuple[str, str, float | None]]:
    assert result.breakdown is not None
    return [(child.title, child.effect, child.marks) for child in result.breakdown.children]


def details(result: QuestionResult) -> list[dict[str, str]]:
    assert result.breakdown is not None
    return [{d.title: d.value for d in child.details} for child in result.breakdown.children]


def combination(combine: str, *checks: dict[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/combination@1", "with": {"combine": combine, "checks": list(checks)}}


def test_code_tests_break_down_into_their_tests_with_what_each_gave() -> None:
    tests = code_tests(
        case('decode(["a", 2])', "'aa'"), case("decode([])", "'x'"), combine="proportional"
    )
    result = mark(tests, DECODE, CODE)
    assert result.breakdown is not None
    assert result.breakdown.title == "Passes these tests, each counting"
    assert rows(result) == [
        ("Makes `decode([\"a\", 2])` give `'aa'`.", "earned", 0.5),
        ("Makes `decode([])` give `'x'`.", "lost", 0.0),
    ]
    assert details(result) == [
        {"Expression": 'decode(["a", 2])', "Expected": "'aa'", "Output": "'aa'"},
        {"Expression": "decode([])", "Expected": "'x'", "Output": "''"},
    ]


def test_code_that_python_cant_read_is_said_once_not_per_test() -> None:
    tests = code_tests(case("f(1)", "1"), case("f(2)", "2"), combine="proportional")
    result = mark(tests, "def f(x)\n    return x", CODE)
    assert result.feedback == "Couldn't run the code: SyntaxError: expected ':' (line 1)."
    assert result.breakdown is not None
    assert [(d.value, d.kind) for d in result.breakdown.children[0].details][-1] == (
        "SyntaxError: expected ':' (line 1)",
        "error",
    )


def test_a_test_says_what_the_code_printed_when_it_gave_the_wrong_value() -> None:
    tests = code_tests(case("f()", "'hi'"), case("len('ab')", "2"))
    result = mark(tests, "def f():\n    print('hi')", CODE)
    assert result.feedback == "Passed 1 of 2 tests.\nf() gave None. Expected 'hi'. It printed “hi”."
    assert details(result)[0]["Printed"] == "hi"
    assert "Printed" not in details(result)[1]


def test_long_values_are_cut_short() -> None:
    result = mark(code_tests(case("f()", "[]")), "def f():\n    return list(range(5000))", CODE)
    assert result.feedback.startswith("f() gave [0, 1, 2,")
    assert result.feedback.endswith("…. Expected [].")
    assert len(result.feedback) < 250


def test_a_marking_of_one_check_has_no_breakdown() -> None:
    assert mark(accepted("Paris"), "Rome").breakdown is None
    assert mark(code_tests(case("f()", "1")), "def f():\n    return 2", CODE).breakdown is None


def test_each_check_says_what_it_did_to_the_marks() -> None:
    marking = combination(
        "all",
        {"uses": "gradeflow/key-point@1", "with": {"point": "mitochondria", "match": "spelling"}},
        {"uses": "gradeflow/length@1", "with": {"max_length": 5, "count": "words"}},
        {"uses": "gradeflow/not-allowed@1", "with": {"check": key_point("nucleus")}},
    )
    result = mark(marking, "Energy is released in the mitochondrion of the cell")
    assert rows(result) == [
        ("Includes `mitochondria` in similar spelling.", "earned", None),
        ("Has at most `5` words.", "lost", None),
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
    assert rows(result) == [
        ("**Total:** Includes `16`.", "earned", 1.0),
        ("**Wrong unit:** Includes `$`.", "lost", -1.0),
        ("**Working:** Includes `each`.", "lost", 0.0),
    ]
    clean = mark(marking, "16 each", TextQuestion(max_marks=2))
    assert [effect for _, effect, _ in rows(clean)] == ["earned", "uncounted", "earned"]


def test_only_the_best_of_any_one_counts_unless_none_gives_marks() -> None:
    marking = combination(
        "any", accepted("20", name="Expected"), accepted("10", name="Perimeter", weight=0.5)
    )
    assert rows(mark(marking, "10")) == [
        ("**Expected:** Matches `20`.", "uncounted", None),
        ("**Perimeter:** Matches `10`.", "earned", None),
    ]
    assert [effect for _, effect, _ in rows(mark(marking, "7"))] == ["lost", "lost"]


def test_a_conditional_says_what_its_condition_decided() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [accepted("a", question="Q1")],
            "then": [accepted("Paris", question="Q2")],
            "otherwise": [accepted("Rome", question="Q2")],
        },
    }
    questions = {"Q1": TextQuestion(), "Q2": TextQuestion()}
    results = mark_group(marking, {"Q1": "a", "Q2": "Paris"}, questions)
    assert rows(results["Q2"]) == [
        ("**Q1:** Matches `a`.", "yes", None),
        ("Matches `Paris`.", "earned", None),
    ]
    other = mark_group(marking, {"Q1": "b", "Q2": "Paris"}, questions)
    assert rows(other["Q2"]) == [
        ("**Q1:** Matches `a`.", "no", None),
        ("Matches `Rome`.", "lost", None),
    ]


def test_fixed_code_shows_the_tests_of_the_fixed_code() -> None:
    tests = code_tests(case("f()", "1"), case("f() + 1", "2"), fix_small_errors=True)
    assert rows(mark(tests, "def f()\n    return 1", CODE)) == [
        ("Makes `f()` give `1`.", "earned", None),
        ("Makes `f() + 1` give `2`.", "earned", None),
    ]


def test_best_answer_shows_the_possible_answer_that_counted() -> None:
    both = code_tests(case("longest_run([1, 1, 1, 2])", "3"), case("longest_run([])", "0"))
    best = {"uses": "gradeflow/best-answer@1", "with": {"check": both}}
    result = mark(best, LONGEST_RUN, GUESSING)
    assert result.breakdown is not None
    assert result.breakdown.title == "**Best possible answer:** Passes both tests"
    assert [child.effect for child in result.breakdown.children] == ["earned", "earned"]


def test_multiple_values_name_each_value() -> None:
    marking = {
        "uses": "gradeflow/multiple-values@1",
        "with": {"combine": "proportional", "checks": [accepted("2"), accepted("-3")]},
    }
    result = mark(marking, "2, 5", ValuesQuestion(value_types=["number", "number"]))
    assert rows(result) == [
        ("**Value 1:** Matches `2`.", "earned", 0.5),
        ("**Value 2:** Matches `-3`.", "lost", 0.0),
    ]


def test_a_breakdown_round_trips() -> None:
    tests = code_tests(case("f()", "1"), case("f() + 1", "2"))
    result = mark(tests, "def f():\n    return 1", CODE)
    assert QuestionResult.model_validate_json(result.model_dump_json()) == result


def test_a_method_file_without_a_description_is_titled_by_what_it_does() -> None:
    file = {
        "version": 1,
        "label": "Positive",
        "summary": "Tests first_positive.",
        "question_types": ["code"],
        "checks": [code_tests(case("f()", "1"), case("f() + 1", "2"))],
    }
    methods = BUILTINS.with_files([("me/positive", dump_yaml(file))])
    [result] = mark_all({"uses": "me/positive@1"}, ["def f():\n    return 1"], CODE, methods)
    assert result.breakdown is not None
    assert result.breakdown.title == "Passes both tests"


type Tree = list[tuple[str, str, float | None, Tree]]


def tree(result: QuestionResult) -> Tree:
    """The breakdown's checks: each one's title, effect, marks to two places, and its parts."""

    def of(check: CheckResult) -> Tree:
        return [
            (
                child.title,
                child.effect,
                None if child.marks is None else round(child.marks, 2),
                of(child),
            )
            for child in check.children
        ]

    assert result.breakdown is not None
    return of(result.breakdown)


GUESSED = CodeQuestion(max_marks=4, reading=CodeReading(guess_indentation=True))
RUBRIC = {
    "uses": "gradeflow/best-answer@1",
    "with": {
        "check": {
            "uses": "gradeflow/combination@1",
            "with": {
                "combine": "total",
                "checks": [
                    {
                        "weight": 2,
                        **code_tests(
                            case("count([1, 2])", "2"),
                            case("count([])", "0"),
                            case("count([5])", "1"),
                            combine="proportional",
                            fix_small_errors=True,
                            fix_penalty=0.25,
                        ),
                    },
                    {"uses": "gradeflow/code-uses@1", "with": {"looks_for": "recursion"}},
                    {
                        "uses": "gradeflow/not-allowed@1",
                        "with": {
                            "check": {
                                "uses": "gradeflow/code-uses@1",
                                "with": {"looks_for": "loop"},
                            }
                        },
                    },
                ],
            },
        }
    },
}
"""A rubric for a recursive function, its indentation lost: tests worth 2 marks, recursion 1 and
no loop 1."""


def test_a_rubric_inside_best_answer_says_the_marks_of_its_own_checks() -> None:
    flat = "def count(xs):\nif not xs:\nreturn 0\nreturn 1 + count(xs[1:])"
    result = mark(RUBRIC, flat, GUESSED)
    assert result.breakdown is not None
    assert result.breakdown.title == "**Best possible answer:** Marks added up"
    [(title, effect, marks, tests), *others] = tree(result)
    assert (title, effect, marks) == ("Passes these tests, each counting", "earned", 2.0)
    assert tests[0] == ("Makes `count([1, 2])` give `2`.", "earned", None, []), "only the top"
    assert others == [
        ("Uses recursion.", "earned", 1.0, []),
        ("**Not allowed:** Uses a loop.", "earned", 1.0, []),
    ]


def test_a_rubric_shows_the_code_as_written_when_fixing_it_gives_no_more() -> None:
    colon = "def count(xs)\n    if xs == []:\n        return None\n    return 1 + count(xs[1:])"
    result = mark(RUBRIC, colon, GUESSED)
    [(_, _, _, tests), recursion, loop] = tree(result)
    assert {effect for _, effect, _, _ in tests} == {"lost"}
    assert result.breakdown is not None
    assert result.breakdown.children[0].details == (), "nothing fixed"
    assert recursion[:3] == ("Uses recursion.", "unmarked", None)
    assert {d.title: d.value for d in result.breakdown.children[1].details} == {
        "Reason": "Couldn't read the code: expected ':' on line 1."
    }
    assert loop[:3] == ("**Not allowed:** Uses a loop.", "unmarked", None)


def number(key: str, low: float, high: float) -> dict[str, Any]:
    return {
        "question": key,
        "uses": "gradeflow/number-range@1",
        "with": {"min_value": low, "max_value": high},
    }


def answer_set(name: str, *checks: dict[str, Any], weight: float = 1) -> dict[str, Any]:
    return {"name": name, "weight": weight, **combination("all", *checks)}


CARRIED = {
    "uses": "gradeflow/answer-sets@1",
    "with": {
        "sets": [
            answer_set("Expected", number("Q3", 23.9, 24.1), number("Q4", 19.9, 20.1)),
            answer_set(
                "Carried from Q3",
                {
                    "uses": "gradeflow/formula-of-answers@1",
                    "with": {"target": "Q4", "formula": "Q3 - 4"},
                },
                weight=0.5,
            ),
            answer_set("Swapped", number("Q3", 19.9, 20.1), number("Q4", 23.9, 24.1), weight=0.5),
            answer_set("Rounded early", number("Q3", 23, 25), number("Q4", 19, 21), weight=0.5),
            answer_set(
                "Units mixed", number("Q3", 2390, 2410), number("Q4", 199, 201), weight=0.25
            ),
        ]
    },
}
"""Five consistent sets of answers to two numbers, one carrying an error from Q3 to Q4."""


def test_answer_sets_show_each_set_with_its_checks_of_this_question() -> None:
    questions = {"Q3": NumberQuestion(), "Q4": NumberQuestion()}
    carried = mark_group(CARRIED, {"Q3": "30", "Q4": "26"}, questions)["Q4"]
    assert tree(carried)[:2] == [
        ("**Expected**", "uncounted", None, [("Is between `19.9` and `20.1`.", "lost", None, [])]),
        (
            "**Carried from Q3:** **Q4** equals `Q3 - 4`, from the student's own answers.",
            "earned",
            None,
            [],
        ),
    ]
    wrong = mark_group(CARRIED, {"Q3": "7", "Q4": "8"}, questions)["Q4"]
    assert {effect for _, effect, _, _ in tree(wrong)} == {"lost"}


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
                    {"uses": "gradeflow/length@1", "with": {"max_length": 30, "count": "words"}},
                    key_point("animals", name="Says animals photosynthesise", weight=-1),
                ),
            }
        ],
        "otherwise": [accepted("respiration", question="Q2")],
    },
}
"""Q1 says which part the student answered; Q2 is marked for that part."""


def test_a_conditional_shows_its_branch_with_marks_only_at_the_top() -> None:
    questions = {
        "Q1": ChoiceQuestion(options=frozenset({"a", "b"})),
        "Q2": TextQuestion(max_marks=2),
    }
    result = mark_group(PART_A_OR_B, {"Q1": "a", "Q2": "animals make oxygen"}, questions)["Q2"]
    assert tree(result) == [
        ("**Q1:** Chooses exactly `a`.", "yes", None, []),
        (
            "Marks added up",
            "partial",
            None,
            [
                (
                    "Includes `oxygen`, `glucose` and `chlorophyll`, each counting.",
                    "partial",
                    None,
                    [
                        ("Includes `oxygen`.", "earned", None, []),
                        ("Includes `glucose`.", "lost", None, []),
                        ("Includes `chlorophyll`.", "lost", None, []),
                    ],
                ),
                ("Has at most `30` words.", "earned", None, []),
                ("**Says animals photosynthesise:** Includes `animals`.", "lost", None, []),
            ],
        ),
    ]
    other = mark_group(PART_A_OR_B, {"Q1": "b", "Q2": "respiration"}, questions)["Q2"]
    assert tree(other) == [
        ("**Q1:** Chooses exactly `a`.", "no", None, []),
        ("Matches `respiration`.", "earned", None, []),
    ]


def test_answer_criteria_are_named_by_what_they_look_for(fake_models: Any) -> None:
    fake_models(lambda model, answer, criterion: "no" if criterion == "Forgot to sum" else "yes")
    criteria = [
        {"text": "Correct total", "marks": 2},
        {"text": "Shows the working", "marks": 1},
        {"text": "Forgot to sum", "marks": -1},
    ]
    marking = {"uses": "gradeflow/answer-criteria@1", "with": {"criteria": criteria}}
    [result] = mark_all(
        marking,
        ["6 + 10 = 16"],
        TextQuestion(max_marks=3),
        model_endpoint=ENDPOINT.model_copy(update={"decision_model": "decider"}),
    )
    assert rows(result) == [
        ("**Correct total**", "earned", 2.0),
        ("**Shows the working**", "earned", 1.0),
        ("**Forgot to sum**", "uncounted", None),
    ]
