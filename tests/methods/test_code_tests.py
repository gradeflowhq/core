import errno
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from gradeflow_core import answer_from_files
from gradeflow_core.methods.code import code_tests as code_tests_module
from gradeflow_core.methods.code import runner

from .helpers import CODE_QUESTION, code_tests, describe, mark, problems_of

F_GIVES_1 = {"expression": "f()", "expected": "1"}
G_GIVES_2 = {"expression": "g()", "expected": "2"}


@pytest.mark.parametrize(("returned", "proportion"), [("1", 1), ("2", 0)])
def test_a_test_compares_the_expression_with_the_expected_value(
    returned: str, proportion: float
) -> None:
    assert (
        mark(code_tests(F_GIVES_1), f"def f():\n    return {returned}", CODE_QUESTION).proportion
        == proportion
    )


@pytest.mark.parametrize(
    ("answer", "feedback"),
    [
        (
            "def f():\n    return undefined_variable",
            "f() failed: NameError: name 'undefined_variable' is not defined. Expected 1.",
        ),
        ("oops(", "Couldn't run the code: SyntaxError: '(' was never closed (line 1)."),
        (
            "def f():\n    x = 1\n  return x",
            "Couldn't run the code: IndentationError: unindent does not match any outer "
            "indentation level (line 3).",
        ),
        (
            "def f():\n    while True:\n        pass",
            "f() failed: it took longer than 1 second. Expected 1.",
        ),
    ],
)
def test_a_test_that_fails_says_why_and_is_still_marked(answer: str, feedback: str) -> None:
    result = mark(code_tests(F_GIVES_1, setup={"time_limit": 1}), answer, CODE_QUESTION)
    assert (result.proportion, result.is_marked, result.feedback) == (0, True, feedback)


def test_feedback_counts_the_tests_passed_and_says_why_the_others_failed() -> None:
    three_tests = [
        F_GIVES_1,
        {"expression": "f() + 1", "expected": "3"},
        {"expression": "f() * 3", "expected": "3"},
    ]
    assert mark(code_tests(*three_tests), "def f():\n    return 1", CODE_QUESTION).feedback == (
        "Passed 2 of 3 tests.\nf() + 1 gave 2. Expected 3."
    )
    assert mark(
        code_tests(*three_tests, marks_by="any_correct"), "def f():\n    return 2", CODE_QUESTION
    ).feedback == ("Passed 1 of 3 tests."), "with Any, failures cost nothing"
    with_colon_fix = code_tests(*three_tests[::2], fixes={"colons": True, "colons_penalty": 0.5})
    feedback = mark(with_colon_fix, "def f()\n    return 1", CODE_QUESTION).feedback
    assert feedback.endswith("after “def f()” (-50%).\nPassed both tests.")


def test_an_error_that_stops_several_tests_is_said_once() -> None:
    first_item_tests = code_tests(
        {"expression": "first([5])", "expected": "5"},
        {"expression": "first([])", "expected": "None"},
        {"expression": "first(list())", "expected": "None"},
    )
    assert mark(first_item_tests, "def first(xs):\n    return xs[0]", CODE_QUESTION).feedback == (
        "Passed 1 of 3 tests.\n"
        "first([]) and first(list()) failed: IndexError: list index out of range."
    )
    assert mark(first_item_tests, "def first(xs):\n    return xs[1]", CODE_QUESTION).feedback == (
        "Passed 0 of 3 tests.\nAll 3 failed: IndexError: list index out of range."
    )


def test_feedback_shows_values_as_python_writes_them() -> None:
    check = code_tests(F_GIVES_1, {"expression": "g()", "expected": "'b'"}, marks_by="proportional")
    assert mark(
        check, "def f():\n    return 1\ndef g():\n    return 'a'", CODE_QUESTION
    ).feedback == ("Passed 1 of 2 tests.\ng() gave 'a'. Expected 'b'.")


@pytest.mark.parametrize(
    ("returned", "expected", "feedback"),
    [
        ("(1, 2)", "(1, 3)", "f() gave (1, 2). Expected (1, 3)."),
        ("{1: 'a'}", "{1: 'b'}", "f() gave {1: 'a'}. Expected {1: 'b'}."),
        ("{1, 2}", "frozenset({1, 2})", "f() gave {1, 2}. Expected frozenset({1, 2})."),
        ("BrokenRepr()", "4", "f() failed: ValueError. Expected 4."),
        ("'x' * 300", "''", f"f() gave '{'x' * 199}…. Expected ''."),
    ],
)
def test_feedback_shows_values_as_the_code_gave_them(
    returned: str, expected: str, feedback: str
) -> None:
    answer = (
        "class BrokenRepr:\n    def __repr__(self):\n        raise ValueError\n"
        f"def f():\n    return {returned}"
    )
    check = code_tests({"expression": "f()", "expected": expected})
    assert mark(check, answer, CODE_QUESTION).feedback == feedback


def test_code_that_cannot_start_is_left_for_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = OSError(errno.EAGAIN, "Resource temporarily unavailable")
    monkeypatch.setattr(runner, "run", Mock(side_effect=unavailable))
    result = mark(code_tests(F_GIVES_1), "def f():\n    return 1", CODE_QUESTION)
    assert (result.is_marked, result.feedback) == (False, "Couldn't mark this answer.")


@pytest.mark.parametrize(
    ("marks_by", "proportion"),
    [("all_correct", 0), ("any_correct", 1), ("total", 1 / 6), ("proportional", 0.5)],
)
def test_marks_by_decides_the_proportion(marks_by: str, proportion: float) -> None:
    two_tests = code_tests(F_GIVES_1, {**F_GIVES_1, "expected": "2"}, marks_by=marks_by)
    assert (
        mark(
            two_tests, "def f():\n    return 1", CODE_QUESTION.model_copy(update={"max_marks": 6})
        ).proportion
        == proportion
    )


EQUALS_ANYTHING = "class A:\n    def __eq__(self, other):\n        return True\n"


@pytest.mark.parametrize(
    ("returned", "expected", "passes"),
    [
        ("1.0", "1", True),
        ("[1, 2.0]", "[1.0, 2]", True),
        ("A()", "5", False),
        ("[A()]", "[5]", False),
        ("{5: A()}", "{5: 5}", False),
        ("True", "1", False),
        ("(1, 2)", "[1, 2]", False),
    ],
)
def test_a_test_passes_only_with_the_types_it_expects(
    returned: str, expected: str, passes: bool
) -> None:
    answer = f"{EQUALS_ANYTHING}def f():\n    return {returned}"
    result = mark(code_tests({"expression": "f()", "expected": expected}), answer, CODE_QUESTION)
    assert result.correct is passes


CLASS_P = (
    "class P:\n"
    "    def __init__(self, x):\n        self.x = x\n"
    "    def __eq__(self, other):\n        return isinstance(other, P) and self.x == other.x\n"
)


def test_a_test_may_expect_a_class_the_setup_defines() -> None:
    check = code_tests({"expression": "f()", "expected": "P(1)"}, setup={"code_before": CLASS_P})
    assert mark(check, "def f():\n    return P(1)", CODE_QUESTION).correct
    assert not mark(check, "def f():\n    return P(2)", CODE_QUESTION).correct


def test_a_test_may_expect_a_class_the_code_defines() -> None:
    check = code_tests({"expression": "P(1)", "expected": "P(1)"})
    assert mark(check, CLASS_P, CODE_QUESTION).correct


@pytest.mark.parametrize(
    ("returned", "expected"),
    [
        ("Counter('aab')", "Counter({'a': 2, 'b': 1})"),
        ("Fraction(2, 4)", "Fraction(1, 2)"),
        ("range(3)", "range(0, 3)"),
        ("Point(1, 2)", "Point(x=1, y=2)"),
    ],
)
def test_a_test_may_expect_any_value_python_compares(returned: str, expected: str) -> None:
    setup = {
        "code_before": "from collections import Counter, namedtuple\n"
        "from fractions import Fraction\nPoint = namedtuple('Point', 'x y')"
    }
    check = code_tests({"expression": "f()", "expected": expected}, setup=setup)
    assert mark(check, f"def f():\n    return {returned}", CODE_QUESTION).correct
    assert not mark(check, "def f():\n    return Counter('xyz')", CODE_QUESTION).correct


@pytest.mark.parametrize(
    ("returned", "feedback"),
    [
        ("a", "f() gave [[...]]. Expected []."),
        ("(x for x in a)", "f() gave <generator object f.<locals>.<genexpr>"),
    ],
)
def test_output_holding_itself_or_a_generator_fails(returned: str, feedback: str) -> None:
    answer = f"def f():\n    a = []\n    a.append(a)\n    return {returned}"
    check = code_tests({"expression": "f()", "expected": "[]"}, setup={"time_limit": 2})
    result = mark(check, answer, CODE_QUESTION)
    assert (result.is_marked, result.correct) == (True, False)
    assert result.feedback.startswith(feedback)


def test_an_expected_value_that_fails_fails_the_test() -> None:
    result = mark(
        code_tests({"expression": "f()", "expected": "Q(1)"}),
        "def f():\n    return 1",
        CODE_QUESTION,
    )
    assert (result.is_marked, result.feedback) == (
        True,
        "f() failed: NameError: name 'Q' is not defined. Expected Q(1).",
    )


def test_expected_values_are_worked_out_inside_indented_setup_code() -> None:
    setup = {"code_before": "def f():", "code_after": "TWO = 2", "indent_spaces": 4}
    check = code_tests({"expression": "f()", "expected": "TWO * 2"}, setup=setup)
    assert mark(check, "return 3", CODE_QUESTION).feedback == "f() gave 3. Expected 4."


def test_a_test_runs_within_its_memory_limit() -> None:
    answer = "def f():\n    data = b'x' * (96 * 1024**2)\n    return 1"
    assert (
        mark(code_tests(F_GIVES_1, setup={"memory_limit": 256}), answer, CODE_QUESTION).proportion
        == 1
    )
    result = mark(code_tests(F_GIVES_1, setup={"memory_limit": 64}), answer, CODE_QUESTION)
    assert (result.proportion, result.feedback) == (0, "f() failed: MemoryError. Expected 1.")


@pytest.mark.parametrize(
    "setup",
    [
        {"indent_spaces": -1},
        {"time_limit": 0},
        {"time_limit": runner.MAX_TIME_LIMIT_S + 1},
        {"memory_limit": runner.MIN_MEMORY_LIMIT_MB - 1},
        {"memory_limit": runner.MAX_MEMORY_LIMIT_MB + 1},
    ],
)
def test_setup_is_bounded(setup: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        code_tests_module.TestSetup.model_validate(setup)


def test_expected_values_are_shown_as_worked_out() -> None:
    check = code_tests(
        {"expression": "f()", "expected": "11 - 9"},
        {"expression": "f() * 2", "expected": "16 // 4"},
    )
    assert mark(check, "def f():\n    return 3", CODE_QUESTION).feedback == (
        "Passed 0 of 2 tests.\nf() gave 3. Expected 2.\nf() * 2 gave 6. Expected 4."
    )


def test_a_test_weight_where_weights_arent_used_is_a_problem() -> None:
    check = code_tests(F_GIVES_1, {**G_GIVES_2, "weight": 2})
    assert problems_of(check, CODE_QUESTION) == [
        ("", "/with/tests/1/weight", "“All must be correct” doesn't use weights.")
    ]
    assert (
        problems_of({**check, "with": {**check["with"], "marks_by": "total"}}, CODE_QUESTION) == []
    )


def test_one_test_is_described_in_a_line() -> None:
    assert describe(code_tests(F_GIVES_1)) == "Makes `f()` give `1`."
    assert (
        describe(code_tests(F_GIVES_1, marks_by="none_correct")) == "Doesn't make `f()` give `1`."
    )


@pytest.mark.parametrize(
    ("marks_by", "opening"),
    [
        ("all_correct", "Passes both tests:"),
        ("any_correct", "Passes any one of these tests:"),
        ("none_correct", "Passes none of these tests:"),
        ("proportional", "Passes these tests, a share for each:"),
        ("total", "Passes these tests, marks added up:"),
    ],
)
def test_several_tests_are_listed_under_how_they_count(marks_by: str, opening: str) -> None:
    assert describe(code_tests(F_GIVES_1, G_GIVES_2, marks_by=marks_by)) == (
        f"{opening}\n- `f()` gives `1`.\n- `g()` gives `2`."
    )
    assert describe(code_tests(F_GIVES_1, G_GIVES_2, F_GIVES_1)).startswith("Passes all 3 tests:\n")


def test_tests_are_listed_with_their_weights() -> None:
    weighted = {**G_GIVES_2, "weight": 2}
    assert describe(code_tests(F_GIVES_1, weighted, marks_by="total")) == (
        "Passes these tests, marks added up:\n- `f()` gives `1`.\n- **Weight 2:** `g()` gives `2`."
    )
    assert describe(
        code_tests(F_GIVES_1, {**G_GIVES_2, "weight": 0.5}, marks_by="any_correct")
    ) == (
        "Passes any one of these tests:\n"
        "- `f()` gives `1`.\n- **50% of the marks:** `g()` gives `2`."
    )
    assert describe(code_tests(weighted, marks_by="total")) == "**Weight 2:** Makes `g()` give `2`."


def test_tests_ignore_surrounding_spaces() -> None:
    spaced = code_tests({"expression": " f() ", "expected": "1\n"})
    assert describe(spaced) == "Makes `f()` give `1`."
    assert (
        mark(spaced, "def f():\n    return 1", CODE_QUESTION).feedback == "f() gave 1, as expected."
    )


def test_tests_found_where_they_are_not_allowed_are_listed_in_the_feedback() -> None:
    check = {"uses": "gradeflow/not-allowed@1", "with": {"check": code_tests(F_GIVES_1, G_GIVES_2)}}
    assert mark(
        check, "def f():\n    return 1\ndef g():\n    return 2", CODE_QUESTION
    ).feedback == ("Not allowed. Passes both tests:\n- “f()” gives “1”.\n- “g()” gives “2”.")


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        ("oops(", "SyntaxError: '(' was never closed (line 1)"),
        ("x = 1", "NameError: name 'f' is not defined"),
    ],
)
def test_code_that_cant_be_tested_gets_no_marks_for_avoiding_the_tests(
    answer: str, error: str
) -> None:
    result = mark(code_tests(F_GIVES_1, marks_by="none_correct"), answer, CODE_QUESTION)
    assert (result.proportion, result.feedback) == (0, f"Couldn't test the code: {error}.")
    not_allowed = {"uses": "gradeflow/not-allowed@1", "with": {"check": code_tests(F_GIVES_1)}}
    result = mark(not_allowed, answer, CODE_QUESTION)
    assert (result.proportion, result.feedback) == (0, f"Couldn't check the answer: {error}.")


ADD_TEST = {"expression": "add(2, 3)", "expected": "5"}
ADD_WITH_TYPOS = "def add(a, b)\n    retrun a + b"
COLON_AND_KEYWORD_FIXES = {"colons": True, "keywords": True}


def test_each_fix_is_listed_with_its_penalty() -> None:
    colon_and_keyword_penalties = {
        **COLON_AND_KEYWORD_FIXES,
        "colons_penalty": 0.25,
        "keywords_penalty": 0.5,
    }
    result = mark(
        code_tests(ADD_TEST, fixes=colon_and_keyword_penalties), ADD_WITH_TYPOS, CODE_QUESTION
    )
    assert result.proportion == 0.25
    assert result.feedback == (
        "Fixed before marking:\n"
        "- Line 1, added “:” after “def add(a, b)” (-25%).\n"
        "- Line 2, “retrun” read as “return” (-50%).\n"
        "add(2, 3) gave 5, as expected."
    )


def test_a_fix_costs_its_penalty_however_the_tests_count() -> None:
    three_tests = [{"expression": f"add({n}, 1)", "expected": str(n + 1)} for n in range(3)]
    fixes = {**COLON_AND_KEYWORD_FIXES, "colons_penalty": 0.25}
    tests = code_tests(*three_tests, marks_by="total", fixes=fixes)
    assert mark(tests, ADD_WITH_TYPOS, CODE_QUESTION).proportion == 0.75


def test_fixes_cost_nothing_unless_a_penalty_is_set() -> None:
    result = mark(
        code_tests(ADD_TEST, fixes=COLON_AND_KEYWORD_FIXES), ADD_WITH_TYPOS, CODE_QUESTION
    )
    assert result.proportion == 1
    assert result.feedback.startswith(
        "Fixed before marking:\n- Line 1, added “:” after “def add(a, b)”.\n"
    )


def test_code_is_fixed_only_as_asked() -> None:
    assert mark(code_tests(ADD_TEST), ADD_WITH_TYPOS, CODE_QUESTION).proportion == 0
    assert (
        mark(code_tests(ADD_TEST, fixes={"colons": True}), ADD_WITH_TYPOS, CODE_QUESTION).proportion
        == 0
    )


def test_code_that_passes_as_written_keeps_its_marks_and_feedback() -> None:
    result = mark(
        code_tests(ADD_TEST, fixes=COLON_AND_KEYWORD_FIXES),
        "def add(a, b):\n    return a + b",
        CODE_QUESTION,
    )
    assert (result.proportion, result.feedback) == (1, "add(2, 3) gave 5, as expected.")


def test_names_the_setup_defines_count_as_defined() -> None:
    setup = {"code_before": "import math\ndef double(x):\n    return 2 * x"}
    check = code_tests({"expression": "f(2)", "expected": "4"}, fixes={"names": True}, setup=setup)
    result = mark(check, "def f(x):\n    return doubel(x)", CODE_QUESTION)
    assert result.proportion == 1
    assert "“doubel” read as “double”" in result.feedback


def test_unindented_code_gets_the_marks_of_its_best_indentation() -> None:
    unindented_code = "def count(xs):\nn = 0\nfor x in xs:\nif x:\nn += 1\nreturn n"
    tests = [{"expression": "count([1, 0, 1])", "expected": "2"}]
    assert mark(code_tests(*tests), unindented_code, CODE_QUESTION).proportion == 0
    indentation_fix_with_penalty = {"indentation": True, "indentation_penalty": 0.1}
    result = mark(
        code_tests(*tests, fixes=indentation_fix_with_penalty), unindented_code, CODE_QUESTION
    )
    assert result.proportion == 0.9
    assert result.feedback.startswith("Fixed before marking:\n- Rebuilt the indentation (-10%).\n")


def test_the_first_file_runs_and_imports_the_others() -> None:
    files = {"main.py": "from helpers import f", "helpers.py": "def f():\n    return 1"}
    assert mark(code_tests(F_GIVES_1), answer_from_files(files), CODE_QUESTION).proportion == 1


def test_a_main_file_that_doesnt_parse_isnt_run() -> None:
    files = {"main.py": "def f(:", "board.py": ""}
    result = mark(code_tests(F_GIVES_1), answer_from_files(files), CODE_QUESTION)
    assert result.feedback == "Couldn't run the code: SyntaxError: invalid syntax (line 1)."


def test_an_imported_file_that_doesnt_parse_is_named() -> None:
    files = {"main.py": "import board\ndef f():\n    return 1", "board.py": "x = ("}
    result = mark(code_tests(F_GIVES_1), answer_from_files(files), CODE_QUESTION)
    assert result.feedback == (
        "f() failed: SyntaxError: '(' was never closed (board.py, line 1). Expected 1."
    )


def test_a_file_never_imported_neednt_parse() -> None:
    files = {"main.py": "def f():\n    return 1", "scratch.py": "x = ("}
    assert mark(code_tests(F_GIVES_1), answer_from_files(files), CODE_QUESTION).proportion == 1


def test_files_that_arent_python_arent_run() -> None:
    files = {"main.py": "def f():\n    return 1", "notes.txt": "Not code ("}
    assert mark(code_tests(F_GIVES_1), answer_from_files(files), CODE_QUESTION).proportion == 1
