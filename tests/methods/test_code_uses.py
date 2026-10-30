from typing import Any

import pytest

from gradeflow_core import InvalidError

from .helpers import CODE, describe, mark, mark_all, problems_of, proportions


def uses(construct: str, **settings: Any) -> dict[str, Any]:
    return {"uses": "gradeflow/code-uses@1", "with": {"looks_for": construct, **settings}}


@pytest.mark.parametrize(
    ("answer", "proportion"),
    [
        ("total = sum(xs)", 1),
        ("total = builtins.sum(xs)", 1),
        ("total = summary(xs)", 0),
        ("total = 0  # sum(xs) is faster", 0),
        ("text = 'sum(xs)'", 0),
        ("sum = 3", 0),
    ],
)
def test_call_looks_for_a_call_by_name_or_attribute(answer: str, proportion: float) -> None:
    assert mark(uses("call", function="sum"), answer, CODE).proportion == proportion


@pytest.mark.parametrize(
    ("answer", "proportion"),
    [
        ("import numpy", 1),
        ("import numpy as np", 1),
        ("import numpy.linalg", 1),
        ("from numpy import array", 1),
        ("from numpy.linalg import norm", 1),
        ("import numpyish", 0),
        ("from . import numpy", 0),
        ("# import numpy", 0),
    ],
)
def test_import_looks_for_the_module_or_its_submodules(answer: str, proportion: float) -> None:
    assert mark(uses("import", module="numpy"), answer, CODE).proportion == proportion


@pytest.mark.parametrize(
    ("answer", "proportion"),
    [
        ("for x in xs:\n    pass", 1),
        ("async def f():\n    async for x in xs:\n        pass", 1),
        ("while True:\n    break", 1),
        ("xs = [x for x in ys]", 0),
        ("# for x in xs:", 0),
    ],
)
def test_loop_looks_for_for_and_while(answer: str, proportion: float) -> None:
    assert mark(uses("loop"), answer, CODE).proportion == proportion


@pytest.mark.parametrize(
    "answer", ["[x for x in xs]", "{x for x in xs}", "{x: 1 for x in xs}", "sum(x for x in xs)"]
)
def test_comprehension_counts_every_kind(answer: str) -> None:
    assert mark(uses("comprehension"), answer, CODE).proportion == 1


RECURSIVE = "def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)"
METHOD = """
class Tree:
    def size(self):
        return 1 + sum(child.size() for child in self.children)

    def depth(self):
        return 1 + max(self.depth() for _ in self.children)
"""


@pytest.mark.parametrize(
    ("answer", "proportion"),
    [
        (RECURSIVE, 1),
        (METHOD, 1),
        ("def fact(n):\n    r = 1\n    for i in range(n):\n        r *= i + 1\n    return r", 0),
        ("def f():\n    return g()\n\ndef g():\n    return 1", 0),
        ("class T:\n    def size(self):\n        return other.size()", 0),
    ],
)
def test_recursion_looks_for_a_function_calling_itself(answer: str, proportion: float) -> None:
    assert mark(uses("recursion"), answer, CODE).proportion == proportion


def test_indented_code_is_read() -> None:
    assert mark(uses("loop"), "    for x in xs:\n        pass", CODE).proportion == 1


def test_code_that_doesnt_parse_is_left_for_a_person() -> None:
    result = mark(uses("loop"), "x = 1\ny = 2\nfor x in xs\n    pass", CODE)
    assert (result.proportion, result.marked) == (0, False)
    assert result.feedback == "Couldn't read the code: expected ':' on line 3."


UNREADABLE = "Couldn't read the code: invalid syntax on line 1."


def test_not_allowed_says_why_it_left_unreadable_code_for_a_person() -> None:
    negated = {"uses": "gradeflow/not-allowed@1", "with": {"check": uses("loop")}}
    result = mark(negated, "def (", CODE)
    assert (result.feedback, result.marked) == (UNREADABLE, False)


def test_unreadable_code_says_so_over_the_checks_own_feedback() -> None:
    result = mark({**uses("loop"), "feedback": "Own: no loops"}, "def (", CODE)
    assert (result.feedback, result.marked) == (UNREADABLE, False)


@pytest.mark.parametrize(
    ("check", "answers", "feedback"),
    [
        (
            uses("call", function="sum"),
            ["sum(xs)", "len(xs)"],
            ["Calls sum.", "Expected a call to sum."],
        ),
        (
            uses("import", module="numpy"),
            ["import numpy", "import math"],
            ["Imports numpy.", "Expected an import of numpy."],
        ),
        (uses("loop"), ["while x: pass", "x"], ["Uses a loop.", "Expected a loop."]),
        (
            uses("comprehension"),
            ["[x for x in y]", "x"],
            ["Uses a comprehension.", "Expected a comprehension."],
        ),
        (uses("recursion"), [RECURSIVE, "x"], ["Uses recursion.", "Expected recursion."]),
    ],
)
def test_feedback_says_what_was_found(
    check: dict[str, Any], answers: list[str], feedback: list[str]
) -> None:
    results = mark_all(check, answers, CODE)
    assert proportions(results) == [1, 0]
    assert [result.feedback for result in results] == feedback


@pytest.mark.parametrize(
    ("check", "description"),
    [
        (uses("call", function="sum"), "Calls `sum`."),
        (uses("import", module="numpy"), "Imports `numpy`."),
        (uses("loop"), "Uses a loop."),
        (uses("comprehension"), "Uses a comprehension."),
        (uses("recursion"), "Uses recursion."),
    ],
)
def test_it_describes_itself(check: dict[str, Any], description: str) -> None:
    assert describe(check) == description


def test_call_and_import_need_a_name() -> None:
    assert problems_of(uses("call"), CODE) == [("", ("function",), "Name the function.")]
    assert problems_of(uses("import"), CODE) == [("", ("module",), "Name the module.")]
    assert problems_of(uses("loop"), CODE) == []


def test_a_function_is_one_name() -> None:
    with pytest.raises(InvalidError, match="function: "):
        problems_of(uses("call", function="np.sum"), CODE)


def test_code_is_read_as_typed() -> None:
    assert mark(uses("call", function="Counter"), "counts = Counter(words)", CODE).proportion == 1
