from typing import Any

import pytest

from gradeflow_core import InvalidError, answer_from_files

from .helpers import CODE_QUESTION, describe, mark, mark_all, problems_of, proportions


def code_uses(looks_for: str, **settings: Any) -> dict[str, Any]:
    return {"uses": "gradeflow/code-uses@1", "with": {"looks_for": looks_for, **settings}}


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
    assert mark(code_uses("call", function="sum"), answer, CODE_QUESTION).proportion == proportion


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
    assert mark(code_uses("import", module="numpy"), answer, CODE_QUESTION).proportion == proportion


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
    assert mark(code_uses("loop"), answer, CODE_QUESTION).proportion == proportion


@pytest.mark.parametrize(
    "answer", ["[x for x in xs]", "{x for x in xs}", "{x: 1 for x in xs}", "sum(x for x in xs)"]
)
def test_comprehension_counts_every_kind(answer: str) -> None:
    assert mark(code_uses("comprehension"), answer, CODE_QUESTION).proportion == 1


RECURSIVE_FACTORIAL = "def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)"
RECURSIVE_CLASS_METHODS = """
class Tree:
    def size(self):
        return 1 + sum(child.size() for child in self.children)

    def depth(self):
        return 1 + max(self.depth() for _ in self.children)
"""


@pytest.mark.parametrize(
    ("answer", "proportion"),
    [
        (RECURSIVE_FACTORIAL, 1),
        (RECURSIVE_CLASS_METHODS, 1),
        ("def fact(n):\n    r = 1\n    for i in range(n):\n        r *= i + 1\n    return r", 0),
        ("def f():\n    return g()\n\ndef g():\n    return 1", 0),
        ("class T:\n    def size(self):\n        return other.size()", 0),
    ],
)
def test_recursion_looks_for_a_function_calling_itself(answer: str, proportion: float) -> None:
    assert mark(code_uses("recursion"), answer, CODE_QUESTION).proportion == proportion


def test_indented_code_is_read() -> None:
    assert mark(code_uses("loop"), "    for x in xs:\n        pass", CODE_QUESTION).proportion == 1


def test_code_that_doesnt_parse_gets_no_marks() -> None:
    result = mark(code_uses("loop"), "x = 1\ny = 2\nfor x in xs\n    pass", CODE_QUESTION)
    assert (result.proportion, result.is_marked) == (0, True)
    assert result.feedback == "Couldn't read the code: expected ':' on line 3."


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        ("def (", "SyntaxError: invalid syntax (line 1)"),
        (
            "def f(x):\nreturn x",
            "IndentationError: expected an indented block after function definition on line 1 "
            "(line 2)",
        ),
    ],
)
def test_not_allowed_gives_unreadable_code_no_marks(answer: str, error: str) -> None:
    not_allowed = {"uses": "gradeflow/not-allowed@1", "with": {"check": code_uses("loop")}}
    result = mark(not_allowed, answer, CODE_QUESTION)
    assert (result.proportion, result.feedback) == (0, f"Couldn't check the answer: {error}.")


def test_fixed_code_without_the_construct_passes_not_allowed() -> None:
    check = code_uses("loop", fixes={"indentation": True})
    not_allowed = {"uses": "gradeflow/not-allowed@1", "with": {"check": check}}
    assert mark(not_allowed, "def f(x):\nreturn x", CODE_QUESTION).proportion == 1


@pytest.mark.parametrize(
    ("check", "answers", "feedback"),
    [
        (
            code_uses("call", function="sum"),
            ["sum(xs)", "len(xs)"],
            ["Calls sum.", "Expected a call to sum."],
        ),
        (
            code_uses("import", module="numpy"),
            ["import numpy", "import math"],
            ["Imports numpy.", "Expected an import of numpy."],
        ),
        (code_uses("loop"), ["while x: pass", "x"], ["Uses a loop.", "Expected a loop."]),
        (
            code_uses("comprehension"),
            ["[x for x in y]", "x"],
            ["Uses a comprehension.", "Expected a comprehension."],
        ),
        (
            code_uses("recursion"),
            [RECURSIVE_FACTORIAL, "x"],
            ["Uses recursion.", "Expected recursion."],
        ),
    ],
)
def test_feedback_says_what_was_found(
    check: dict[str, Any], answers: list[str], feedback: list[str]
) -> None:
    results = mark_all(check, answers, CODE_QUESTION)
    assert proportions(results) == [1, 0]
    assert [result.feedback for result in results] == feedback


@pytest.mark.parametrize(
    ("check", "description"),
    [
        (code_uses("call", function="sum"), "Calls `sum`."),
        (code_uses("import", module="numpy"), "Imports `numpy`."),
        (code_uses("loop"), "Uses a loop."),
        (code_uses("comprehension"), "Uses a comprehension."),
        (code_uses("recursion"), "Uses recursion."),
    ],
)
def test_it_describes_itself(check: dict[str, Any], description: str) -> None:
    assert describe(check) == description


def test_call_and_import_need_a_name() -> None:
    assert problems_of(code_uses("call"), CODE_QUESTION) == [
        ("", "/with/function", "Name the function.")
    ]
    assert problems_of(code_uses("import"), CODE_QUESTION) == [
        ("", "/with/module", "Name the module.")
    ]
    assert problems_of(code_uses("loop"), CODE_QUESTION) == []


def test_a_function_is_one_name() -> None:
    with pytest.raises(InvalidError, match="function: "):
        problems_of(code_uses("call", function="np.sum"), CODE_QUESTION)


def test_code_is_read_as_typed() -> None:
    assert (
        mark(
            code_uses("call", function="Counter"), "counts = Counter(words)", CODE_QUESTION
        ).proportion
        == 1
    )


def test_fix_code_lets_unreadable_code_be_checked() -> None:
    unindented_code = "def factorial(n):\nif n < 2:\nreturn 1\nreturn n * factorial(n - 1)"
    without_fixes = mark(code_uses("recursion"), unindented_code, CODE_QUESTION)
    assert without_fixes.feedback.startswith("Couldn't read the code")
    indentation_fix_at_half = {"indentation": True, "indentation_penalty": 0.5}
    result = mark(
        code_uses("recursion", fixes=indentation_fix_at_half), unindented_code, CODE_QUESTION
    )
    assert (result.proportion, result.is_marked) == (0.5, True)
    assert result.feedback == (
        "Fixed before marking:\n- Rebuilt the indentation (-50%).\nUses recursion."
    )


def test_every_python_file_that_parses_is_checked() -> None:
    files = {
        "main.py": "import board",
        "board.py": "for x in xs:\n    pass",
        "scratch.py": "while (",
        "a.txt": "for",
    }
    assert mark(code_uses("loop"), answer_from_files(files), CODE_QUESTION).proportion == 1


def test_a_main_file_that_doesnt_parse_is_named() -> None:
    files = {"main.py": "import board\nfor x in xs", "board.py": "x = 1"}
    answer = answer_from_files(files)
    assert mark(code_uses("loop"), answer, CODE_QUESTION).feedback == (
        "Couldn't read main.py: expected ':' on line 2."
    )
    not_allowed = {"uses": "gradeflow/not-allowed@1", "with": {"check": code_uses("loop")}}
    assert mark(not_allowed, answer, CODE_QUESTION).feedback == (
        "Couldn't check the answer: SyntaxError: expected ':' (main.py, line 2)."
    )
