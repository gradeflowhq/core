from dataclasses import replace

from pydantic import ConfigDict, Field

from ...common.hints import ADVANCED, hint
from ...common.text import (
    capitalised,
    format_number,
    markdown_code,
    plain_join,
    plain_markdown,
    plural,
    sentence,
)
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..logic.combination import COMBINE_LABELS, Combine, proportion_and_parts, weighting
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..settings import Settings
from .code_test import Test, TestSetup, tested
from .fixes import FixedCode, fixed_code


class CodeTestsSettings(Settings):
    model_config = ConfigDict(title="Code Tests")

    tests: list[Test] = Field(
        min_length=1,
        title="Tests",
        description="An expression to run, and the value it should give.",
    )
    setup: TestSetup = Field(
        default={},
        validate_default=True,
        title="Test setup",
        description="Code before and after the student's, and each test's time limit.",
        json_schema_extra=ADVANCED,
    )
    evaluate_expected: bool = Field(
        default=True,
        title="Work out expected values",
        description="Feedback shows what an expected expression gives, such as 5 for 2 + 3.",
        json_schema_extra=ADVANCED,
    )
    combine: Combine = Field(
        default="all", title="How tests count", json_schema_extra=hint(labels=COMBINE_LABELS)
    )
    fix_small_errors: bool = Field(
        default=False,
        title="Fix small errors",
        description="Code that fails is tested again with slips like a missing colon corrected.",
    )
    fix_penalty: float = Field(
        default=0,
        ge=0,
        le=1,
        title="Marks off for fixes",
        description="The share of the marks fixed code loses, from 0 to 1.",
        json_schema_extra=hint(applies_when={"fix_small_errors": True}),
    )
    max_fixes: int = Field(
        default=2,
        ge=0,
        le=10,
        title="Most fixes",
        description="Code that needs more fixes gets no second try.",
        json_schema_extra=hint(advanced=True, applies_when={"fix_small_errors": True}),
    )


class CodeTests(AnswerMethod[CodeTestsSettings]):
    name = "gradeflow/code-tests"
    version = 1
    label = "Code Tests"
    summary = "Marks for each test the student's code passes."
    guide = (
        "**Use it when:** Students write Python, such as a function, and you want to check what "
        "it does.\n\n"
        "**How it marks:** The student's code runs, then each test runs an expression and "
        "compares what it gives with the expected value, so `add(2, 3)` should give `5`. **How "
        "tests count** sets whether all tests are needed, any one is enough, or each earns its "
        "share. With **Fix small errors**, code that fails is tested again with slips such as a "
        "missing `:` fixed, and loses the share of the marks in **Marks off for fixes**.\n\n"
        "**Good to know:** Test an ordinary case and an edge case, such as an empty list or "
        "zero. Each test has a time limit, under **Test setup**.\n\n"
        "**Related:** When the question guesses lost indentation, put Code Tests inside Best "
        "Answer. To check how the code is written, not what it does, use Code Uses."
    )
    level = "advanced"
    question_types = frozenset({"code"})
    settings = CodeTestsSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        tests, combine = settings.tests, settings.combine
        if len(tests) == 1:
            [test] = tests
            return f"{_weighting(test, combine)}{_test_title(test, negated=combine == 'none')}"
        listed = "\n".join(
            f"- {_weighting(test, combine)}{markdown_code(test.expression)} gives "
            f"{markdown_code(test.expected)}."
            for test in tests
        )
        return f"Passes {_which_tests(combine, len(tests))}:\n{listed}"

    def _mark_answer(
        self, settings: CodeTestsSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, str)  # a code question reading one answer
        written = _run(settings, answer, question.max_marks)
        if not settings.fix_small_errors:
            return written
        fixed = fixed_code(answer, settings.setup.around("pass"), settings.max_fixes)
        if fixed is None:
            return written
        run = _run(settings, fixed.code, question.max_marks, penalty=settings.fix_penalty)
        if run.proportion <= written.proportion:
            return written
        return replace(run, feedback=f"{_fixed_line(fixed, settings.fix_penalty)}\n{run.feedback}")


def _run(settings: CodeTestsSettings, code: str, max_marks: float, penalty: float = 0) -> Outcome:
    """The tests run on ``code``, each test's marks cut by ``penalty``, combined."""
    weighted = []
    for test in settings.tests:
        outcome = tested(code, test, settings.setup, evaluate_expected=settings.evaluate_expected)
        cut = outcome.proportion * (1 - penalty)
        weighted.append((test.weight, replace(outcome, proportion=cut, title=_test_title(test))))
    combine = settings.combine
    proportion, tests = proportion_and_parts(combine, weighted, max_marks)
    said = _feedback(combine, tests, proportion)
    if len(tests) == 1:  # one test is shown as the method itself
        return Outcome(proportion, said, parts=tests, details=tests[0].details)
    return Outcome(proportion, said, parts=tests, children=tests)


def _feedback(combine: Combine, tests: tuple[Outcome, ...], proportion: float) -> str:
    """How many tests passed, and why the others failed. A test passes when it gives marks, even
    marks cut for fixes."""
    if combine == "none":
        found = [sentence(plain_markdown(test.title)) for test in tests if test.proportion > 0]
        return f"Not allowed. {' '.join(found)}" if found else CORRECT
    if len(tests) == 1:
        return tests[0].feedback
    failed = [test for test in tests if not test.proportion]
    if not failed:
        return "Passed both tests." if len(tests) == 2 else f"Passed all {len(tests)} tests."
    if len(failed) == len(tests) and len({test.feedback for test in failed}) == 1:
        return failed[0].feedback  # such as code that couldn't run
    passed = f"Passed {len(tests) - len(failed)} of {len(tests)} tests."
    if combine == "any" and proportion > 0:
        return passed
    return "\n".join([passed, *_failures(failed, len(tests))])


def _failures(failed: list[Outcome], count: int) -> list[str]:
    """Each failed test's feedback; tests the same error stopped share one line."""
    by_error: dict[str, list[Outcome]] = {}
    for test in failed:
        by_error.setdefault(_error(test) or test.feedback, []).append(test)
    lines = []
    for error, stopped in by_error.items():
        if len(stopped) == 1:
            lines.append(stopped[0].feedback)
        elif len(stopped) == count:
            lines.append(sentence(f"All {count} failed: {error}"))
        else:
            expressions = plain_join(
                [_detail(test, "Expression") for test in stopped], conjunction="and"
            )
            lines.append(sentence(f"{expressions} failed: {error}"))
    return lines


def _error(test: Outcome) -> str | None:
    return next((detail.value for detail in test.details if detail.kind == "error"), None)


def _detail(test: Outcome, title: str) -> str:
    return next(detail.value for detail in test.details if detail.title == title)


def _fixed_line(fixed: FixedCode, penalty: float) -> str:
    count = len(fixed.fixes)
    off = f", for {format_number(penalty * 100)}% off" if penalty else ""
    fixes = " ".join(f"{capitalised(fix)}." for fix in fixed.fixes)
    return f"Fixed {count} small {plural('error', count)} before testing{off}. {fixes}"


def _test_title(test: Test, *, negated: bool = False) -> str:
    """What ``test`` checks: ``Makes `add(2, 3)` give `5`.``"""
    makes = "Doesn't make" if negated else "Makes"
    return f"{makes} {markdown_code(test.expression)} give {markdown_code(test.expected)}."


def _weighting(test: Test, combine: Combine) -> str:
    return "".join(f"**{capitalised(note)}:** " for note in weighting(test.weight, combine))


def _which_tests(combine: Combine, count: int) -> str:
    match combine:
        case "all":
            return "both tests" if count == 2 else f"all {count} tests"
        case "any":
            return "any one of these tests"
        case "none":
            return "none of these tests"
        case "total":
            return "these tests, marks added up"
        case "proportional":
            return "these tests, each counting"
