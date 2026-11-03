from dataclasses import replace

from pydantic import ConfigDict, Field

from ...common.hints import ADVANCED_HINT, hint
from ...common.text import (
    as_sentence,
    capitalized,
    code_span,
    plain_join,
    plain_markdown,
)
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..logic.combination import COMBINE_LABELS, Combine, proportion_and_parts, weight_notes
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..settings import Settings
from .code_fixes import CodeFixes, best_fixed_outcome, fixes_field
from .code_test import Test, TestSetup, outcome_of_test


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
        json_schema_extra=ADVANCED_HINT,
    )
    fixes: CodeFixes = fixes_field()
    evaluate_expected: bool = Field(
        default=True,
        title="Work out expected values",
        description="Feedback shows what an expected expression gives, such as 5 for 2 + 3.",
        json_schema_extra=ADVANCED_HINT,
    )
    combine: Combine = Field(
        default="all", title="How tests count", json_schema_extra=hint(labels=COMBINE_LABELS)
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
        "share. **Fix code** also tests the code with common errors fixed, such as lost "
        "indentation. Each fix can carry a **Penalty**.\n\n"
        "**Good to know:** Test an ordinary case and an edge case, such as an empty list or "
        "zero. Each test has a time limit, under **Test setup**.\n\n"
        "**Related:** To check how the code is written, not what it does, use Code Uses."
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
            return f"{_weight_prefix(test, combine)}{_test_title(test, negated=combine == 'none')}"
        test_lines = "\n".join(
            f"- {_weight_prefix(test, combine)}{code_span(test.expression)} gives "
            f"{code_span(test.expected)}."
            for test in tests
        )
        return f"Passes {_which_tests(combine, len(tests))}:\n{test_lines}"

    def _mark_answer(
        self, settings: CodeTestsSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, str)  # a code question's answer
        return best_fixed_outcome(
            answer,
            settings.fixes,
            settings.setup.program_for("pass"),
            lambda code, penalty: _tests_outcome(settings, code, question.max_marks, penalty),
        )


def _tests_outcome(
    settings: CodeTestsSettings, code: str, max_marks: float, penalty: float
) -> Outcome:
    """The tests run on ``code``, each test's marks reduced by ``penalty``, combined."""
    weighted = []
    for test in settings.tests:
        outcome = outcome_of_test(
            code, test, settings.setup, evaluate_expected=settings.evaluate_expected
        )
        proportion_after_penalty = outcome.proportion * (1 - penalty)
        weighted.append(
            (
                test.weight,
                replace(outcome, proportion=proportion_after_penalty, title=_test_title(test)),
            )
        )
    combine = settings.combine
    proportion, test_outcomes = proportion_and_parts(combine, weighted, max_marks)
    feedback = _feedback(combine, test_outcomes, proportion)
    if len(test_outcomes) == 1:  # one test is shown as the method itself
        return Outcome(proportion, feedback, parts=test_outcomes, details=test_outcomes[0].details)
    return Outcome(proportion, feedback, parts=test_outcomes, children=test_outcomes)


def _feedback(combine: Combine, test_outcomes: tuple[Outcome, ...], proportion: float) -> str:
    """How many tests passed, and why the others failed. A test passes when it gives marks, even
    marks reduced by the penalty for fixes."""
    if combine == "none":
        passed_titles = [
            as_sentence(plain_markdown(outcome.title))
            for outcome in test_outcomes
            if outcome.proportion > 0
        ]
        return f"Not allowed. {' '.join(passed_titles)}" if passed_titles else CORRECT
    if len(test_outcomes) == 1:
        return test_outcomes[0].feedback
    failed_outcomes = [outcome for outcome in test_outcomes if not outcome.proportion]
    if not failed_outcomes:
        return (
            "Passed both tests."
            if len(test_outcomes) == 2
            else f"Passed all {len(test_outcomes)} tests."
        )
    if (
        len(failed_outcomes) == len(test_outcomes)
        and len({outcome.feedback for outcome in failed_outcomes}) == 1
    ):
        return failed_outcomes[0].feedback  # such as code that couldn't run
    passed_summary = (
        f"Passed {len(test_outcomes) - len(failed_outcomes)} of {len(test_outcomes)} tests."
    )
    if combine == "any" and proportion > 0:
        return passed_summary
    return "\n".join([passed_summary, *_failures(failed_outcomes, len(test_outcomes))])


def _failures(failed_outcomes: list[Outcome], count: int) -> list[str]:
    """Each failed test's feedback; tests the same error stopped share one line."""
    by_error: dict[str, list[Outcome]] = {}
    for outcome in failed_outcomes:
        by_error.setdefault(_error_detail(outcome) or outcome.feedback, []).append(outcome)
    lines = []
    for error, outcomes_with_error in by_error.items():
        if len(outcomes_with_error) == 1:
            lines.append(outcomes_with_error[0].feedback)
        elif len(outcomes_with_error) == count:
            lines.append(as_sentence(f"All {count} failed: {error}"))
        else:
            expressions = plain_join(
                [_detail_value(outcome, "Expression") for outcome in outcomes_with_error],
                conjunction="and",
            )
            lines.append(as_sentence(f"{expressions} failed: {error}"))
    return lines


def _error_detail(outcome: Outcome) -> str | None:
    return next((detail.value for detail in outcome.details if detail.kind == "error"), None)


def _detail_value(outcome: Outcome, title: str) -> str:
    return next(detail.value for detail in outcome.details if detail.title == title)


def _test_title(test: Test, *, negated: bool = False) -> str:
    """What ``test`` checks: ``Makes `add(2, 3)` give `5`.``"""
    makes = "Doesn't make" if negated else "Makes"
    return f"{makes} {code_span(test.expression)} give {code_span(test.expected)}."


def _weight_prefix(test: Test, combine: Combine) -> str:
    return "".join(f"**{capitalized(note)}:** " for note in weight_notes(test.weight, combine))


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
            return "these tests, a share for each"
