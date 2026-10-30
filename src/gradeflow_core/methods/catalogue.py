"""The methods core ships: in Python, one file each, or composed in YAML beside them."""

from pathlib import Path
from typing import Any

from .answers.accepted_answers import AcceptedAnswers
from .answers.equivalent_expression import EquivalentExpression
from .answers.formula_of_answers import FormulaOfAnswers
from .answers.multiple_choice import MultipleChoice
from .answers.multiple_values import MultipleValues
from .answers.number_range import NumberRange
from .answers.order import Order
from .code.code_fixes import CodeFixes
from .code.code_test import CodeTest
from .code.code_uses import CodeUses
from .code.custom_code import CustomCode
from .code.custom_code_for_a_group import CustomCodeForAGroup
from .collection import Methods, load_files
from .logic.best_answer import BestAnswer
from .logic.combination import Combination
from .logic.full_marks import FullMarks
from .method import Method
from .names import BUILTIN_NAMESPACE
from .text.answer_criterion import AnswerCriterion
from .text.key_point import KeyPoint
from .text.length import Length
from .text.pattern import Pattern

PYTHON: tuple[Method[Any], ...] = (
    BestAnswer(),
    AcceptedAnswers(),
    EquivalentExpression(),
    NumberRange(),
    MultipleChoice(),
    MultipleValues(),
    Order(),
    Length(),
    FullMarks(),
    Combination(),
    Pattern(),
    KeyPoint(),
    AnswerCriterion(),
    CodeUses(),
    CustomCode(),
    FormulaOfAnswers(),
    CustomCodeForAGroup(),
    CodeTest(),
    CodeFixes(),
)
ORDER = (
    "gradeflow/best-answer",
    "gradeflow/accepted-answers",
    "gradeflow/key-points",
    "gradeflow/answer-criteria",
    "gradeflow/equivalent-expression",
    "gradeflow/number-range",
    "gradeflow/multiple-choice",
    "gradeflow/multiple-values",
    "gradeflow/order",
    "gradeflow/length",
    "gradeflow/full-marks",
    "gradeflow/combination",
    "gradeflow/not-allowed",
    "gradeflow/pattern",
    "gradeflow/code-tests",
    "gradeflow/code-uses",
    "gradeflow/custom-code",
    "gradeflow/conditional",
    "gradeflow/answer-sets",
    "gradeflow/formula-of-answers",
    "gradeflow/custom-code-for-a-group",
    "gradeflow/key-point",
    "gradeflow/answer-criterion",
    "gradeflow/code-test",
    "gradeflow/code-fixes",
)
"""Every built-in method's name, in catalogue order: basic methods, then advanced ones, then those
for question groups. A name no Python method has is composed from its YAML file."""
FILES = tuple(
    path
    for path in sorted(Path(__file__).parent.glob("*/*.yaml"))
    if not path.name.endswith(".examples.yaml")
)


def _builtins() -> Methods:
    files = [(f"{BUILTIN_NAMESPACE}/{path.stem}", path.read_text()) for path in FILES]
    methods = load_files(files, Methods(PYTHON))
    return Methods(sorted(methods, key=lambda method: (ORDER.index(method.name), method.version)))


BUILTINS = _builtins()
