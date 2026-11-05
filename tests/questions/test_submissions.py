import pytest
from pydantic import ValidationError

from gradeflow_core import AnswerSheet


@pytest.mark.parametrize("marks", [float("nan"), float("inf"), float("-inf")])
def test_marks_given_are_finite(marks: float) -> None:
    with pytest.raises(ValidationError):
        AnswerSheet(answers={"Q1": "a"}, marks={"Q1": marks})
