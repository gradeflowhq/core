import pytest
from pydantic import ValidationError

from gradeflow_core.common.hints import Hint
from gradeflow_core.editing import HINT_KEY


def test_hints_leave_out_defaults() -> None:
    assert Hint().schema_extra() == {HINT_KEY: {}}
    assert Hint(advanced=True, labels={"A": "Apple"}).schema_extra() == {
        HINT_KEY: {"advanced": True, "labels": {"A": "Apple"}}
    }


def test_hints_say_what_a_field_holds_beyond_its_schema() -> None:
    hint = Hint(
        kind="markdown", applies_when={"approximate": True}, accepts_answers=True
    ).schema_extra()[HINT_KEY]

    assert Hint.model_validate(hint, strict=True) == Hint(
        kind="markdown", applies_when={"approximate": True}, accepts_answers=True
    )


def test_hints_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        Hint.model_validate({"advance": True})
