import pytest

from gradeflow_core import GradeFlowError
from gradeflow_core.adapters.errors import adapter_errors


def test_adapter_errors_passes_gradeflow_errors_through() -> None:
    error = GradeFlowError("Row 2 has no student ID.")
    with pytest.raises(GradeFlowError) as caught, adapter_errors():
        raise error
    assert caught.value is error


def test_adapter_errors_say_the_file_could_not_be_read() -> None:
    with pytest.raises(GradeFlowError, match=r"^Couldn't read the file: boom$"), adapter_errors():
        raise RuntimeError("boom")
