from collections.abc import Iterator
from contextlib import contextmanager

from ..common.error_formatting import shortened_reason
from ..common.exceptions import GradeFlowError


@contextmanager
def file_read_errors() -> Iterator[None]:
    """Report failures while reading a file as GradeFlow errors."""
    try:
        yield
    except GradeFlowError:
        raise
    except Exception as error:
        raise GradeFlowError(f"Couldn't read the file: {shortened_reason(str(error))}") from error
