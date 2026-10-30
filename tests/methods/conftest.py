from collections.abc import Iterator

import pytest

from gradeflow_core.methods.code import runner


@pytest.fixture(autouse=True, scope="session")
def close_code_workers() -> Iterator[None]:
    """Stops the code runner's workers before the session ends, so their coverage is counted."""
    yield
    runner._close_workers()
