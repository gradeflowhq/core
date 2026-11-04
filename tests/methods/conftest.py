from collections.abc import Iterator

import pytest

from gradeflow_core.methods.code import runner


@pytest.fixture(autouse=True, scope="session")
def close_code_worker() -> Iterator[None]:
    """Stops the code runner's worker before the session ends, so its coverage is counted."""
    yield
    runner._worker.close()
