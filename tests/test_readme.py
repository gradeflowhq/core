import re
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fake_models import FakeModels, Oracle, examples_oracle

README = (Path(__file__).parents[1] / "README.md").read_text()
EXAMPLES = re.findall(r"```python\n(.*?)```", README, re.DOTALL)


@pytest.mark.parametrize("code", EXAMPLES, ids=[f"example {n + 1}" for n in range(len(EXAMPLES))])
def test_the_readmes_examples_run(code: str, fake_models: Callable[[Oracle], FakeModels]) -> None:
    fake_models(examples_oracle)  # the Language models example's service
    exec(compile(code, "README.md", "exec"), {})
