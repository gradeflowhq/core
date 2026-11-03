import re
from pathlib import Path

import pytest

from tests.fake_models import InstallFakeModels, example_verdicts

README = (Path(__file__).parents[1] / "README.md").read_text()
EXAMPLES = re.findall(r"```python\n(.*?)```", README, re.DOTALL)


@pytest.mark.parametrize(
    "code", EXAMPLES, ids=[f"example {number}" for number in range(1, len(EXAMPLES) + 1)]
)
def test_the_readmes_examples_run(code: str, fake_models: InstallFakeModels) -> None:
    fake_models(example_verdicts)  # the Language models example's service
    exec(compile(code, "README.md", "exec"), {})
