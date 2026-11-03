from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from gradeflow_core.common.schemas import json_schema, type_field


def test_tags_are_fixed_and_read_only() -> None:
    class Tagged(BaseModel):
        method: Literal["A"] = type_field("A")

    assert json_schema(Tagged)["properties"]["method"]["readOnly"] is True
    with pytest.raises(ValidationError):
        Tagged().method = "A"
