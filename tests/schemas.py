from collections.abc import Iterator
from typing import Any


def schema_nodes(schema: object) -> Iterator[dict[str, Any]]:
    """Every object in a JSON schema, outermost first."""
    if isinstance(schema, dict):
        yield schema
        schema = list(schema.values())
    if isinstance(schema, list):
        for value in schema:
            yield from schema_nodes(value)
