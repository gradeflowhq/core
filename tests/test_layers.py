"""Dependencies point down: each package imports only itself and the packages before it."""

import ast
from pathlib import Path

import gradeflow_core

LAYERS = ("common", "models", "questions", "steps", "methods", "marking", "adapters", "editing")
PACKAGE = Path(gradeflow_core.__file__).parent


def imported_packages(path: Path) -> set[str]:
    """The core packages the module at ``path`` imports from."""
    folders = path.relative_to(PACKAGE).parent.parts
    return {
        [*folders[: len(folders) - node.level + 1], *(node.module or "").split(".")][0]
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level
    }


def test_every_package_has_a_layer() -> None:
    assert {path.parent.name for path in PACKAGE.glob("*/__init__.py")} == set(LAYERS)


def test_each_package_imports_only_itself_and_the_packages_before_it() -> None:
    imports_from_above = [
        f"{path.relative_to(PACKAGE)}: {package}"
        for layer, name in enumerate(LAYERS)
        for path in (PACKAGE / name).rglob("*.py")
        for package in imported_packages(path)
        if package not in LAYERS[: layer + 1]
    ]
    assert not imports_from_above
