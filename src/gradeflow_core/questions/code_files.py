"""Code answers of several files, written as txtar: each file after a ``-- path --`` line."""

import re
from collections.abc import Mapping

from ..common.exceptions import InvalidError
from ..common.problems import Problem
from ..common.text import quoted

_FILE_MARKER = re.compile(r"^-- [^\S\n]*(\S(?:.*\S)?)[^\S\n]* --$\n?", re.MULTILINE)


def answer_from_files(files: Mapping[str, str]) -> str:
    """A code answer of ``files`` by path, the first being the one that runs. Raises
    ``InvalidError`` for an empty or space-padded path, or a line that reads as a file marker."""
    sections = []
    for path, source in files.items():
        if not (path and path == path.strip() and "\n" not in path):
            raise InvalidError([Problem(f"{quoted(path)} isn't a file path.")])
        if marker := _FILE_MARKER.search(source):
            line_number = source.count("\n", 0, marker.start()) + 1
            raise InvalidError([Problem(f"{path} line {line_number} looks like a file marker.")])
        sections.append(f"-- {path} --\n{source.removesuffix('\n')}\n")
    return "".join(sections)


def split_answer(answer: str) -> dict[str, str]:
    """The files of a code answer by path, the one that runs first. An answer that doesn't start
    with a file marker is one file, at path ``""``."""
    comment, *paths_and_sources = _FILE_MARKER.split(answer)
    if comment or not paths_and_sources:
        return {"": answer}
    return dict(zip(paths_and_sources[::2], paths_and_sources[1::2], strict=True))


def join_answer(files: Mapping[str, str]) -> str:
    """The answer ``split_answer`` read ``files`` from."""
    return files[""] if "" in files else answer_from_files(files)


def python_files(files: Mapping[str, str]) -> dict[str, str]:
    """The first of ``files``, which runs, and the other ``.py`` files, which it can import."""
    return {
        path: source
        for index, (path, source) in enumerate(files.items())
        if index == 0 or path.endswith(".py")
    }
