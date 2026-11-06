import pytest

from gradeflow_core import InvalidError, answer_from_files
from gradeflow_core.questions.code_files import join_answer, python_files, split_answer

GAME = "from board import Board\n"
BOARD = "class Board:\n    size = 3\n"


def test_an_answer_of_several_files_puts_each_after_its_marker() -> None:
    assert answer_from_files({"game.py": GAME, "pkg/board.py": BOARD}) == (
        f"-- game.py --\n{GAME}-- pkg/board.py --\n{BOARD}"
    )


@pytest.mark.parametrize("source", ["x = 1", ""])
def test_each_file_ends_with_a_newline(source: str) -> None:
    assert answer_from_files({"a.py": source}) == f"-- a.py --\n{source}\n"


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"board.sql": "x\n" * 11 + "-- a --\n"}, "board.sql line 12 looks like a file marker."),
        ({"a.py": "x = 1\n-- b.py --"}, "a.py line 2 looks like a file marker."),
        ({"": "x = 1\n"}, "“” isn't a file path."),
        ({"  ": "x = 1\n"}, "“  ” isn't a file path."),
    ],
)
def test_files_that_wouldnt_read_back_are_refused(files: dict[str, str], message: str) -> None:
    with pytest.raises(InvalidError, match=message):
        answer_from_files(files)


@pytest.mark.parametrize(
    "line", ["--  --", "-- --", "--a.py --", "-- a.py--", " -- a.py --", "x\n-- a.py --"]
)
def test_an_answer_not_starting_with_a_marker_is_one_file(line: str) -> None:
    assert split_answer(line) == {"": line}


def test_an_answer_starting_with_a_marker_is_read_file_by_file() -> None:
    answer = f"-- game.py --\n{GAME}--   pkg/board.py   --\n{BOARD}-- notes.txt --"
    assert split_answer(answer) == {"game.py": GAME, "pkg/board.py": BOARD, "notes.txt": ""}


def test_files_read_back_as_written() -> None:
    files = {"game.py": GAME, "board.py": BOARD, "-- --": "-- not a marker\n"}
    assert split_answer(answer_from_files(files)) == files


def test_the_python_files_are_the_first_and_those_named_py() -> None:
    files = {"main.txt": GAME, "notes.md": "", "board.py": BOARD}
    assert python_files(files) == {"main.txt": GAME, "board.py": BOARD}


@pytest.mark.parametrize("answer", ["x = 1", f"-- game.py --\n{GAME}-- board.py --\n{BOARD}"])
def test_join_answer_writes_back_what_was_read(answer: str) -> None:
    assert join_answer(split_answer(answer)) == answer
