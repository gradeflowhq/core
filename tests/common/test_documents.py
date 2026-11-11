import pytest

from gradeflow_core import InvalidError, Problem
from gradeflow_core.editing import dump_yaml, load_yaml


def test_text_of_several_lines_is_written_as_a_block() -> None:
    code = "correct = False\nif answer:\n    correct = True\n"
    assert dump_yaml({"code": code, "name": "Q1"}) == (
        "code: |\n  correct = False\n  if answer:\n      correct = True\nname: Q1\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "  indented\nfirst line",
        "trailing  \n\nspaces",
        "no final newline\n",
        "two\n\n",
        "a\tb\nc",
        "windows\r\nline endings\r\n",
        "old mac\rline endings",
    ],
)
def test_text_of_several_lines_reads_back_as_written(text: str) -> None:
    document = {"checks": [{"with": {"code": text}}]}
    assert load_yaml(dump_yaml(document)) == document


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "label: A\nchecks:\n  - uses: a\nchecks: [oops]\n",
            'Line 4: "checks" is written twice.',
        ),
        ("{a: 1, a: 2}", 'Line 1: "a" is written twice.'),
        ("a: 1\n? [b]\n: 2\n", "Line 2: A key can't be a list."),
        ("a: &x 1\nb: *y\n", "Line 2: found undefined alias 'y'."),
        ("[" * 1000 + "]" * 1000, "Line 1: It nests more than 100 levels deep."),
    ],
)
def test_text_that_isnt_yaml_is_refused_at_its_line(text: str, message: str) -> None:
    with pytest.raises(InvalidError) as exc_info:
        load_yaml(text)
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("a: &a [x, x]\nb: &b [*a, *a]\nc: [*b, *b]\n", 2),
        ("a: &a {x: 1}\nb: *a\n", 2),
        ("&a [*a]\n", 1),
        ("a: &a 1\nb: *a\n", 2),
        ("a: &a text\nb: [*a]\n", 2),
    ],
)
def test_aliases_are_refused_at_their_line(text: str, line: int) -> None:
    with pytest.raises(InvalidError) as exc_info:
        load_yaml(text)
    assert exc_info.value.problems == [Problem("Anchors and aliases aren't supported.", line=line)]
    assert str(exc_info.value) == f"Line {line}: Anchors and aliases aren't supported."


def test_a_value_used_twice_is_written_twice_so_it_loads_back() -> None:
    check = {"uses": "gradeflow/accepted-answers@1"}
    data = {"checks": [check, check]}
    assert load_yaml(dump_yaml(data)) == data
