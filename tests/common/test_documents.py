import pytest

from gradeflow_core.common.documents import DocumentError
from gradeflow_core.editing import dump_yaml, load_yaml


def test_text_of_several_lines_is_written_as_a_block() -> None:
    code = "correct = False\nif answer:\n    correct = True\n"
    assert dump_yaml({"code": code, "name": "Q1"}) == (
        "code: |\n  correct = False\n  if answer:\n      correct = True\nname: Q1\n"
    )


@pytest.mark.parametrize(
    "text",
    ["  indented\nfirst line", "trailing  \n\nspaces", "no final newline\n", "two\n\n", "a\tb\nc"],
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
        ("a: &x 1\nb: *y\n", "Line 2: found undefined alias 'y'."),
    ],
)
def test_text_that_isnt_yaml_is_refused_at_its_line(text: str, message: str) -> None:
    with pytest.raises(DocumentError) as caught:
        load_yaml(text)
    assert str(caught.value) == message
