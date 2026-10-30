"""Feedback students read and descriptions markers read are plain and tidy."""

import re
from typing import Any

import pytest

from gradeflow_core import BUILTINS, Method
from gradeflow_core.common.text import (
    capitalised,
    format_fraction,
    format_number,
    markdown_code,
    markdown_join,
    plain_markdown,
    plain_text,
    plural,
    quoted,
    sentence,
)

UNTIDY = re.compile(r"\.\.|[?!]\.|\s[.,]|\bNone\b(?! of them)|\{\{|\{%|[A-Z]{2,}_[A-Z]+|\S  ")
EXAMPLES = [method for method in BUILTINS if method.examples]


@pytest.mark.parametrize("method", EXAMPLES, ids=lambda method: method.name)
def test_feedback_and_descriptions_are_tidy(method: Method[Any]) -> None:
    for example in method.examples:
        assert not UNTIDY.search(example.description), example.description
        for row in example.rows:
            assert row.expected
            for expected in row.expected.values():
                assert not UNTIDY.search(expected.feedback), expected.feedback


def test_markdown_code_handles_backticks() -> None:
    assert markdown_code("plain") == "`plain`"
    assert markdown_code("uses `backticks`") == "`` uses `backticks` ``"


def test_markdown_join_uses_the_conjunction() -> None:
    assert markdown_join(["a"], conjunction="and") == "`a`"
    assert markdown_join(["a", "b"], conjunction="or") == "`a` or `b`"
    assert markdown_join(["a", "b", "c"], conjunction="and") == "`a`, `b` and `c`"


def test_numbers_are_written_as_people_do() -> None:
    assert [markdown_code(v) for v in (3.0, 1e-06, 12.57, 8800.0, 2)] == [
        "`3`",
        "`0.000001`",
        "`12.57`",
        "`8800`",
        "`2`",
    ]
    assert format_number(-0.5) == "-0.5"
    assert [format_number(v) for v in (6e23, 6.05e23, 1.6e-19)] == ["6e23", "6.05e23", "1.6e-19"]


def test_fractions_are_written_as_people_would() -> None:
    assert [format_fraction(v) for v in (0.5, 0.25, 0.75, 2, 0.3)] == ["½", "¼", "¾", "2", "0.3"]


def test_sentence_adds_a_full_stop_only_when_missing() -> None:
    assert sentence("Match `a`") == "Match `a`."
    assert sentence("Match `a.`") == "Match `a.`"
    assert sentence("Is it `a?`") == "Is it `a?`"
    assert sentence("Done.") == "Done."


def test_a_noun_is_plural_unless_there_is_one() -> None:
    assert [plural("cell", count) for count in (1, 2)] == ["cell", "cells"]


def test_capitalised_raises_only_the_first_letter() -> None:
    assert [capitalised(text) for text in ("line 2, “a” read as “b”", "")] == [
        "Line 2, “a” read as “b”",
        "",
    ]


def test_plain_text_undoes_escapes_and_types_out_formulas() -> None:
    markdown = r"2\. Which state $(r,k)$ costs $\frac{1}{2}$, not \$5?" + "\n\n$$\nx \\le 3\n$$"
    assert plain_text(markdown) == "2. Which state (r,k) costs 1/2, not $5?\n\nx<=3"
    assert plain_text(r"A\_1 is $\text{50\% of \$5}$") == "A_1 is 50% of $5"
    for text in ("costs $5 or $6", "a $ x$ b", "a 2$x$ b", "$x$5"):
        assert plain_text(text) == text


def test_quoting_drops_a_final_full_stop_so_the_sentence_ends_once() -> None:
    assert quoted("Villi increase absorption.") == "“Villi increase absorption”"
    assert quoted("Why?") == "“Why?”"


def test_a_description_reads_as_plain_text_with_code_in_quotes() -> None:
    assert plain_markdown("**A:** Contain `x` or `` a`b ``.") == "A: Contain “x” or “a`b”."
