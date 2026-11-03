"""Feedback students read and descriptions markers read are plain and tidy."""

import re
from typing import Any

import pytest

from gradeflow_core import BUILTINS, Method
from gradeflow_core.common.text import (
    as_sentence,
    capitalized,
    code_join,
    code_span,
    escaped_markdown,
    format_fraction,
    format_number,
    plain_markdown,
    plural,
    quoted,
    reads_better_listed,
    unescaped_markdown,
)

UNTIDY_TEXT = re.compile(r"\.\.|[?!]\.|\s[.,]|\bNone\b(?! of them)|\{\{|\{%|[A-Z]{2,}_[A-Z]+|\S  ")
METHODS_WITH_EXAMPLES = [method for method in BUILTINS if method.examples]


@pytest.mark.parametrize("method", METHODS_WITH_EXAMPLES, ids=lambda method: method.name)
def test_feedback_and_descriptions_are_tidy(method: Method[Any]) -> None:
    for example in method.examples:
        assert not UNTIDY_TEXT.search(example.description), example.description
        for row in example.rows:
            assert row.expected
            for expected in row.expected.values():
                assert not UNTIDY_TEXT.search(expected.feedback), expected.feedback


def test_code_span_handles_backticks() -> None:
    assert code_span("plain") == "`plain`"
    assert code_span("uses `backticks`") == "`` uses `backticks` ``"


def test_code_join_uses_the_conjunction() -> None:
    assert code_join(["a"], conjunction="and") == "`a`"
    assert code_join(["a", "b"], conjunction="or") == "`a` or `b`"
    assert code_join(["a", "b", "c"], conjunction="and") == "`a`, `b` and `c`"


def test_numbers_are_written_as_people_do() -> None:
    assert [code_span(number) for number in (3.0, 1e-06, 12.57, 8800.0, 2)] == [
        "`3`",
        "`0.000001`",
        "`12.57`",
        "`8800`",
        "`2`",
    ]
    assert format_number(-0.5) == "-0.5"
    assert [format_number(number) for number in (6e23, 6.05e23, 1.6e-19)] == [
        "6e23",
        "6.05e23",
        "1.6e-19",
    ]


def test_fractions_are_written_as_people_would() -> None:
    assert [format_fraction(number) for number in (0.5, 0.25, 0.75, 2, 0.3)] == [
        "½",
        "¼",
        "¾",
        "2",
        "0.3",
    ]


def test_as_sentence_adds_a_full_stop_only_when_missing() -> None:
    assert as_sentence("Match `a`") == "Match `a`."
    assert as_sentence("Match `a.`") == "Match `a.`"
    assert as_sentence("Is it `a?`") == "Is it `a?`"
    assert as_sentence("Done.") == "Done."


def test_a_noun_is_plural_unless_there_is_one() -> None:
    assert [plural("cell", count) for count in (1, 2)] == ["cell", "cells"]


def test_capitalized_raises_only_the_first_letter() -> None:
    assert [capitalized(text) for text in ("line 2, “a” read as “b”", "")] == [
        "Line 2, “a” read as “b”",
        "",
    ]


def test_unescaped_markdown_undoes_escapes_and_types_out_formulas() -> None:
    markdown = r"2\. Which state $(r,k)$ costs $\frac{1}{2}$, not \$5?" + "\n\n$$\nx \\le 3\n$$"
    assert unescaped_markdown(markdown) == "2. Which state (r,k) costs 1/2, not $5?\n\nx<=3"
    assert unescaped_markdown(r"A\_1 is $\text{50\% of \$5}$") == "A_1 is 50% of $5"


@pytest.mark.parametrize("text", ["costs $5 or $6", "a $ x$ b", "a 2$x$ b", "$x$5"])
def test_unescaped_markdown_keeps_dollars_around_no_formula(text: str) -> None:
    assert unescaped_markdown(text) == text


def test_quoting_drops_a_final_full_stop_so_the_sentence_ends_once() -> None:
    assert quoted("Villi increase absorption.") == "“Villi increase absorption”"
    assert quoted("Why?") == "“Why?”"


def test_a_description_reads_as_plain_text_with_code_in_quotes() -> None:
    assert plain_markdown("**A:** Contain `x` or `` a`b ``.") == "A: Contain “x” or “a`b”."
    assert plain_markdown(r"Matches `\(\d+\)`, `a**b` and `$x$`, not \*.") == (
        r"Matches “\(\d+\)”, “a**b” and “$x$”, not *."
    )


def test_text_reads_as_written_with_nothing_in_it_taken_for_markdown() -> None:
    assert escaped_markdown("Uses a*b*c, not [x]") == r"Uses a\*b\*c\, not \[x\]"
    written = r"- 1. Uses `a*b*c` and $x$, not \n or <b> # _x_"
    assert plain_markdown(escaped_markdown(written)) == written


def test_items_read_better_listed_past_about_two_lines() -> None:
    assert not reads_better_listed(
        ["light", "water", "carbon dioxide", "chlorophyll", "glucose", "oxygen"]
    )
    assert not reads_better_listed(["x" * 200])
    assert not reads_better_listed(["a" * 39, "b" * 39])
    assert reads_better_listed(["a" * 39, "b" * 40])
