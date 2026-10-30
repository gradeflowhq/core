import re
from typing import get_args

import pytest

from gradeflow_core import (
    BUILTINS,
    MethodFileError,
    Methods,
)
from gradeflow_core.common.hints import hint_of

from .helpers import describe, mark, mark_all, proportions

HEAD = "label: Term\nversion: 1\nsummary: A term.\nquestion_types: [text]\n"
TERMS = (
    HEAD
    + """inputs:
  terms:
    title: Terms
  strict:
    default: false
checks:
  - for_each: "{{ inputs.terms }}"
    name: "Term {{ index }}"
    uses: gradeflow/key-point@1
    with:
      point: "{{ each }}"
  - if: "{{ inputs.strict }}"
    uses: gradeflow/length@1
    with: {max_length: 3, count: words}
description: "Say {{ inputs.terms | quoted | and_list }}."
feedback: >-
  {% if correct %}Good.{% else %}{{ parts | selectattr('correct') | list | length }} found.
  {%- endif %}
"""
)


def library(**files: str) -> Methods:
    return BUILTINS.with_files([(f"sample/{name}", text) for name, text in files.items()])


def test_a_method_file_repeats_checks_for_each_item_and_keeps_checks_whose_conditions_hold() -> (
    None
):
    lib = library(terms=TERMS)
    check = {"uses": "sample/terms@1", "with": {"terms": ["light", "water"]}}
    strict = {"uses": "sample/terms@1", "with": {"terms": ["light", "water"], "strict": True}}
    assert proportions(mark_all(check, ["light and water here", "light"], methods=lib)) == [1, 0]
    assert proportions(mark_all(strict, ["light and water here", "light water"], methods=lib)) == [
        0,
        1,
    ]
    assert [r.feedback for r in mark_all(check, ["light, water", "light"], methods=lib)] == [
        "Good.",
        "1 found.",
    ]
    assert mark_all(check, ["x"], methods=lib)[0].method == "Term"


def test_a_method_files_feedback_shows_the_answer_as_written() -> None:
    echo = (
        HEAD
        + """checks:
  - uses: gradeflow/key-point@1
    with: {point: paris}
feedback: "You wrote {{ answer }}."
"""
    )
    [result] = mark_all({"uses": "sample/echo@1"}, ["  Paris "], methods=library(echo=echo))
    assert result.feedback == "You wrote Paris."


def test_a_method_file_without_a_summary_has_an_empty_one() -> None:
    text = HEAD.replace("summary: A term.\n", "") + "checks: [{uses: gradeflow/full-marks@1}]\n"
    assert library(bare=text).find("sample/bare@1").summary == ""


def test_a_repeated_check_is_named_by_its_index() -> None:
    text = HEAD + (
        "inputs:\n  terms: {}\n"
        'checks:\n  - for_each: "{{ inputs.terms }}"\n'
        '    name: "Term {{ index }}"\n'
        '    uses: gradeflow/key-point@1\n    with: {point: "{{ each }}"}\n'
    )
    lib = BUILTINS.with_files([("sample/terms", text)])
    [result] = mark_all(
        {"uses": "sample/terms@1", "with": {"terms": ["a", "b"]}}, ["a"], methods=lib
    )
    assert result.feedback == "Term 2: Missing “b”."


def test_inputs_are_typed_by_the_settings_they_fill() -> None:
    method = BUILTINS.with_files([("sample/terms", TERMS)])[-1]
    assert method.ref == "sample/terms@1"
    fields = method.settings.model_fields
    assert (fields["terms"].title, fields["terms"].is_required()) == ("Terms", True)
    assert (fields["strict"].annotation, fields["strict"].default) == (bool, False)
    with pytest.raises(ValueError, match=re.escape("terms")):
        method.check_model.model_validate({"uses": "sample/terms@1", "with": {"terms": [1]}})


def test_an_input_applies_when_what_fills_the_settings_its_setting_depends_on_does() -> None:
    spelt = HEAD + (
        "inputs:\n  strictness: {default: strict}\n"
        "checks:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: spelling, strictness: "{{ inputs.strictness }}"}\n'
    )
    renamed = HEAD + (
        "inputs:\n  how: {default: exact}\n  whole: {default: false}\n"
        "checks:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: "{{ inputs.how }}", whole_words: "{{ inputs.whole }}"}\n'
    )
    [spelling, named] = library(spelt=spelt, named=renamed)[-2:]
    assert hint_of(spelling.settings.model_fields["strictness"]).applies_when is None
    assert hint_of(named.settings.model_fields["whole"]).applies_when == {"how": "exact"}


def test_an_input_titles_the_fields_of_its_items() -> None:
    marked = HEAD + (
        "inputs:\n  terms:\n    items:\n"
        "      marks: {title: Marks, description: Its marks., examples: [2, 1]}\n"
        'checks:\n  - for_each: "{{ inputs.terms }}"\n    weight: "{{ each.marks }}"\n'
        '    uses: gradeflow/key-point@1\n    with: {point: "{{ each.text }}"}\n'
    )
    [method] = library(marked=marked)[-1:]
    [item] = get_args(method.settings.model_fields["terms"].annotation)
    marks, point = item.model_fields["marks"], item.model_fields["text"]
    assert (marks.title, marks.description, marks.examples) == ("Marks", "Its marks.", [2, 1])
    assert point.title == "Point", "as the setting it fills"


def test_an_input_without_a_default_must_be_filled_in() -> None:
    text = HEAD.replace("[text]", "[number]") + (
        "inputs:\n  low: {title: Low}\n"
        "checks:\n  - uses: gradeflow/number-range@1\n"
        '    with: {min_value: "{{ inputs.low }}"}\n'
    )
    method = BUILTINS.with_files([("sample/low", text)])[-1]
    assert method.settings.model_fields["low"].is_required()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("checks: [\n", "Line 6: expected the node content, but found '<stream end>'."),
        (
            "inputs:\n  t: {}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n    with:\n      point: {{ inputs.t }}\n",
            'Line 10: Quote {{ … }}, as in "{{ inputs.name }}".',
        ),
        ("colour: red\nchecks: []\n", "Line 5: colour: Not allowed here."),
        (
            "combine: most\nchecks:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 5: Choose 'all', 'any', 'none', 'total' or 'proportional'.",
        ),
        ("checks:\n  - uses: gradeflow/nope@1\n", "There's no method gradeflow/nope@1."),
        (
            "checks:\n  - uses: gradeflow/key-point@1\n    colour: red\n    with: {point: a}\n",
            "Line 7: A check has no colour:.",
        ),
        (
            "checks:\n  - uses: gradeflow/key-point@1\n    with: {point: a, colour: red}\n",
            "Line 7: Key Point has no setting colour.",
        ),
        (
            "checks:\n  - uses: gradeflow/key-point@1\n    with: bad\n",
            "Line 7: A check's settings aren't a mapping.",
        ),
        (
            "checks:\n  - uses: gradeflow/length@1\n    with: {max_length: -1}\n",
            "Line 7: Must be 0 or more.",
        ),
        (
            "inputs:\n  s: {}\n"
            'checks:\n  - uses: gradeflow/key-point@1\n    with: "{{ inputs.s }}"\n',
            "Line 9: Fill in a method's settings one by one, not with inputs.s.",
        ),
        (
            "inputs:\n  t: {}\n  u: {}\n"
            'checks:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ inputs.t }}"}\n',
            "Line 7: Input u isn't used.",
        ),
        (
            "inputs:\n  t: {}\n  u: {title: Marking criteria}\n"
            'checks:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ inputs.t }}"}\n',
            "Line 7: Marking criteria isn't used.",
        ),
        (
            'checks:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ inputs.t }}"}\n',
            "Line 7: There's no inputs.t.",
        ),
        (
            "inputs:\n  a: {}\n  b: {}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ inputs.a - inputs.b }}"}\n',
            "Line 10: A template can't do arithmetic; use a filter, such as "
            "{{ proportion | percent }}.",
        ),
        (
            "inputs:\n  a: {}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ inputs.a | upper }}"}\n',
            "Line 9: A setting takes an input as it is, such as {{ inputs.name }}.",
        ),
        (
            "checks:\n  - for_each: [a, b]\n"
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ each }}"}\n',
            "Line 6: for_each needs one {{ inputs.… }}.",
        ),
        (
            'inputs:\n  t: {}\nchecks:\n  - for_each: "{{ inputs.t }}"\n'
            "    uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 8: A repeated check uses {{ each }}.",
        ),
        (
            "inputs:\n  t: {items: {marks: {title: Marks}}}\n"
            'checks:\n  - for_each: "{{ inputs.t }}"\n'
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ each.text }}"}\n',
            "Line 8: Each item has no marks.",
        ),
        (
            'inputs:\n  t: {}\nchecks:\n  - for_each: "{{ inputs.t }}"\n'
            '    if: "{{ loop.first }}"\n'
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ each }}"}\n',
            "Line 9: There's no loop.",
        ),
        (
            "checks:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n"
            '    feedback: "{% set x = 1 %}"\n',
            "Line 8: A template can't use {% set %}.",
        ),
        (
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    if: "{{ checks.x.correct }}"\n    with: {point: a}\n',
            "Line 7: No check before this one is called x.",
        ),
        (
            'checks:\n  - uses: gradeflow/key-point@1\n    if: "{{ true }}"\n'
            "    with: {point: a}\n",
            "Line 7: Make a condition read an earlier check or an input, such as "
            "{{ checks.test.correct }}.",
        ),
        (
            'description: "{{ inputs.t }}"\n'
            "checks:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 5: inputs has no t.",
        ),
        (
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "Has {{ checks.x.feedback }}"}\n',
            "Line 7: There's no checks.",
        ),
        (
            "inputs:\n  whole: {default: true}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: a, match: spelling, whole_words: "{{ inputs.whole }}"}\n',
            "Line 9: Whole words only applies only when Match is “Exactly”.",
        ),
    ],
)
def test_a_method_file_that_cant_work_is_refused_at_its_line(text: str, message: str) -> None:
    with pytest.raises(MethodFileError) as caught:
        BUILTINS.with_files([("sample/term", HEAD + text)])
    assert str(caught.value).partition(": ")[2] == message


def test_composed_methods_may_use_each_other_but_not_in_a_cycle() -> None:
    wrapper = (
        "label: Wrapper\nversion: 1\nsummary: Wraps.\nquestion_types: [text]\n"
        "checks:\n  - uses: sample/terms@1\n    with: {terms: [a]}\n"
    )
    lib = library(wrapper=wrapper, terms=TERMS)
    assert [method.ref for method in lib][-2:] == ["sample/wrapper@1", "sample/terms@1"]
    a = "label: A\nversion: 1\nsummary: A.\nquestion_types: [text]\nchecks:\n  - uses: sample/b@1\n"
    b = "label: B\nversion: 1\nsummary: B.\nquestion_types: [text]\nchecks:\n  - uses: sample/a@1\n"
    with pytest.raises(
        MethodFileError,
        match=re.escape("sample/a@1 can't use itself: sample/a@1 → sample/b@1 → sample/a@1."),
    ):
        library(a=a, b=b)
    with pytest.raises(MethodFileError, match="^" + re.escape("sample/a: Line 2: expected")):
        library(a="checks: [\n")


def test_a_composed_method_describes_itself_and_marks_like_a_built_in_one() -> None:
    key_points = {"uses": "gradeflow/key-points@1", "with": {"points": ["a"]}}
    assert describe(key_points) == "Includes `a`."
    assert mark(key_points, "a").method == "Key Points"


def test_a_method_lists_what_its_checks_are_set_up_with() -> None:
    assert library(terms=TERMS)[-1].setting_titles == ["Terms", "strict"]
