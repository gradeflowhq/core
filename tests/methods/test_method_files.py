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

METHOD_FILE_HEADER = "label: Term\nversion: 1\nsummary: A term.\nquestion_types: [text]\n"
TERMS_METHOD_FILE = (
    METHOD_FILE_HEADER
    + """inputs:
  terms:
    title: Terms
  limit_to_three_words:
    default: false
checks:
  - for_each: "{{ inputs.terms }}"
    name: "Term {{ index }}"
    uses: gradeflow/key-point@1
    with:
      point: "{{ item }}"
  - if: "{{ inputs.limit_to_three_words }}"
    uses: gradeflow/length@1
    with: {max_length: 3, unit: words}
description: "Say {{ inputs.terms | quoted | and_list }}."
feedback: >-
  {% if correct %}Good.{% else %}{{ parts | selectattr('correct') | list | length }} found.
  {%- endif %}
"""
)


def with_sample_files(**files: str) -> Methods:
    return BUILTINS.with_files([(f"sample/{name}", text) for name, text in files.items()])


def test_a_method_file_repeats_checks_for_each_item_and_keeps_checks_whose_conditions_hold() -> (
    None
):
    methods = with_sample_files(terms=TERMS_METHOD_FILE)
    check = {"uses": "sample/terms@1", "with": {"terms": ["light", "water"]}}
    with_word_limit = {
        "uses": "sample/terms@1",
        "with": {"terms": ["light", "water"], "limit_to_three_words": True},
    }
    assert proportions(mark_all(check, ["light and water here", "light"], methods=methods)) == [
        1,
        0,
    ]
    assert proportions(
        mark_all(with_word_limit, ["light and water here", "light water"], methods=methods)
    ) == [
        0,
        1,
    ]
    assert [
        result.feedback for result in mark_all(check, ["light, water", "light"], methods=methods)
    ] == [
        "Good.",
        "1 found.",
    ]
    assert mark_all(check, ["x"], methods=methods)[0].method_label == "Term"


def test_a_method_files_feedback_shows_the_answer_as_written() -> None:
    echo_file = (
        METHOD_FILE_HEADER
        + """checks:
  - uses: gradeflow/key-point@1
    with: {point: paris}
feedback: "You wrote {{ answer }}."
"""
    )
    [result] = mark_all(
        {"uses": "sample/echo@1"}, ["  Paris "], methods=with_sample_files(echo=echo_file)
    )
    assert result.feedback == "You wrote Paris."


def test_a_method_file_without_a_summary_has_an_empty_one() -> None:
    method_file = (
        METHOD_FILE_HEADER.replace("summary: A term.\n", "")
        + "checks: [{uses: gradeflow/full-marks@1}]\n"
    )
    assert with_sample_files(bare=method_file).find("sample/bare@1").summary == ""


def test_a_repeated_check_is_named_by_its_index() -> None:
    method_file = METHOD_FILE_HEADER + (
        "inputs:\n  terms: {}\n"
        'checks:\n  - for_each: "{{ inputs.terms }}"\n'
        '    name: "Term {{ index }}"\n'
        '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n'
    )
    methods = BUILTINS.with_files([("sample/terms", method_file)])
    [result] = mark_all(
        {"uses": "sample/terms@1", "with": {"terms": ["a", "b"]}}, ["a"], methods=methods
    )
    assert result.feedback == "Term 2: Missing “b”."


def test_inputs_are_typed_by_the_settings_they_fill() -> None:
    method = BUILTINS.with_files([("sample/terms", TERMS_METHOD_FILE)])[-1]
    assert method.ref == "sample/terms@1"
    fields = method.settings.model_fields
    assert (fields["terms"].title, fields["terms"].is_required()) == ("Terms", True)
    assert (fields["limit_to_three_words"].annotation, fields["limit_to_three_words"].default) == (
        bool,
        False,
    )
    with pytest.raises(ValueError, match=re.escape("terms")):
        method.check_model.model_validate({"uses": "sample/terms@1", "with": {"terms": [1]}})


def test_an_input_applies_when_the_setting_it_fills_applies() -> None:
    fixed_strictness_file = METHOD_FILE_HEADER + (
        "inputs:\n  strictness: {default: strict}\n"
        "checks:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: spelling, strictness: "{{ inputs.strictness }}"}\n'
    )
    renamed_inputs_file = METHOD_FILE_HEADER + (
        "inputs:\n  how_to_match: {default: exact}\n  only_whole_words: {default: false}\n"
        "checks:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: "{{ inputs.how_to_match }}",'
        ' whole_words: "{{ inputs.only_whole_words }}"}\n'
    )
    [fixed_strictness, renamed_inputs] = with_sample_files(
        spelt=fixed_strictness_file, named=renamed_inputs_file
    )[-2:]
    assert hint_of(fixed_strictness.settings.model_fields["strictness"]).applies_when is None
    assert hint_of(renamed_inputs.settings.model_fields["only_whole_words"]).applies_when == {
        "how_to_match": "exact"
    }


def test_an_input_titles_the_fields_of_its_items() -> None:
    weighted_items_file = METHOD_FILE_HEADER + (
        "inputs:\n  terms:\n    items:\n"
        "      marks: {title: Marks, description: Its marks., examples: [2, 1]}\n"
        'checks:\n  - for_each: "{{ inputs.terms }}"\n    weight: "{{ item.marks }}"\n'
        '    uses: gradeflow/key-point@1\n    with: {point: "{{ item.text }}"}\n'
    )
    [method] = with_sample_files(weights=weighted_items_file)[-1:]
    [item_model] = get_args(method.settings.model_fields["terms"].annotation)
    marks, text_field = item_model.model_fields["marks"], item_model.model_fields["text"]
    assert (marks.title, marks.description, marks.examples) == ("Marks", "Its marks.", [2, 1])
    assert text_field.title == "Point", "as the setting it fills"


def test_an_input_without_a_default_must_be_filled_in() -> None:
    method_file = METHOD_FILE_HEADER.replace("[text]", "[number]") + (
        "inputs:\n  minimum: {title: Minimum}\n"
        "checks:\n  - uses: gradeflow/number-range@1\n"
        '    with: {min_value: "{{ inputs.minimum }}"}\n'
    )
    method = BUILTINS.with_files([("sample/minimum", method_file)])[-1]
    assert method.settings.model_fields["minimum"].is_required()


@pytest.mark.parametrize(
    ("method_file", "message"),
    [
        ("checks: [\n", "Line 6: expected the node content, but found '<stream end>'."),
        (
            "inputs:\n  terms: {}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n"
            "    with:\n      point: {{ inputs.terms }}\n",
            'Line 10: Quote {{ … }}, as in "{{ inputs.name }}".',
        ),
        ("colour: red\nchecks: []\n", "Line 5: colour: Unknown setting."),
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
            "inputs:\n  settings: {}\n"
            'checks:\n  - uses: gradeflow/key-point@1\n    with: "{{ inputs.settings }}"\n',
            "Line 9: Fill in a method's settings one by one, not with inputs.settings.",
        ),
        (
            "inputs:\n  terms: {}\n  unused: {}\n"
            'checks:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ inputs.terms }}"}\n',
            "Line 7: Input unused isn't used.",
        ),
        (
            "inputs:\n  terms: {}\n  unused: {title: Marking criteria}\n"
            'checks:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ inputs.terms }}"}\n',
            "Line 7: Marking criteria isn't used.",
        ),
        (
            'checks:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ inputs.terms }}"}\n',
            "Line 7: There's no inputs.terms.",
        ),
        (
            "inputs:\n  first: {}\n  second: {}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ inputs.first - inputs.second }}"}\n',
            "Line 10: A template can't do arithmetic; use a filter, such as "
            "{{ proportion | percent }}.",
        ),
        (
            "inputs:\n  first: {}\n"
            "checks:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ inputs.first | upper }}"}\n',
            "Line 9: A setting takes an input as it is, such as {{ inputs.name }}.",
        ),
        (
            "checks:\n  - for_each: [a, b]\n"
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n',
            "Line 6: for_each needs one {{ inputs.… }}.",
        ),
        (
            'inputs:\n  terms: {}\nchecks:\n  - for_each: "{{ inputs.terms }}"\n'
            "    uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 8: A repeated check uses {{ item }}.",
        ),
        (
            "inputs:\n  terms: {items: {marks: {title: Marks}}}\n"
            'checks:\n  - for_each: "{{ inputs.terms }}"\n'
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ item.text }}"}\n',
            "Line 8: Each item has no marks.",
        ),
        (
            'inputs:\n  terms: {}\nchecks:\n  - for_each: "{{ inputs.terms }}"\n'
            '    if: "{{ loop.first }}"\n'
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n',
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
            'description: "{{ inputs.terms }}"\n'
            "checks:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 5: inputs has no terms.",
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
            "Line 9: Whole words only applies when Match is “Exactly”.",
        ),
    ],
)
def test_a_method_file_that_cant_work_is_refused_at_its_line(
    method_file: str, message: str
) -> None:
    with pytest.raises(MethodFileError) as caught:
        BUILTINS.with_files([("sample/term", METHOD_FILE_HEADER + method_file)])
    assert str(caught.value).partition(": ")[2] == message


def test_composed_methods_may_use_each_other_but_not_in_a_cycle() -> None:
    wrapper_file = (
        "label: Wrapper\nversion: 1\nsummary: Wraps.\nquestion_types: [text]\n"
        "checks:\n  - uses: sample/terms@1\n    with: {terms: [a]}\n"
    )
    methods = with_sample_files(wrapper=wrapper_file, terms=TERMS_METHOD_FILE)
    assert [method.ref for method in methods][-2:] == ["sample/wrapper@1", "sample/terms@1"]
    method_a = (
        "label: A\nversion: 1\nsummary: A.\nquestion_types: [text]\nchecks:\n  - uses: sample/b@1\n"
    )
    method_b = (
        "label: B\nversion: 1\nsummary: B.\nquestion_types: [text]\nchecks:\n  - uses: sample/a@1\n"
    )
    with pytest.raises(
        MethodFileError,
        match=re.escape("sample/a@1 can't use itself: sample/a@1 → sample/b@1 → sample/a@1."),
    ):
        with_sample_files(a=method_a, b=method_b)
    with pytest.raises(MethodFileError, match="^" + re.escape("sample/a: Line 2: expected")):
        with_sample_files(a="checks: [\n")


def test_a_composed_method_describes_itself_and_marks_like_a_built_in_one() -> None:
    key_points = {"uses": "gradeflow/key-points@1", "with": {"points": ["a"]}}
    assert describe(key_points) == "Includes `a`."
    assert mark(key_points, "a").method_label == "Key Points"


def test_a_method_lists_what_its_checks_are_set_up_with() -> None:
    assert with_sample_files(terms=TERMS_METHOD_FILE)[-1].setting_titles == [
        "Terms",
        "limit_to_three_words",
    ]
