import re
from typing import get_args

import pytest

from gradeflow_core import (
    BUILTINS,
    InvalidError,
    MethodFileError,
    Methods,
)
from gradeflow_core.common.hints import hint_of
from gradeflow_core.steps.step import valid_step

from .helpers import describe, mark, mark_all, proportions

METHOD_FILE_HEADER = "label: Term\nversion: 1\nsummary: A term.\nquestion_types: [text]\n"
TERMS_METHOD_FILE = (
    METHOD_FILE_HEADER
    + """settings:
  terms:
    title: Terms
  limit_to_three_words:
    default: false
steps:
  - for_each: "{{ settings.terms }}"
    name: "Term {{ index }}"
    uses: gradeflow/key-point@1
    with:
      point: "{{ item }}"
  - if: "{{ settings.limit_to_three_words }}"
    uses: gradeflow/length@1
    with: {max_length: 3, unit: words}
description: "Say {{ settings.terms | quoted | and_list }}."
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
        + """steps:
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
        + "steps: [{uses: gradeflow/full-marks@1}]\n"
    )
    assert with_sample_files(bare=method_file).find("sample/bare@1").summary == ""


def test_a_repeated_check_is_named_by_its_index() -> None:
    method_file = METHOD_FILE_HEADER + (
        "settings:\n  terms: {}\n"
        'steps:\n  - for_each: "{{ settings.terms }}"\n'
        '    name: "Term {{ index }}"\n'
        '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n'
    )
    methods = BUILTINS.with_files([("sample/terms", method_file)])
    [result] = mark_all(
        {"uses": "sample/terms@1", "with": {"terms": ["a", "b"]}}, ["a"], methods=methods
    )
    assert result.feedback == "Term 2: Missing “b”."


def test_settings_are_typed_by_the_fields_they_fill() -> None:
    method = BUILTINS.with_files([("sample/terms", TERMS_METHOD_FILE)]).by_ref["sample/terms@1"]
    fields = method.settings.model_fields
    assert (fields["terms"].title, fields["terms"].is_required()) == ("Terms", True)
    assert (fields["limit_to_three_words"].annotation, fields["limit_to_three_words"].default) == (
        bool,
        False,
    )
    with pytest.raises(InvalidError, match=re.escape("terms")):
        valid_step({"uses": "sample/terms@1", "with": {"terms": [1]}}, Methods([method]).by_ref)


def test_a_setting_applies_when_the_field_it_fills_applies() -> None:
    fixed_strictness_file = METHOD_FILE_HEADER + (
        "settings:\n  strictness: {default: strict}\n"
        "steps:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: spelling, strictness: "{{ settings.strictness }}"}\n'
    )
    renamed_settings_file = METHOD_FILE_HEADER + (
        "settings:\n  how_to_match: {default: exact}\n  only_whole_words: {default: false}\n"
        "steps:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: "{{ settings.how_to_match }}",'
        ' whole_words: "{{ settings.only_whole_words }}"}\n'
    )
    methods_by_ref = with_sample_files(
        **{"fixed-strictness": fixed_strictness_file, "renamed-settings": renamed_settings_file}
    ).by_ref
    fixed_strictness = methods_by_ref["sample/fixed-strictness@1"]
    renamed_settings = methods_by_ref["sample/renamed-settings@1"]
    assert hint_of(fixed_strictness.settings.model_fields["strictness"]).applies_when is None
    assert hint_of(renamed_settings.settings.model_fields["only_whole_words"]).applies_when == {
        "how_to_match": ["exact"]
    }


def test_a_setting_titles_the_fields_of_its_items() -> None:
    weighted_items_file = METHOD_FILE_HEADER + (
        "settings:\n  terms:\n    items:\n"
        "      marks: {title: Marks, description: Its marks., examples: [2, 1]}\n"
        'steps:\n  - for_each: "{{ settings.terms }}"\n    weight: "{{ item.marks }}"\n'
        '    uses: gradeflow/key-point@1\n    with: {point: "{{ item.text }}"}\n'
    )
    method = with_sample_files(weights=weighted_items_file).by_ref["sample/weights@1"]
    [item_model] = get_args(method.settings.model_fields["terms"].annotation)
    marks, text_field = item_model.model_fields["marks"], item_model.model_fields["text"]
    assert (marks.title, marks.description, marks.examples) == ("Marks", "Its marks.", [2, 1])
    assert text_field.title == "Point", "as the setting it fills"


def test_a_setting_without_a_default_must_be_filled_in() -> None:
    method_file = METHOD_FILE_HEADER.replace("[text]", "[number]") + (
        "settings:\n  minimum: {title: Minimum}\n"
        "steps:\n  - uses: gradeflow/number-range@1\n"
        '    with: {min_value: "{{ settings.minimum }}"}\n'
    )
    method = BUILTINS.with_files([("sample/minimum", method_file)]).by_ref["sample/minimum@1"]
    assert method.settings.model_fields["minimum"].is_required()


def test_a_setting_may_be_used_inside_a_field_template() -> None:
    method_file = METHOD_FILE_HEADER + (
        "settings:\n  term: {title: Term}\n  topic: {title: Topic}\n"
        "steps:\n  - uses: gradeflow/accepted-answers@1\n"
        '    with: {answers: ["the {{ settings.term }}"]}\n'
        "  - uses: gradeflow/key-point@1\n"
        '    with: {point: "Mentions {{ settings.topic }}"}\n'
    )
    methods = BUILTINS.with_files([("sample/term", method_file)])
    marking = {"uses": "sample/term@1", "with": {"term": "cell", "topic": "cells"}}
    [result] = mark_all(marking, ["a cell"], methods=methods)
    assert result.feedback == "Expected “the cell”.\nMissing “Mentions cells”."


@pytest.mark.parametrize(
    ("method_file", "message"),
    [
        ("steps: [\n", "Line 6: expected the node content, but found '<stream end>'."),
        (
            "settings:\n  terms: {}\n"
            "steps:\n  - uses: gradeflow/key-point@1\n"
            "    with:\n      point: {{ settings.terms }}\n",
            'Line 10: Quote {{ … }}, as in "{{ settings.name }}".',
        ),
        ("colour: red\nsteps: []\n", "Line 5: colour: Unknown setting."),
        (
            "marks_by: most\nsteps:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 5: Choose 'all_correct', 'any_correct', 'none_correct', 'total' or "
            "'proportional'.",
        ),
        ("steps:\n  - uses: gradeflow/nope@1\n", "Line 6: There's no method gradeflow/nope@1."),
        (
            "steps:\n  - uses: gradeflow/key-point@1\n    colour: red\n    with: {point: a}\n",
            "Line 7: A step has no colour:.",
        ),
        (
            "steps:\n  - uses: gradeflow/key-point@1\n    with: {point: a, colour: red}\n",
            "Line 7: Key Point has no setting colour.",
        ),
        (
            "steps:\n  - uses: gradeflow/key-point@1\n    with: bad\n",
            "Line 7: A step's settings aren't a mapping.",
        ),
        (
            "steps:\n  - uses: gradeflow/length@1\n    with: {max_length: -1}\n",
            "Line 7: Must be 0 or more.",
        ),
        (
            "settings:\n  settings: {}\n"
            'steps:\n  - uses: gradeflow/key-point@1\n    with: "{{ settings.settings }}"\n',
            "Line 9: Fill in a method's settings one by one, not with settings.settings.",
        ),
        (
            "settings:\n  terms: {}\n  unused: {}\n"
            'steps:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ settings.terms }}"}\n',
            "Line 7: Setting unused isn't used.",
        ),
        (
            "settings:\n  terms: {}\n  unused: {title: Marking criteria}\n"
            'steps:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ settings.terms }}"}\n',
            "Line 7: Marking criteria isn't used.",
        ),
        (
            'steps:\n  - uses: gradeflow/key-point@1\n    with: {point: "{{ settings.terms }}"}\n',
            "Line 7: There's no settings.terms.",
        ),
        (
            "settings:\n  first: {}\n  second: {}\n"
            "steps:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ settings.first - settings.second }}"}\n',
            "Line 10: A template can't do arithmetic; use a filter, such as "
            "{{ proportion | percent }}.",
        ),
        (
            "settings:\n  first: {}\n"
            "steps:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ settings.first + }}"}\n',
            "Line 9: Can't read this reference.",
        ),
        (
            "settings:\n  first: {}\n"
            "steps:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "{{ settings.first | upper }}"}\n',
            "Line 9: Use one {{ settings.name }} and nothing else.",
        ),
        (
            "steps:\n  - for_each: [a, b]\n"
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n',
            "Line 6: for_each needs one {{ settings.… }}.",
        ),
        (
            'settings:\n  terms: {}\nsteps:\n  - for_each: "{{ settings.terms }}"\n'
            "    uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 8: A repeated step uses {{ item }}.",
        ),
        (
            "settings:\n  terms: {items: {marks: {title: Marks}}}\n"
            'steps:\n  - for_each: "{{ settings.terms }}"\n'
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ item.text }}"}\n',
            "Line 8: Each item has no marks.",
        ),
        (
            'settings:\n  terms: {}\nsteps:\n  - for_each: "{{ settings.terms }}"\n'
            '    if: "{{ loop.first }}"\n'
            '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n',
            "Line 9: There's no loop.",
        ),
        (
            "steps:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n"
            '    feedback: "{% set x = 1 %}"\n',
            "Line 8: A template can't use {% set %}.",
        ),
        (
            "steps:\n  - uses: gradeflow/key-point@1\n"
            '    if: "{{ steps.x.output.correct }}"\n    with: {point: a}\n',
            "Line 7: No step before this one is called x.",
        ),
        (
            'steps:\n  - uses: gradeflow/key-point@1\n    if: "{{ true }}"\n    with: {point: a}\n',
            "Line 7: Make a condition read an earlier step or a setting, such as "
            "{{ steps.test.output.correct }}.",
        ),
        (
            'description: "{{ settings.terms }}"\n'
            "steps:\n  - uses: gradeflow/key-point@1\n    with: {point: a}\n",
            "Line 5: settings has no terms.",
        ),
        *(
            (
                'settings:\n  terms: {}\nsteps:\n  - for_each: "{{ settings.terms }}"\n'
                f'    name: "Term {{{{ {expression} }}}}"\n'
                '    uses: gradeflow/key-point@1\n    with: {point: "{{ item }}"}\n',
                f"Line 9: {message}",
            )
            for expression, message in [
                ("bogus", "There's no bogus."),
                ("item.__class__", "There's no __class__."),
                (
                    "index + 1",
                    "A template can't do arithmetic; use a filter, such as "
                    "{{ proportion | percent }}.",
                ),
            ]
        ),
        (
            "settings:\n  term: {}\n"
            "steps:\n  - uses: gradeflow/accepted-answers@1\n"
            '    with: {answers: ["the {{ settings.term }}", " "]}\n',
            "Line 9: Can't be blank.",
        ),
        (
            "steps:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: "Has {{ steps.x.output.feedback }}"}\n',
            "Line 7: There's no steps.",
        ),
        (
            "settings:\n  whole: {default: true}\n"
            "steps:\n  - uses: gradeflow/key-point@1\n"
            '    with: {point: a, match: spelling, whole_words: "{{ settings.whole }}"}\n',
            "Line 9: Whole words only applies when Match is “Exactly”.",
        ),
        (
            "steps:\n  - uses: gradeflow/workflow@1\n    with: {steps: 5}\n",
            "Line 7: Must be a list.",
        ),
        *(
            (
                f"settings:\n  {name}: {{default: [a]}}\n"
                "steps:\n  - uses: gradeflow/accepted-answers@1\n"
                f'    with: {{answers: "{{{{ settings.{name} }}}}"}}\n',
                f"Line 6: A setting can't be called {name}. Choose another name.",
            )
            for name in ("expansion", "model_dump", "_hidden")
        ),
    ],
)
def test_a_method_file_that_cant_work_is_refused_at_its_line(
    method_file: str, message: str
) -> None:
    with pytest.raises(MethodFileError) as exc_info:
        BUILTINS.with_files([("sample/term", METHOD_FILE_HEADER + method_file)])
    assert str(exc_info.value).partition(": ")[2] == message


def test_composed_methods_may_use_each_other_but_not_in_a_cycle() -> None:
    wrapper_file = (
        "label: Wrapper\nversion: 1\nsummary: Wraps.\nquestion_types: [text]\n"
        "steps:\n  - uses: sample/terms@1\n    with: {terms: [a]}\n"
    )
    methods = with_sample_files(wrapper=wrapper_file, terms=TERMS_METHOD_FILE)
    assert [method.ref for method in methods][-2:] == ["sample/wrapper@1", "sample/terms@1"]
    method_a = (
        "label: A\nversion: 1\nsummary: A.\nquestion_types: [text]\nsteps:\n  - uses: sample/b@1\n"
    )
    method_b = (
        "label: B\nversion: 1\nsummary: B.\nquestion_types: [text]\nsteps:\n  - uses: sample/a@1\n"
    )
    with pytest.raises(
        MethodFileError,
        match=re.escape("sample/a@1: Can't use itself: sample/a@1 → sample/b@1 → sample/a@1."),
    ):
        with_sample_files(a=method_a, b=method_b)
    with pytest.raises(MethodFileError, match="^" + re.escape("sample/a: Line 2: expected")):
        with_sample_files(a="steps: [\n")


def test_a_composed_method_describes_itself_and_marks_like_a_built_in_one() -> None:
    key_points = {"uses": "gradeflow/key-points@1", "with": {"points": ["a"]}}
    assert describe(key_points) == "Includes `a`."
    assert mark(key_points, "a").method_label == "Key Points"


def test_a_setting_applies_for_a_list_of_values_not_one() -> None:
    method_file = METHOD_FILE_HEADER + (
        "settings:\n  match: {default: exact}\n"
        "  whole_words: {default: false, applies_when: {match: exact}}\n"
        "steps:\n  - uses: gradeflow/key-point@1\n"
        '    with: {point: a, match: "{{ settings.match }}",'
        ' whole_words: "{{ settings.whole_words }}"}\n'
    )
    with pytest.raises(MethodFileError, match="list"):
        with_sample_files(single=method_file)


def test_a_step_whose_settings_its_method_file_refuses_needs_no_model() -> None:
    methods = with_sample_files(
        between="""label: Between
version: 1
summary: Between two numbers.
question_types: [number]
settings:
  low: {}
  high: {}
steps:
  - uses: gradeflow/number-range@1
    with:
      min_value: "{{ settings.low }}"
      max_value: "{{ settings.high }}"
"""
    )
    step = valid_step({"uses": "sample/between@1", "with": {"low": 2, "high": 1}}, methods.by_ref)
    assert not step.needs_model_endpoint()


def test_a_setting_used_only_in_a_step_name_is_used() -> None:
    method_file = METHOD_FILE_HEADER + (
        "settings:\n  topic: {default: cells}\n"
        "steps:\n  - uses: gradeflow/key-point@1\n"
        '    name: "{{ settings.topic }} check"\n    with: {point: a}\n'
    )
    method = with_sample_files(topic=method_file).by_ref["sample/topic@1"]
    assert "topic" in method.settings.model_fields


def test_a_nested_workflow_may_give_one_of_its_steps_output() -> None:
    method_file = METHOD_FILE_HEADER + (
        "steps:\n"
        "  - id: num\n    uses: gradeflow/workflow@1\n"
        "    with:\n      output: step_output\n"
        '      output_step: "{{ steps.part.output }}"\n'
        "      steps:\n        - id: part\n          uses: gradeflow/extract@1\n"
        "          with: {type: number}\n"
        '  - uses: gradeflow/number-range@1\n    input: "{{ steps.num.output }}"\n'
        "    with: {min_value: 1}\n"
    )
    assert with_sample_files(nested=method_file)[-1].ref == "sample/nested@1"


def test_a_nested_workflow_output_step_must_name_one_of_its_own_steps() -> None:
    method_file = METHOD_FILE_HEADER + (
        "steps:\n"
        "  - id: other\n    uses: gradeflow/extract@1\n    with: {type: number}\n"
        "  - uses: gradeflow/workflow@1\n"
        "    with:\n      output: step_output\n"
        '      output_step: "{{ steps.other.output }}"\n'
        "      steps:\n        - id: part\n          uses: gradeflow/extract@1\n"
        "          with: {type: number}\n"
    )
    with pytest.raises(
        MethodFileError, match=re.escape("Line 12: No step before this one is called other.")
    ):
        with_sample_files(nested=method_file)


def test_a_template_under_a_settings_key_named_like_a_step_key_is_checked() -> None:
    method_file = METHOD_FILE_HEADER + (
        "steps:\n  - uses: gradeflow/custom-code@1\n"
        '    with:\n      code: "correct = True"\n'
        '      parameters:\n        input: {type: text, value: "x {{ settings.typo }} y"}\n'
    )
    with pytest.raises(
        MethodFileError, match="^" + re.escape("sample/term@1: Line 10: settings has no typo.")
    ):
        with_sample_files(term=method_file)
