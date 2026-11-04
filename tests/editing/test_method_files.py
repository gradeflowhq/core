"""Method files saved over time load as far as they can."""

from gradeflow_core import BUILTINS
from gradeflow_core.editing import FailedFile, dump_yaml, load_method_files


def method_file(check: dict[str, object]) -> str:
    return dump_yaml(
        {
            "label": "Saved",
            "version": 1,
            "summary": "A saved method.",
            "question_types": ["text"],
            "checks": [check],
        }
    )


def test_a_file_that_no_longer_loads_stops_only_the_files_that_use_it() -> None:
    old = method_file({"uses": "gradeflow/key-points@1", "with": {"evaluate_expected": False}})
    files = [
        ("sample/user", method_file({"uses": "sample/old@1"})),
        ("sample/old", old),
        ("sample/other", method_file({"uses": "gradeflow/full-marks@1"})),
        ("sample", old),
        ("sample/broken", "label: [unclosed"),
    ]
    loaded = load_method_files(files, BUILTINS)
    assert [method.ref for method in loaded.methods[len(BUILTINS) :]] == ["sample/other@1"]
    assert set(loaded.failed) == {"sample/user@1", "sample/old@1", "sample", "sample/broken"}
    at_fault = loaded.failed["sample/old@1"]
    assert at_fault.ref == "sample/old@1"
    assert at_fault.messages == ["Line 9: Key Points has no setting evaluate_expected."]
    assert loaded.failed["sample/user@1"] == at_fault
    assert loaded.failed["sample"] == FailedFile(
        "sample", ["Name it as a namespace and name, such as intro-biology/term."]
    )
    assert loaded.failed["sample/broken"].ref == "sample/broken"


def test_a_file_that_no_longer_loads_says_each_problem() -> None:
    old = dump_yaml(
        {
            "label": "Saved",
            "version": 1,
            "summary": "A saved method.",
            "question_types": ["text"],
            "checks": [
                {"uses": "gradeflow/key-points@1", "with": {"old": 1, "points": ["a"], "older": 2}},
                {"uses": "gradeflow/best-answer@1"},
                {
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": "{{ inputs.missing }}"},
                },
            ],
        }
    )
    [failed] = load_method_files([("sample/old", old)], BUILTINS).failed.values()
    assert failed.messages == [
        "Line 9: Key Points has no setting old.",
        "Line 12: Key Points has no setting older.",
        "Line 13: There's no method gradeflow/best-answer@1.",
        "Line 16: There's no inputs.missing.",
    ]
