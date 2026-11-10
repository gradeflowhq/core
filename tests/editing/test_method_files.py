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
            "steps": [check],
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
            "steps": [
                {"uses": "gradeflow/key-points@1", "with": {"old": 1, "points": ["a"], "older": 2}},
                {"uses": "gradeflow/best-answer@1"},
                {
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": "{{ settings.missing }}"},
                },
            ],
        }
    )
    [failed] = load_method_files([("sample/old", old)], BUILTINS).failed.values()
    assert failed.messages == [
        "Line 9: Key Points has no setting old.",
        "Line 12: Key Points has no setting older.",
        "Line 13: There's no method gradeflow/best-answer@1.",
        "Line 16: There's no settings.missing.",
    ]


def test_a_file_that_no_longer_reads_fails_under_its_ref() -> None:
    old = method_file({"uses": "gradeflow/full-marks@1"}) + "retired: true\n"
    version_2 = old.replace("version: 1", "version: 2")
    files = [
        ("sample/old", old),
        ("sample/old", version_2),
        ("sample/user", method_file({"uses": "sample/old@1"})),
        ("sample/broken", "version: 1\nlabel: [unclosed"),
    ]
    failed = load_method_files(files, BUILTINS).failed
    assert set(failed) == {"sample/old@1", "sample/old@2", "sample/user@1", "sample/broken"}
    assert failed["sample/old@1"].ref == "sample/old@1"
    assert failed["sample/old@2"].ref == "sample/old@2"
    assert failed["sample/user@1"] == failed["sample/old@1"]


def test_cycles_and_duplicates_stop_only_their_files() -> None:
    files = [
        ("sample/a", method_file({"uses": "sample/b@1"})),
        ("sample/b", method_file({"uses": "sample/a@1"})),
        ("sample/user", method_file({"uses": "sample/a@1"})),
        ("sample/twice", method_file({"uses": "gradeflow/full-marks@1"})),
        ("sample/twice", method_file({"uses": "gradeflow/full-marks@1"})),
        ("sample/other", method_file({"uses": "gradeflow/full-marks@1"})),
    ]
    loaded = load_method_files(files, BUILTINS)
    assert [method.ref for method in loaded.methods[len(BUILTINS) :]] == ["sample/other@1"]
    assert loaded.failed == {
        "sample/a@1": FailedFile(
            "sample/a@1", ["Can't use itself: sample/a@1 → sample/b@1 → sample/a@1."]
        ),
        "sample/b@1": FailedFile(
            "sample/b@1", ["Can't use itself: sample/b@1 → sample/a@1 → sample/b@1."]
        ),
        "sample/user@1": loaded.failed["sample/a@1"],
        "sample/twice@1": FailedFile("sample/twice@1", ["Duplicate method."]),
    }
    again = load_method_files(files[-1:], loaded.methods).failed
    assert again == {"sample/other@1": FailedFile("sample/other@1", ["Duplicate method."])}
