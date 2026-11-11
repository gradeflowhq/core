from gradeflow_core import BUILTINS, Step
from gradeflow_core.steps.step import STEP_OWN_PROPERTIES, STEP_PROPERTIES, load_step

ACCEPTED_ANSWERS = "gradeflow/accepted-answers@1"
EXTRACT = "gradeflow/extract@1"
YAML_ORDER = ["id", "name", "question", "input", "if", "weight", "uses", "with", "feedback"]


def problems(step: Step) -> list[tuple[tuple[str | int, ...], str]]:
    return [(problem.field, problem.message) for problem in step.resolution.data_problems]


def test_a_steps_schema_has_its_properties_in_the_order_yaml_shows_them() -> None:
    schema = Step.model_json_schema()
    assert list(schema["properties"]) == list(STEP_PROPERTIES) == YAML_ORDER
    assert schema["required"] == ["uses"]
    assert schema["properties"]["weight"]["default"] == 1
    assert list(STEP_OWN_PROPERTIES) == [key for key in YAML_ORDER if key not in ("uses", "with")]


def test_a_step_dumps_what_it_was_written_with_in_that_order() -> None:
    data = {
        "with": {"answers": ["9.81"]},
        "feedback": {"correct": "Yes."},
        "weight": 2,
        "uses": ACCEPTED_ANSWERS,
        "if": "{{ true }}",
        "id": "g",
    }
    dumped = load_step({**data, "name": None}, BUILTINS.by_ref).dump()
    assert dumped == data
    assert list(dumped) == [key for key in YAML_ORDER if key in data]


def test_a_step_whose_data_isnt_valid_keeps_what_is_valid() -> None:
    step = load_step(
        {
            "id": "1st",
            "name": "Units",
            "nmae": "x",
            "uses": ACCEPTED_ANSWERS,
            "with": {"answers": 3},
        },
        BUILTINS.by_ref,
    )
    assert (step.id, step.name, step.weight, step.settings) == (None, "Units", 1, None)
    assert [field for field, _ in problems(step)] == [("id",), ("nmae",), ("answers",)]


def test_a_transform_has_no_weight_or_feedback() -> None:
    settings = {"pattern": "x = (.+)"}
    step = load_step(
        {"id": "n", "weight": 2, "uses": EXTRACT, "with": settings, "feedback": "Read."},
        BUILTINS.by_ref,
    )
    assert problems(step) == [
        (("weight",), "Unknown setting."),
        (("feedback",), "Unknown setting."),
    ]
    assert (step.weight, step.feedback) == (0, None)
    transform = load_step({"id": "n", "uses": EXTRACT, "with": settings}, BUILTINS.by_ref)
    assert transform.dump() == {"id": "n", "uses": EXTRACT, "with": settings}


def test_a_copy_of_a_step_keeps_what_loading_worked_out() -> None:
    step = load_step({"input": "{{ values[0] }}", "uses": ACCEPTED_ANSWERS}, BUILTINS.by_ref)
    copy = step.model_copy(update={"input": None})
    assert copy.input is None
    assert copy.resolution is step.resolution


def test_what_loading_works_out_is_not_a_property_of_the_step() -> None:
    step = load_step({"uses": ACCEPTED_ANSWERS, "resolution": {}}, BUILTINS.by_ref)
    assert problems(step)[0] == (("resolution",), "Unknown setting.")
    assert "resolution" not in step.dump()
    assert "resolution" not in Step.model_json_schema()["properties"]
