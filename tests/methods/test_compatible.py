from gradeflow_core import BUILTINS, Methods, QuestionSet
from gradeflow_core.editing import Position, compatible_methods
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import CodeReading
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(),
        "Q2": ChoiceQuestion(options=frozenset({"a", "b"})),
        "Q3": ValuesQuestion(value_types=["text", "number"]),
        "Q4": NumberQuestion(),
        "Q5": CodeQuestion(reading=CodeReading(guess_indentation=True)),
        "Q6": CodeQuestion(),
    }
)


def names(position: Position) -> list[str]:
    return [info.name.removeprefix("gradeflow/") for info in compatible_methods(position)]


def test_compatible_methods_are_the_supplied_objects_in_the_supplied_order() -> None:
    methods = Methods(reversed(BUILTINS))
    position = Position(QUESTIONS, "Q1")
    compatible = compatible_methods(position, methods=methods)
    expected = [
        method for method in methods if method.level != "internal" and method.fits(position)
    ]
    assert len(compatible) == len(expected)
    assert all(actual is supplied for actual, supplied in zip(compatible, expected, strict=True))


def test_compatible_methods_fit_the_question_or_group() -> None:
    assert names(Position(QUESTIONS, "Q1")) == [
        "accepted-answers",
        "key-points",
        "answer-criteria",
        "equivalent-expression",
        "length",
        "full-marks",
        "combination",
        "not-allowed",
        "pattern",
        "custom-code",
    ]
    assert names(Position(QUESTIONS, in_group=True)) == [
        "combination",
        "conditional",
        "answer-sets",
        "custom-code-for-a-group",
    ]


def test_a_value_of_an_answer_offers_what_marks_its_type() -> None:
    assert names(Position(QUESTIONS, "Q3"))[0] == "multiple-values"
    assert names(Position(QUESTIONS, "Q3", 2, in_list=True))[:3] == [
        "accepted-answers",
        "number-range",
        "full-marks",
    ]


ONE_CODE_ANSWER = [
    "accepted-answers",
    "answer-criteria",
    "full-marks",
    "combination",
    "not-allowed",
    "pattern",
    "code-tests",
    "code-uses",
    "custom-code",
]


def test_code_read_as_typed_offers_the_methods_that_mark_code() -> None:
    assert names(Position(QUESTIONS, "Q6")) == ONE_CODE_ANSWER


def test_code_read_as_several_layouts_offers_what_marks_several_answers() -> None:
    assert names(Position(QUESTIONS, "Q5")) == [
        "best-answer",
        "full-marks",
        "combination",
        "not-allowed",
        "custom-code",
    ]


def test_a_check_inside_best_answer_reads_one_of_the_layouts() -> None:
    assert names(Position(QUESTIONS, "Q5", one_answer=True)) == ONE_CODE_ANSWER


def test_a_method_made_of_checks_fits_answers_read_as_its_checks_read_them() -> None:
    text = """label: Best code
version: 1
question_types: [code]
checks:
  - uses: gradeflow/best-answer@1
    with:
      check: {uses: gradeflow/accepted-answers@1, with: {answers: ["print(1)"]}}
"""
    plain = text.replace("best-answer@1", "accepted-answers@1").replace(
        """    with:
      check: {uses: gradeflow/accepted-answers@1, with: {answers: ["print(1)"]}}""",
        """    with: {answers: ["print(1)"]}""",
    )
    methods = BUILTINS.with_files([("sample/best", text), ("sample/plain", plain)])
    best, exact = methods.find("sample/best@1"), methods.find("sample/plain@1")
    guessed, typed = Position(QUESTIONS, "Q5"), Position(QUESTIONS, "Q6")
    assert (best.fits(guessed), best.fits(typed)) == (True, False)
    assert (exact.fits(guessed), exact.fits(typed)) == (False, True)
