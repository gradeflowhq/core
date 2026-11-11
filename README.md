# GradeFlow Core

A Python library for marking students' answers. You give it questions, a mark scheme and answers,
and it gives back marks and feedback. Mark schemes are built from methods such as Accepted Answers
and Key Points, and methods can be combined into new ones.

`gradeflow_core` is for scripts. `gradeflow_core.editing` adds what an editor of questions, markings
and methods needs ([Editing](#editing)). Before 1.0.0, minor releases may break the API
([Versioning](#versioning)).

## Installation

Python 3.14 or newer:

```bash
uv add "gradeflow-core @ git+https://github.com/gradeflowhq/core@v0.1.0"
```

Two extras add model-based marking:

- `ml` runs local models, for Same meaning matches and grouping answers by meaning.
- `llm` calls an OpenAI-compatible endpoint, for Answer Criteria.

The `ml` models download on first use, into `GRADEFLOW_CACHE_DIR` or `~/.cache/gradeflow`. To fetch
them ahead of time:

```bash
python -m gradeflow_core setup
```

## Marking

```python
from gradeflow_core import AnswersCsv, MarkScheme, QuestionSet, ResultsCsv, mark

students = AnswersCsv(student_id_column="student_id", answer_columns=["Q1", "Q2"]).read(
    "student_id,Q1,Q2\n10000001,red,10\n10000002,blue,8\n10000003,red,9\n"
)
questions = QuestionSet().updated_from(students)
scheme = MarkScheme.from_yaml("""
- question: Q1
  uses: gradeflow/multiple-choice@1
  with: {correct: [red]}
- question: Q2
  uses: gradeflow/number-range@1
  with: {min_value: 9, max_value: 10}
""")
assert scheme.problems(questions) == []

marked = list(mark(questions, scheme, students))
for student in marked:
    print(student.sheet.student_id, student.total_marks, "/", student.max_marks)
    print("\n".join(map(str, student.report())))

print(ResultsCsv().write(questions, marked))
```

`mark()` yields each sheet as it is marked, as a `Marked`: the `sheet`, a `QuestionResult` per
question in `results`, and `max_marks`. `total_marks` adds the results up, and `report()` gives a
printable `ReportLine` per question. A `Submission` is an `AnswerSheet` with a `student_id`.

When a marking combines checks, a result's `breakdown` is a `CheckResult` that shows how each one
marked the answer, such as what each code test's expression gave.

A blank answer gets nothing, except from Full Marks. An answer the question can't read, a check that
fails to run, or a marking with problems leaves the answer unmarked, with the reason as feedback. A
question with no marking keeps the marks given on the sheet, if any.

Questions and mark schemes round-trip through `to_yaml()` and `from_yaml()`, or `dump()` and
`load()` for plain data:

```python
from gradeflow_core import NumberQuestion, QuestionSet

questions = QuestionSet({"Q1": NumberQuestion(max_marks=2)})
assert QuestionSet.from_yaml(questions.to_yaml()) == questions
```

YAML aliases are refused.

## Language models

Answer Criteria asks a language model which of the lecturer's criteria an answer meets. With
`marks_by: best_fit`, it asks which one fits best. The endpoint is passed to `mark()`, not written
into the marking:

```python
from gradeflow_core import (
    MarkScheme,
    ModelEndpoint,
    QuestionSet,
    Submission,
    best_models,
    list_models,
    mark,
    try_model,
    try_models,
)

questions = QuestionSet.load({"Q1": {"type": "text", "max_marks": 2}})
criteria = [
    {"text": "Says ice is less dense than water", "marks": 1},
    {"text": "Explains that water expands as it freezes", "marks": 1},
]
scheme = MarkScheme.load(
    [{"question": "Q1", "uses": "gradeflow/answer-criteria@1", "with": {"criteria": criteria}}]
)
endpoint = ModelEndpoint(url="https://models.example.com/v1", api_key="your-api-key")
fast_model, careful_model = best_models(try_models(endpoint))
endpoint = endpoint.model_copy(update={"fast_model": fast_model, "careful_model": careful_model})
students = [Submission(student_id="1", answers={"Q1": "Because ice is less dense than water."})]
for marked in mark(questions, scheme, students, model_endpoint=endpoint):
    assert marked.results["Q1"].proportion == 0.5

for model in list_models(endpoint):
    print(try_model(endpoint, "careful", model))
```

`try_models` tries the endpoint's newest models on Answer Criteria's own examples, as the fast model
and then as the careful one, until two in each role mark them all correctly. It yields each
`ModelTrial` as it ends, skips those passed as `tried` unless the service failed on them, and calls
`on_start` as each starts.
`best_models` picks, in each role, the one with the most answers right, then the fastest.
`list_models` lists the endpoint's models, and `try_model` tries one of them in one role.

Answers go 12 to a request (`answers_per_request`) and sheets 60 at a time (`chunk_size`), with
each `Progress` reported to `on_progress`. Without an endpoint, these answers are left unmarked.
`MarkScheme.needs_model_endpoint()` tells you whether a scheme needs one.

## Questions

A `QuestionSet` maps question IDs to questions, in the order they are given.

| Type | Class | An answer is |
|---|---|---|
| `text` | `TextQuestion` | Text |
| `number` | `NumberQuestion` | A number, with fractions, units (`9.81 m/s^2`) and powers of ten (`6.02 x 10^23`) |
| `choice` | `ChoiceQuestion` | One or more options, like `B` or `A, C` |
| `values` | `ValuesQuestion` | Several values |
| `code` | `CodeQuestion` | Code, as written |

`updated_from(sheets)` adds questions for new answers, adds the options students chose after those
listed, and drops questions nobody answered. Options keep the order they're listed in. Code Tests
and Code Uses can fix minor errors before marking, such as lost indentation, with `fixes`.

Answers are always strings. A values question converts to and from its values:

```python
from gradeflow_core import ValuesQuestion

question = ValuesQuestion.load({"value_types": ["number", "text"]})
question.answer_values("3, east")  # {"value_1": "3", "value_2": "east"}
question.answer_from({"value_1": "4", "value_2": "west"})  # "4,west"
```

A code answer of several files is one string in txtar format. Code Tests runs the first file, which
can import the others:

```python
from gradeflow_core import answer_from_files

answer = answer_from_files({"game.py": "from board import Board", "board.py": "class Board: ..."})
assert answer == "-- game.py --\nfrom board import Board\n-- board.py --\nclass Board: ...\n"
```

## Methods and checks

A `Method` is a way of marking, named with a version: `gradeflow/key-points@1`. A check is a method
with its settings, and a marking is a check on a question:

```yaml
question: Q3
uses: gradeflow/workflow@1
with:
  marks_by: any_correct
  steps:
    - name: Correct
      uses: gradeflow/accepted-answers@1
      with: {answers: ["9.81"]}
    - name: Sign error
      weight: 0.5
      uses: gradeflow/accepted-answers@1
      with: {answers: ["-9.81"]}
```

Steps run in order. A check gives marks. A transform reads the answer as its question reads it, and
gives another answer that later steps read as their `input`:

```yaml
question: Q4
uses: gradeflow/workflow@1
with:
  steps:
    - id: final
      uses: gradeflow/extract@1
      with: {type: number, pattern: "Answer:\\s*(.+)"}
    - input: "{{ steps.final.output }}"
      uses: gradeflow/accepted-answers@1
      with: {answers: ["12"]}
```

A step can also have an `if` condition, such as `{{ steps.id.output.correct }}`, and a `feedback`
template. A `weight` of 0 gives no marks. A marking without a `question` marks a question group, and
its steps name the questions they mark.

A `MarkScheme` is a list of markings, each a `Step`: what its YAML holds, with `settings` for its
`with`. `problems(questions)` lists what would stop it marking. Each `Problem` gives the JSON
Pointer of the check at fault, such as `/markings/0/with/steps/1`, and of the field in it, such as
`/with/answers/0`.

```python
from gradeflow_core import MarkScheme

scheme = MarkScheme.load(
    [{"question": "Q3", "uses": "gradeflow/accepted-answers@1", "with": {"answers": ["9.81"]}}]
)
[check] = scheme.markings
print(check.question, check.uses, check.settings, check.weight)
print(check.describe(), check.questions_marked(), check.questions_read())
assert scheme.dump() == [check.dump()]
```

## Method collections

`BUILTINS` is a `Methods` collection of the built-in methods. Anything that reads checks takes
`methods=` to use others. `with_files` adds methods written in YAML, named `namespace/name`:

```python
from gradeflow_core import BUILTINS, MarkScheme

term = """
version: 1
label: Term
summary: Names the term.
question_types: [text]
settings:
  term:
    title: Term
steps:
  - uses: gradeflow/accepted-answers@1
    with: {answers: ["{{ settings.term }}"]}
"""
methods = BUILTINS.with_files([("intro-biology/term", term)])
scheme = MarkScheme.from_yaml(
    """
- question: Q1
  uses: intro-biology/term@1
  with: {term: osmosis}
""",
    methods=methods,
)
print(methods.find("gradeflow/key-points@1").guide)
for method in methods.latest():
    print(method.ref, method.label, sorted(method.question_types))
```

A `Method` has a `name`, `version`, `ref`, `label`, `summary`, `guide`, `level` and `examples`.
It marks the `question_types` it lists, and a question group if it `marks_question_group`.
`needs_model_endpoint` says whether it asks a language model, and `gives_marks(settings)` whether a
step with those settings is a check.

Files can use each other. A file that doesn't load raises `MethodFileError`, with its `ref` and its
`problems`, each on its `line`.

A shorter version of the built-in Key Points, which is composed:

```yaml
version: 1
label: Key Points
summary: Marks for each point the answer includes.
question_types: [text]
settings:
  points:
    title: Points
  marks_by:
    title: How points count
    default: proportional
marks_by: "{{ settings.marks_by }}"
steps:
  - for_each: "{{ settings.points }}"
    uses: gradeflow/key-point@1
    with:
      point: "{{ item }}"
```

A declared setting takes the type of the field it fills, and is required without a `default`.
`for_each` repeats a check with `item` and `index`. With `marks_by: proportional`, weights share out
the marks. With `marks_by: total`, each weight is a number of marks, capped at full marks. With
`marks_by: any_correct`, a weight is the share of the marks a check can give; `all_correct` and
`none_correct` take only weights of 0 and 1. A file with `output: step_output` and
`output_step: "{{ steps.id.output }}"` is a transform. Templates are sandboxed Jinja, and `{{ … }}`
on its own keeps its value's type.

## Writing a method in Python

An `AnswerMethod` marks one answer. Its `settings` are a `Settings` model, which is what a check's
`with` holds. `describe` says what a check with those settings does, and `mark_answer` gives an
`Outcome`: the proportion of the marks the answer earns, from 0 to 1, and feedback.

```python
from pydantic import Field

from gradeflow_core import (
    BUILTINS,
    Answer,
    AnswerMethod,
    MarkScheme,
    Methods,
    Outcome,
    Question,
    QuestionSet,
    Settings,
    Submission,
    mark,
)


class StartsWithSettings(Settings):
    prefix: str = Field(title="Starts with")


class StartsWith(AnswerMethod[StartsWithSettings]):
    name = "examples/starts-with"
    version = 1
    label = "Starts With"
    summary = "Full marks if the answer starts with the text."
    question_types = frozenset({"text"})
    settings = StartsWithSettings

    def describe(self, settings: StartsWithSettings) -> str:
        return f"Starts with `{settings.prefix}`."

    def mark_answer(
        self, settings: StartsWithSettings, answer: Answer, question: Question
    ) -> Outcome:
        if str(answer).startswith(settings.prefix):
            return Outcome(1.0, "Correct.")
        return Outcome(0.0, f"Expected it to start with {settings.prefix}.")


methods = Methods([*BUILTINS, StartsWith()])
questions = QuestionSet.load({"Q1": {"type": "text"}})
scheme = MarkScheme.load(
    [{"question": "Q1", "uses": "examples/starts-with@1", "with": {"prefix": "photo"}}],
    methods=methods,
)
students = [Submission(student_id="1", answers={"Q1": "Photosynthesis"})]
[marked] = mark(questions, scheme, students)
assert marked.results["Q1"].feedback == "Correct."
```

The answer is an `Answer`, as the question reads it: text, lowercased unless the question keeps
case, a number, a set of options or a list of values. Blank and unreadable answers never reach
`mark_answer`, unless the method sets `marks_blank_answers = True` to mark blank ones.
`Outcome(0.0, reason, marked=False)` leaves an answer for a person to mark, and `details` holds
each `Detail` a breakdown shows for the check. `guide`, `level` and `examples` are optional.

A `TransformMethod` gives no marks. `transform` gives another answer, or `Unreadable` with the
reason there is none, and `output_question` gives the question that reads it:

```python
from pydantic import Field

from gradeflow_core import (
    BUILTINS,
    Answer,
    MarkScheme,
    Methods,
    Question,
    QuestionSet,
    Settings,
    Submission,
    TransformMethod,
    Unreadable,
    mark,
)


class AfterSettings(Settings):
    marker: str = Field(title="After")


class After(TransformMethod[AfterSettings]):
    name = "examples/after"
    version = 1
    label = "After"
    summary = "Finds what comes after the text."
    question_types = frozenset({"text"})
    settings = AfterSettings

    def describe(self, settings: AfterSettings) -> str:
        return f"Finds what comes after `{settings.marker}`."

    def output_question(self, settings: AfterSettings, question: Question) -> Question:
        return question

    def transform(
        self, settings: AfterSettings, answer: Answer, question: Question
    ) -> Answer | Unreadable:
        _, marker, rest = str(answer).partition(settings.marker)
        return rest.strip() if marker else Unreadable(f"Couldn't find {settings.marker}.")


scheme = MarkScheme.from_yaml(
    """
- question: Q1
  uses: gradeflow/workflow@1
  with:
    steps:
      - id: final
        uses: examples/after@1
        with: {marker: "answer:"}
      - input: "{{ steps.final.output }}"
        uses: gradeflow/accepted-answers@1
        with: {answers: ["12"]}
""",
    methods=Methods([*BUILTINS, After()]),
)
questions = QuestionSet.load({"Q1": {"type": "text"}})
students = [Submission(student_id="1", answers={"Q1": "3 × 4. Answer: 12"})]
[marked] = mark(questions, scheme, students)
assert marked.results["Q1"].proportion == 1
```

## Examples

Each method declares examples: questions, settings, and answers with the marks and feedback they
should get. `mark_examples` marks them, and each `MarkedExample` says whether each row matches:

```python
from gradeflow_core import mark_examples

for example in mark_examples("gradeflow/key-points@1"):
    print(example.description, [row.matches_expected for row in example.rows])
```

## Comparing answers

`compare_answers` groups similar answers, by `exact`, `normalized`, `spelling` or `meaning`. The
comparison can be stored and regrouped at any threshold without the models:

```python
from gradeflow_core import AnswerComparison, compare_answers

comparison = compare_answers(["water", "waters", "salt"], similarity="spelling")
stored = comparison.model_dump_json()
assert AnswerComparison.model_validate_json(stored).groups(threshold=0.85) == [[0, 1], [2]]
```

## Adapters

- `AnswersCsv` reads answers, and any marks already given, from CSV.
- `ResultsCsv` writes answers, marks, feedback, totals and reports to CSV.
- `read_answer_key` reads an ExamSoft answer key row as an `AnswerSheet`.
- `read_question_document` sets questions up from ExamSoft's Print/Export as DOC.

## Errors

The library raises `GradeFlowError`, with messages fit to show people. `InvalidError` carries the
`problems` with invalid data: each `Problem` has a `message`, the `step_path` and `field` at
fault and, if it was read from YAML, its `line`. `where` joins the two into one JSON Pointer.
`MethodFileError` is the `InvalidError` of a method file, which its `ref` names.

## Editing

`gradeflow_core.editing` is for tools that edit markings and methods field by field.

`MarkingDraft` edits a marking by the JSON Pointer of each check: `""` is the marking itself,
`/with/steps/0` its first step. It gives each check's schema, values, fitting methods and problems,
and can add, update, change the method of, move and remove checks:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import Condition, ConditionRow, MarkingDraft

questions = QuestionSet.load({"Q1": {"type": "text"}})
draft = MarkingDraft.new("gradeflow/workflow@1", questions, "Q1")
draft.update("", {"marks_by": "any_correct"})
draft.add("/with/steps/0", "gradeflow/accepted-answers@1")
draft.update("/with/steps/0", {"answers": ["Paris"]})
draft.add("/with/steps/1", "gradeflow/key-points@1")
incorrect = Condition((ConditionRow("/with/steps/0", "incorrect"),))
draft.update("/with/steps/1", {"points": ["capital"], "weight": 0.5, "if": incorrect})
assert draft.problems == []
print(draft.notes("/with/steps/1"))
print(draft.finalized().dump())
```

A check's `values` hold its settings and its own properties as one. `field_in_values` gives where a
problem's `field` is among them: `/with/answers/0` is `/answers/0`.

`MethodDraft` writes a composed method from example questions. Its steps are one Workflow at `""`,
edited as a marking's are: both drafts are a `Draft`. `bind_new_setting` turns a field of a step
into a declared setting, and `bind` and `unbind` connect or disconnect them. Its `problems` are its
`step_problems`, then the method's own. `tested_step` gives the method with its test values as a
`Step`:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import MethodDraft

draft = MethodDraft.new(
    QuestionSet.load({"Q1": {"type": "number"}}),
    name="examples/within-range",
    version=1,
    label="Within range",
    summary="A number in a range.",
)
draft.add("/with/steps/0", "gradeflow/number-range@1")
draft.update("/with/steps/0", {"min_value": 10, "max_value": 20})
draft.bind_new_setting("/with/steps/0/with/min_value")
assert draft.problems == []
print(draft.to_yaml())
```

`load_method_files` loads saved method files as far as it can. A file that no longer loads, such
as one written for an earlier version, stops only the files that use it, and `failed` holds its
`MethodFileError`:

```python
from gradeflow_core import BUILTINS
from gradeflow_core.editing import load_method_files

old = """\
version: 1
label: Old
summary: Written for an earlier version.
question_types: [text]
steps:
  - uses: gradeflow/key-points@1
    with: {points: [light], unknown_setting: true}
"""
loaded = load_method_files([("intro-biology/old", old)], BUILTINS)
print(*loaded.failed["intro-biology/old@1"].problems, sep="\n")
```

`compatible_methods` lists the methods that fit at a `Position`, and `settings_schema` gives a
method's settings there:

```python
from gradeflow_core import BUILTINS, QuestionSet
from gradeflow_core.editing import Position, compatible_methods

questions = QuestionSet.load({"Q1": {"type": "number"}})
print([method.label for method in compatible_methods(Position(questions, "Q1"))])
number_range = BUILTINS.find("gradeflow/number-range@1")
print(list(number_range.settings_schema(Position(questions, "Q1"))["properties"]))
```

`mark_rows` marks example rows with given settings. Each `RowResult` is a marked `ExampleRow`, and
`as_example()` records its results as what the row declares:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import ExampleRow, mark_rows

questions = QuestionSet.load({"Q1": {"type": "number"}})
rows = [ExampleRow(answers={"Q1": "15"}), ExampleRow(answers={"Q1": "25"})]
results = mark_rows(
    "gradeflow/number-range@1",
    questions=questions,
    question="Q1",
    settings={"min_value": 10, "max_value": 20},
    rows=rows,
)
assert [result.sheet for result in results] == rows
recorded = [result.as_example() for result in results]
```

`MarkingDraft.mark_by_example` changes the marking so that chosen answers get a decision:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import MarkingDraft

questions = QuestionSet.load({"Q1": {"type": "text"}})
draft = MarkingDraft.new("gradeflow/accepted-answers@1", questions, "Q1")
draft.update("", {"answers": ["Paris"]})
draft.mark_by_example(["Lyon"], "not_allowed")
assert draft.answer_decisions() == {"paris": "full_marks", "lyon": "not_allowed"}
```

The module also has functions for question types (`new_question`, `converted_question`), method
references (`method_ref`, `split_ref`, `rewrite_references`) and YAML (`load_yaml`, `dump_yaml`).

## Running code

Custom Code, Custom Code for a Group and Code Tests run CPython 3.14 in a fresh
[Eryx](https://github.com/eryx-org/eryx) WebAssembly sandbox, with the standard library but no host
files, packages or network. A run gets 5 seconds and 512 MiB of memory unless its check sets up to 60
seconds and 1 GiB, and 1 MiB each of output and returned values.

## Adding to the library

Built-in methods are in `src/gradeflow_core/methods/`, in `answers/`, `text/`, `code/` and `logic/`.

- **Composed method:** a YAML file named after the method, and its name in `CATALOGUE_ORDER` in
  `methods/catalogue.py`.
- **Coded method:** a method [written in Python](#writing-a-method-in-python), an instance in
  `PYTHON_METHODS`, its name in `CATALOGUE_ORDER`, and its examples in
  `<method-name>.examples.yaml` beside it.
- **Question type:** a subclass of `Question` implementing `read` and `python_type`, added to
  `AnyQuestion` in `questions/__init__.py` and to `QuestionType` in `questions/types.py`.

The tests check that each built-in's examples get the results they declare.

## Versioning

Semantic versioning. Before 1.0.0, minor releases (`0.x.0`) may break compatibility and patch
releases (`0.x.y`) don't. Compatibility covers both public modules, the saved formats of questions,
mark schemes and method files, and documented marking behaviour. Release notes list breaking changes
and any change to awarded marks.

## Development

```bash
uv sync --all-extras
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```

Tests that need the `ml` extra are marked `ml`.

## License

MIT.
