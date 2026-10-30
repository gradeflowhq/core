import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gradeflow_core.ml import cache_directory, nli

SCORES = {("a", "x"): 0.2, ("x", "a"): -0.5, ("a", "y"): 0.9, ("y", "a"): 0.8, ("a", "z"): -0.3}


@pytest.fixture
def agreement(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Directed pairs the model is asked to score, from ``SCORES``."""
    calls: list[tuple[str, str]] = []

    def score(premise: str, hypothesis: str) -> float:
        calls.append((premise, hypothesis))
        return SCORES.get((premise, hypothesis), 1.0)

    monkeypatch.setattr(nli, "_agreement", score)
    return calls


def test_an_answer_must_mean_the_same_both_ways(agreement: list[tuple[str, str]]) -> None:
    assert nli.best_match("a", ["x", "y"], "", 0.1) == (1, True)
    assert agreement == [("a", "x"), ("x", "a"), ("a", "y"), ("y", "a")]


def test_without_a_match_the_closest_is_named_without_reversing_failed_pairs(
    agreement: list[tuple[str, str]],
) -> None:
    assert nli.best_match("a", ["z", "x"], "", 0.5) == (1, False)
    assert agreement == [("a", "z"), ("a", "x")]


def test_the_question_comes_before_both_texts(agreement: list[tuple[str, str]]) -> None:
    nli.best_match("a", ["b"], "Why?", 0)
    assert agreement == [
        ("Question: Why?\nAnswer: a", "Question: Why?\nAnswer: b"),
        ("Question: Why?\nAnswer: b", "Question: Why?\nAnswer: a"),
    ]


def test_the_question_is_read_as_plain_text(agreement: list[tuple[str, str]]) -> None:
    nli.best_match("a", ["b"], r"2\. Is $\frac{1}{2}$ half?", 0)
    assert agreement[0][0] == "Question: 2. Is 1/2 half?\nAnswer: a"


def test_an_answer_says_a_point_in_one_direction(agreement: list[tuple[str, str]]) -> None:
    assert nli.says("a", "x", "", 0.2)
    assert not nli.says("a", "z", "", -0.2)
    assert agreement == [("a", "x"), ("a", "z")]


def test_the_question_comes_before_the_answer_and_the_point(
    agreement: list[tuple[str, str]],
) -> None:
    nli.says("a", "b", r"Is $\frac{1}{2}$ half?", 0)
    assert agreement == [("Question: Is 1/2 half?\nAnswer: a", "Question: Is 1/2 half?\nAnswer: b")]


@pytest.mark.parametrize(
    ("entailment", "contradiction", "expected"),
    [(2, 0, 1 / 3), (0, 2, -1 / 3), (1, 0, 1 / 6)],
)
def test_agreement_is_entailment_minus_contradiction(
    monkeypatch: pytest.MonkeyPatch, entailment: int, contradiction: int, expected: float
) -> None:
    np = pytest.importorskip("numpy")
    pairs: list[tuple[str, str]] = []

    def encode(premise: str, hypothesis: str) -> SimpleNamespace:
        pairs.append((premise, hypothesis))
        return SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1], type_ids=[0, 0, 1])

    def run(outputs: None, feed: dict[str, Any]) -> list[Any]:
        assert outputs is None
        assert set(feed) == {"input_ids", "attention_mask"}
        for name, values in {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}.items():
            assert feed[name].dtype == np.int64
            np.testing.assert_array_equal(feed[name], [values])
        # A large common offset checks numerical stability; softmax gives [1/6, 2/6, 3/6].
        return [np.asarray([[1000, 1000 + math.log(2), 1000 + math.log(3)]])]

    model = nli._Model(
        SimpleNamespace(run=run),
        SimpleNamespace(encode=encode),
        frozenset({"input_ids", "attention_mask"}),
        entailment,
        contradiction,
    )
    monkeypatch.setattr(nli, "_model", lambda: model)
    assert nli._agreement.__wrapped__("premise", "hypothesis") == pytest.approx(expected)
    assert pairs == [("premise", "hypothesis")]


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("GRADEFLOW_CACHE_DIR", str(tmp_path))
    return nli._directory()


@pytest.fixture
def prepared_cache(monkeypatch: pytest.MonkeyPatch, cache: Path) -> Path:
    cache.mkdir(parents=True)
    pinned = dict(nli.DOWNLOADS)
    for name in ("config.json", "tokenizer.json"):
        (cache / name).write_text(name)
        pinned[name] = hashlib.sha256(name.encode()).hexdigest()
    for name in (nli.GRAPH, f"{nli.GRAPH}.data"):
        (cache / name).write_text(name)
    monkeypatch.setattr(nli, "DOWNLOADS", pinned)
    return cache


def test_setup_downloads_nothing_once_prepared(
    monkeypatch: pytest.MonkeyPatch, prepared_cache: Path
) -> None:
    hub = pytest.importorskip("huggingface_hub")

    def forbidden(*args: Any, **kwargs: Any) -> str:
        pytest.fail("Prepared files need no download.")

    monkeypatch.setattr(hub, "hf_hub_download", forbidden)
    nli.setup()


@pytest.mark.parametrize("previous", ["unversioned", "different-source"])
def test_setup_replaces_a_previously_prepared_model(
    monkeypatch: pytest.MonkeyPatch, prepared_cache: Path, previous: str
) -> None:
    hub = pytest.importorskip("huggingface_hub")
    if previous == "unversioned":
        legacy = cache_directory() / nli.REPOSITORY.replace("/", "--")
        for path in prepared_cache.iterdir():
            path.rename(legacy / path.name)
    else:
        monkeypatch.setitem(nli.DOWNLOADS, nli.SOURCE, "0" * 64)

    def download(repository: str, name: str, *, revision: str, local_dir: Path) -> str:
        assert not nli._prepared(local_dir)
        raise RuntimeError("A fresh model download was requested.")

    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="A fresh model download was requested"):
        nli.setup()


def test_setup_refuses_a_file_that_is_not_the_pinned_one(
    monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    hub = pytest.importorskip("huggingface_hub")

    def download(repository: str, name: str, *, revision: str, local_dir: Path) -> str:
        assert (repository, revision) == (nli.REPOSITORY, nli.REVISION)
        path = Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tampered")
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="isn't the expected file"):
        nli.setup()


ENZYME = "What happens to an enzyme's activity at a temperature well above its optimum?"
REFERENCE = "The enzyme denatures because its active site changes shape, so its activity falls."


@pytest.mark.ml
@pytest.mark.parametrize(
    ("answer", "matches"),
    [
        (
            "The active site loses its shape, the enzyme is denatured and stops working well.",
            True,
        ),
        (
            "The enzyme is not denatured, so its activity keeps increasing.",
            False,
        ),
        ("Enzymes are made in the ribosomes.", False),
    ],
)
def test_the_model_distinguishes_paraphrases_from_wrong_answers(answer: str, matches: bool) -> None:
    pytest.importorskip("onnxruntime")
    # INT8 scores vary across CPU kernels; check decisions at the normal matching cutoff.
    assert nli.best_match(answer, [REFERENCE], ENZYME, 0.005) == (0, matches)
    assert nli.best_match(REFERENCE, [answer], ENZYME, 0.005) == (0, matches)


@pytest.mark.ml
@pytest.mark.parametrize(
    ("answer", "said"),
    [
        ("The active site changes shape, so the enzyme stops working.", True),
        ("The enzyme works faster and faster as it gets hotter.", False),
    ],
)
def test_the_model_tells_whether_an_answer_says_a_point(answer: str, said: bool) -> None:
    pytest.importorskip("onnxruntime")
    assert nli.says(answer, "The enzyme's active site changes shape.", ENZYME, 0.1) is said
