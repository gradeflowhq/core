"""Whether two answers to a question say the same thing, judged by a small natural language
inference model run locally."""

import functools
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..common.text import unescaped_markdown
from .local import MISSING_MODEL_MESSAGE, cache_directory

if TYPE_CHECKING:
    from tokenizers import Tokenizer

REPOSITORY = "cross-encoder/nli-deberta-v3-xsmall"
REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
# Unsigned weights avoid U8S8 saturation on AVX2 CPUs; this graph also runs on ARM.
DOWNLOADED_MODEL = "onnx/model_quint8_avx2.onnx"
DOWNLOAD_DIGESTS = {
    "config.json": "8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342",
    "tokenizer.json": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    DOWNLOADED_MODEL: "21b14751a95520953bfcc607ceeb617de7cbeaeb6d60f4c8966716c743985337",
}
# The graph as ONNX Runtime optimizes it, with its weights in a file of their own. Loading it
# neither optimizes again nor keeps a second copy of the weights: 150 MB less, same scores.
OPTIMIZED_MODEL = "model.onnx"
MAX_TOKENS = 512
INTRA_OP_THREADS = 4


def setup() -> None:
    """Download the model, check it, and save it optimized for loading."""
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(MISSING_MODEL_MESSAGE) from error
    directory = _model_directory()
    if _is_prepared(directory):
        return
    for name, digest in DOWNLOAD_DIGESTS.items():
        path = Path(hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=directory))
        if not _matches_digest(path, digest):
            raise RuntimeError(f"{path} isn't the expected file; delete it and run setup again.")
    source = directory / DOWNLOADED_MODEL
    options = onnxruntime.SessionOptions()
    # The level below ENABLE_ALL, whose layout changes suit only the CPU that saves the graph.
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(directory / OPTIMIZED_MODEL)
    options.add_session_config_entry(
        "session.optimized_model_external_initializers_file_name", f"{OPTIMIZED_MODEL}.data"
    )
    options.add_session_config_entry(
        "session.optimized_model_external_initializers_min_size_in_bytes", "1024"
    )
    onnxruntime.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    source.unlink()
    source.parent.rmdir()


def is_available() -> bool:
    """Whether the packages to set up and run the model are installed."""
    return all(
        find_spec(name) for name in ("huggingface_hub", "numpy", "onnxruntime", "tokenizers")
    )


def first_match_index(
    answer: str, references: Sequence[str], question_markdown: str, cutoff: float
) -> int | None:
    """The index of the first reference meaning the same as ``answer`` at ``cutoff``, if any.

    The score is the lower, over both directions, of entailment minus contradiction, from -1 to
    1. ``question_markdown``, the question in Markdown, comes before both texts as plain text when
    given.
    """
    question_text = unescaped_markdown(question_markdown)
    answer_with_question = _with_question(answer, question_text)
    for index, reference in enumerate(
        _with_question(reference, question_text) for reference in references
    ):
        if (
            _entailment_score(answer_with_question, reference) >= cutoff
            and _entailment_score(reference, answer_with_question) >= cutoff
        ):
            return index
    return None


def entails(answer: str, point: str, question_markdown: str, cutoff: float) -> bool:
    """Whether ``answer`` entails ``point`` at ``cutoff``: entailment minus contradiction, with the
    whole answer as premise, each framed by ``question_markdown`` as in ``first_match_index``."""
    question_text = unescaped_markdown(question_markdown)
    return (
        _entailment_score(
            _with_question(answer, question_text), _with_question(point, question_text)
        )
        >= cutoff
    )


def _with_question(text: str, question_text: str) -> str:
    return f"Question: {question_text}\nAnswer: {text}" if question_text else text


@dataclass(frozen=True)
class _Model:
    session: Any
    tokenizer: Tokenizer
    input_names: frozenset[str]
    entailment_index: int
    contradiction_index: int


@functools.cache
def _model() -> _Model:
    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError(MISSING_MODEL_MESSAGE) from error
    directory = _model_directory()
    if not _is_prepared(directory):
        setup()
    config = json.loads((directory / "config.json").read_text())
    labels = {label: int(index) for index, label in config["id2label"].items()}
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(MAX_TOKENS, strategy="longest_first")
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(directory / OPTIMIZED_MODEL), options, providers=["CPUExecutionProvider"]
    )
    return _Model(
        session,
        tokenizer,
        frozenset(session_input.name for session_input in session.get_inputs()),
        labels["entailment"],
        labels["contradiction"],
    )


@functools.lru_cache(maxsize=4096)
def _entailment_score(premise: str, hypothesis: str) -> float:
    """Entailment minus contradiction, one pair per call: padded INT8 batches change verdicts."""
    import numpy as np

    model = _model()
    encoding = model.tokenizer.encode(premise, hypothesis)
    model_inputs = {
        "input_ids": encoding.ids,
        "attention_mask": encoding.attention_mask,
        "token_type_ids": encoding.type_ids,
    }
    [logits] = model.session.run(
        None, {name: np.asarray([model_inputs[name]], dtype=np.int64) for name in model.input_names}
    )[0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return float(probabilities[model.entailment_index]) - float(
        probabilities[model.contradiction_index]
    )


def _model_directory() -> Path:
    return cache_directory() / REPOSITORY.replace("/", "--") / DOWNLOAD_DIGESTS[DOWNLOADED_MODEL]


def _is_prepared(directory: Path) -> bool:
    return all(
        _matches_digest(directory / name, DOWNLOAD_DIGESTS[name])
        for name in ("config.json", "tokenizer.json")
    ) and all((directory / name).is_file() for name in (OPTIMIZED_MODEL, f"{OPTIMIZED_MODEL}.data"))


def _matches_digest(path: Path, digest: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == digest
