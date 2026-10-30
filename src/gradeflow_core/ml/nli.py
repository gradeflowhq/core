"""Whether two answers to a question say the same thing, judged by a small natural language
inference model run locally."""

import functools
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..common.text import plain_text
from . import MISSING_MODEL, cache_directory

REPOSITORY = "cross-encoder/nli-deberta-v3-xsmall"
REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
# Unsigned weights avoid U8S8 saturation on AVX2 CPUs; this graph also runs on ARM.
SOURCE = "onnx/model_quint8_avx2.onnx"
DOWNLOADS = {
    "config.json": "8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342",
    "tokenizer.json": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    SOURCE: "21b14751a95520953bfcc607ceeb617de7cbeaeb6d60f4c8966716c743985337",
}
# The graph as ONNX Runtime optimizes it, with its weights in a file of their own. Loading it
# neither optimizes again nor keeps a second copy of the weights: 150 MB less, same scores.
GRAPH = "model.onnx"
MAX_TOKENS = 512
THREADS = 4


def setup() -> None:
    """Download the model, check it, and save it optimized for loading."""
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(MISSING_MODEL) from error
    directory = _directory()
    if _prepared(directory):
        return
    for name, digest in DOWNLOADS.items():
        path = Path(hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=directory))
        if not _intact(path, digest):
            raise RuntimeError(f"{path} isn't the expected file; delete it and run setup again.")
    source = directory / SOURCE
    options = onnxruntime.SessionOptions()
    # The level below ENABLE_ALL, whose layout changes suit only the CPU that saves the graph.
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(directory / GRAPH)
    options.add_session_config_entry(
        "session.optimized_model_external_initializers_file_name", f"{GRAPH}.data"
    )
    options.add_session_config_entry(
        "session.optimized_model_external_initializers_min_size_in_bytes", "1024"
    )
    onnxruntime.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    source.unlink()
    source.parent.rmdir()


def available() -> bool:
    """Whether the packages to set up and run the model are installed."""
    return all(
        find_spec(name) for name in ("huggingface_hub", "numpy", "onnxruntime", "tokenizers")
    )


def best_match(
    answer: str, references: Sequence[str], context: str, cutoff: float
) -> tuple[int, bool]:
    """The first reference meaning the same as ``answer`` at ``cutoff``, or else the closest.

    The score is the lower, over both directions, of entailment minus contradiction, from -1 to
    1. ``context``, the question in Markdown, comes before both texts as plain text when given.
    """
    framed = _framing(context)
    closest, highest = 0, -2.0
    for index, reference in enumerate(references):
        score = _agreement(framed(answer), framed(reference))
        if score >= cutoff:
            score = min(score, _agreement(framed(reference), framed(answer)))
            if score >= cutoff:
                return index, True
        if score > highest:
            closest, highest = index, score
    return closest, False


def says(answer: str, point: str, context: str, cutoff: float) -> bool:
    """Whether ``answer`` says ``point`` at ``cutoff``: entailment minus contradiction, with the
    whole answer as premise, each framed by ``context`` as in ``best_match``."""
    framed = _framing(context)
    return _agreement(framed(answer), framed(point)) >= cutoff


def _framing(context: str) -> Callable[[str], str]:
    question = plain_text(context)
    return (lambda text: f"Question: {question}\nAnswer: {text}") if question else str


@dataclass(frozen=True)
class _Model:
    session: Any
    tokenizer: Any
    inputs: frozenset[str]
    entailment: int
    contradiction: int


@functools.cache
def _model() -> _Model:
    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError(MISSING_MODEL) from error
    directory = _directory()
    if not _prepared(directory):
        setup()
    config = json.loads((directory / "config.json").read_text())
    labels = {label: int(index) for index, label in config["id2label"].items()}
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(MAX_TOKENS, strategy="longest_first")
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = THREADS
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(directory / GRAPH), options, providers=["CPUExecutionProvider"]
    )
    return _Model(
        session,
        tokenizer,
        frozenset(given.name for given in session.get_inputs()),
        labels["entailment"],
        labels["contradiction"],
    )


@functools.lru_cache(maxsize=4096)
def _agreement(premise: str, hypothesis: str) -> float:
    """Entailment minus contradiction, one pair per call: padded INT8 batches change verdicts."""
    import numpy as np

    model = _model()
    encoding = model.tokenizer.encode(premise, hypothesis)
    feed = {
        "input_ids": encoding.ids,
        "attention_mask": encoding.attention_mask,
        "token_type_ids": encoding.type_ids,
    }
    [logits] = model.session.run(
        None, {name: np.asarray([feed[name]], dtype=np.int64) for name in model.inputs}
    )[0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return float(probabilities[model.entailment]) - float(probabilities[model.contradiction])


def _directory() -> Path:
    return cache_directory() / REPOSITORY.replace("/", "--") / DOWNLOADS[SOURCE]


def _prepared(directory: Path) -> bool:
    return all(
        _intact(directory / name, DOWNLOADS[name]) for name in ("config.json", "tokenizer.json")
    ) and all((directory / name).is_file() for name in (GRAPH, f"{GRAPH}.data"))


def _intact(path: Path, digest: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == digest
