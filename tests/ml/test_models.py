import subprocess
import sys
from pathlib import Path

import pytest

from gradeflow_core import __main__ as cli
from gradeflow_core.ml import MISSING_MODEL, cache_directory, embeddings, nli


@pytest.mark.parametrize(
    ("xdg", "override", "relative"),
    [
        (None, None, ".cache/gradeflow/models"),
        ("xdg-cache", None, "xdg-cache/gradeflow/models"),
        ("xdg-cache", "chosen-cache", "chosen-cache/models"),
        (None, "~/chosen-cache", "chosen-cache/models"),
    ],
)
def test_models_are_kept_in_the_configured_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    xdg: str | None,
    override: str | None,
    relative: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("GRADEFLOW_CACHE_DIR", raising=False)
    if xdg:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / xdg))
    if override:
        monkeypatch.setenv(
            "GRADEFLOW_CACHE_DIR",
            override if override.startswith("~") else str(tmp_path / override),
        )
    assert cache_directory() == tmp_path / relative


def test_importing_the_core_does_not_load_optional_model_packages() -> None:
    optional = ["fastembed", "huggingface_hub", "numpy", "onnxruntime", "tokenizers"]
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import gradeflow_core; assert not {{*sys.modules}} & {set(optional)}",
        ],
        check=True,
    )


def test_setup_command_prepares_every_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared: list[str] = []
    monkeypatch.setenv("GRADEFLOW_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(embeddings, "setup", lambda: prepared.append("embeddings"))
    monkeypatch.setattr(nli, "setup", lambda: prepared.append("nli"))
    assert cli.main(["setup"]) == 0
    assert prepared == ["embeddings", "nli"]
    assert str(tmp_path / "models") in capsys.readouterr().out


def test_setup_command_explains_a_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing() -> Path:
        raise ImportError(MISSING_MODEL)

    monkeypatch.setattr(cli, "setup_models", missing)
    with pytest.raises(SystemExit) as error:
        cli.main(["setup"])
    assert error.value.code == 1
    assert MISSING_MODEL in capsys.readouterr().err
