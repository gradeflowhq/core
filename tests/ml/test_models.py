from pathlib import Path
from unittest.mock import Mock

import pytest

from gradeflow_core import __main__ as cli
from gradeflow_core.ml import MISSING_MODEL_MESSAGE, cache_directory, embeddings, nli


@pytest.mark.parametrize(
    ("xdg_cache_home", "cache_dir", "expected_path"),
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
    xdg_cache_home: str | None,
    cache_dir: str | None,
    expected_path: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("GRADEFLOW_CACHE_DIR", raising=False)
    if xdg_cache_home:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / xdg_cache_home))
    if cache_dir:
        monkeypatch.setenv(
            "GRADEFLOW_CACHE_DIR",
            cache_dir if cache_dir.startswith("~") else str(tmp_path / cache_dir),
        )
    assert cache_directory() == tmp_path / expected_path


def test_setup_command_prepares_every_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models_set_up: list[str] = []
    monkeypatch.setenv("GRADEFLOW_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(embeddings, "setup", lambda: models_set_up.append("embeddings"))
    monkeypatch.setattr(nli, "setup", lambda: models_set_up.append("nli"))
    assert cli.main(["setup"]) == 0
    assert models_set_up == ["embeddings", "nli"]
    assert str(tmp_path / "models") in capsys.readouterr().out


def test_setup_command_explains_a_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "setup_models", Mock(side_effect=ImportError(MISSING_MODEL_MESSAGE)))
    with pytest.raises(SystemExit) as caught:
        cli.main(["setup"])
    assert caught.value.code == 1
    assert MISSING_MODEL_MESSAGE in capsys.readouterr().err
