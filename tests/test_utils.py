"""JSON helpers, project-root discovery, and the module entrypoint."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from churn.config import project_root, registry_dir
from churn.utils import jsonable


def test_jsonable_converts_numpy_and_paths():
    payload = jsonable(
        {
            "count": np.int64(3),
            "score": np.float64(0.5),
            "rows": np.array([1, 2]),
            "path": Path("model_registry"),
            "flag": True,
            "label": "ok",
            "empty": None,
            "items": (np.int64(4),),
        }
    )
    assert payload["count"] == 3
    assert payload["score"] == pytest.approx(0.5)
    assert payload["rows"] == [1, 2]
    assert payload["items"] == [4]
    assert payload["path"] == "model_registry"
    assert payload["flag"] is True
    assert payload["label"] == "ok"
    assert payload["empty"] is None
    assert isinstance(jsonable(object()), str)


def test_project_root_contains_the_package(tmp_path: Path):
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "churn").is_dir()
    nested = root / "src" / "churn" / "config.py"
    assert project_root([nested, root]) == root
    missing = tmp_path / "a" / "b"
    missing.mkdir(parents=True)
    assert project_root([missing, missing.parent]) == Path.cwd().resolve()


def test_registry_dir_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHURN_REGISTRY_DIR", str(tmp_path))
    assert registry_dir() == tmp_path
    monkeypatch.delenv("CHURN_REGISTRY_DIR", raising=False)
    assert registry_dir() == project_root() / "model_registry"


def test_module_entrypoints_delegate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("churn.cli.main", lambda: 3)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("churn", run_name="__main__")
    assert caught.value.code == 3
    monkeypatch.setattr("sys.argv", ["churn", "--help"])
    with pytest.raises(SystemExit) as caught_cli:
        runpy.run_path(
            str(project_root() / "src" / "churn" / "cli.py"),
            run_name="__main__",
        )
    assert caught_cli.value.code == 0
