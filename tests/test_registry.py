"""Versioned registry: register, promote, reject unsafe version ids."""

import json
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from churn.registry.store import ModelRegistry


def _tiny_model():
    model = LogisticRegression(max_iter=50)
    model.fit([[0], [1], [0], [1]], [0, 1, 0, 1])
    return model


def test_register_promote_and_load(tmp_path: Path):
    registry = ModelRegistry(tmp_path)
    baseline = {"rows": 4, "features": {}}
    first = registry.register(_tiny_model(), {"model_name": "logistic_regression"}, baseline)
    second = registry.register(_tiny_model(), {"model_name": "gradient_boosting"}, baseline)
    assert first == "v001"
    assert second == "v002"
    with pytest.raises(LookupError):
        registry.load()
    registry.promote(second)
    model, metadata = registry.load()
    assert metadata["version"] == "v002"
    assert metadata["model_name"] == "gradient_boosting"
    assert model.predict([[1]])[0] in {0, 1}
    loaded_baseline = registry.load_baseline(version=first)
    assert loaded_baseline["rows"] == 4
    explicit, explicit_meta = registry.load(version=first)
    assert explicit_meta["version"] == "v001"
    assert explicit.predict([[0]]).shape == (1,)


def test_rejects_bad_names_and_duplicates(tmp_path: Path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.register(_tiny_model(), {}, {}, version="../etc")
    registry.register(_tiny_model(), {"model_name": "one"}, {"rows": 1}, version="v001")
    with pytest.raises(FileExistsError):
        registry.register(_tiny_model(), {"model_name": "one"}, {"rows": 1}, version="v001")
    with pytest.raises(LookupError):
        registry.promote("v009")
    with pytest.raises(ValueError):
        registry.promote("v001", alias="bad alias")
    with pytest.raises(LookupError):
        registry.resolve(alias="challenger")


def test_missing_artifact_is_an_error(tmp_path: Path):
    version_dir = tmp_path / "v001"
    version_dir.mkdir()
    index = {"versions": ["v001"], "aliases": {"champion": "v001"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path).load()
