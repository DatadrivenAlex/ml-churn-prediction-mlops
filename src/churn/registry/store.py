"""Filesystem model registry.

Each version is a directory with ``model.joblib``, ``metadata.json``, and
``drift_baseline.json``. ``index.json`` records versions and aliases such as
``champion``. Artifacts are joblib/pickle: only load files this project wrote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib

from churn.utils import jsonable, utc_now

_VERSION_RE = re.compile(r"^v\d{3,}$")


class ModelRegistry:
    """Read and write versioned pipelines under a single root directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def next_version(self) -> str:
        numbers = [int(version[1:]) for version in self._read_index()["versions"]]
        return f"v{max(numbers, default=0) + 1:03d}"

    def register(
        self,
        model: Any,
        metadata: dict,
        baseline: dict,
        version: str | None = None,
    ) -> str:
        """Store a fitted pipeline and return the version id."""
        version = version or self.next_version()
        self._check_version(version)
        destination = self.root / version
        if destination.exists():
            raise FileExistsError(f"Model version already exists: {version}")
        destination.mkdir(parents=True)
        joblib.dump(model, destination / "model.joblib")
        payload = {**metadata, "version": version, "registered_at": utc_now()}
        _write_json(destination / "metadata.json", payload)
        _write_json(destination / "drift_baseline.json", baseline)
        index = self._read_index()
        index["versions"].append(version)
        self._write_index(index)
        return version

    def promote(self, version: str, alias: str = "champion") -> None:
        """Point ``alias`` at an existing version."""
        self._check_version(version)
        self._require_version(version)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,40}", alias):
            raise ValueError(f"Invalid alias: {alias}")
        index = self._read_index()
        index["aliases"][alias] = version
        self._write_index(index)

    def resolve(self, version: str | None = None, alias: str = "champion") -> str:
        """Return a concrete version id, following ``alias`` when needed."""
        if version is None:
            found = self._read_index()["aliases"].get(alias)
            if not found:
                raise LookupError(f"No model alias '{alias}' is registered")
            version = str(found)
        self._check_version(version)
        self._require_version(version)
        return version

    def load(self, version: str | None = None, alias: str = "champion") -> tuple[Any, dict]:
        """Load a pipeline and its metadata."""
        resolved = self.resolve(version=version, alias=alias)
        destination = self.root / resolved
        model = joblib.load(destination / "model.joblib")
        metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        return model, metadata

    def load_baseline(self, version: str | None = None, alias: str = "champion") -> dict:
        """Load the training-split drift baseline stored beside a model."""
        resolved = self.resolve(version=version, alias=alias)
        path = self.root / resolved / "drift_baseline.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _require_version(self, version: str) -> None:
        if version not in self._read_index()["versions"]:
            raise LookupError(f"Unknown model version: {version}")
        if not (self.root / version / "model.joblib").is_file():
            raise FileNotFoundError(f"Missing model artifact for {version}")

    def _read_index(self) -> dict:
        path = self.root / "index.json"
        if not path.is_file():
            return {"versions": [], "aliases": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("versions", [])
        payload.setdefault("aliases", {})
        return payload

    def _write_index(self, index: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        index["updated_at"] = utc_now()
        _write_json(self.root / "index.json", index)

    @staticmethod
    def _check_version(version: str) -> None:
        if not _VERSION_RE.fullmatch(version):
            raise ValueError(f"Version must look like v001, got {version!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
