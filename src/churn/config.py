"""Default paths. Override the registry with CHURN_REGISTRY_DIR when serving."""

from __future__ import annotations

import os
from pathlib import Path


def project_root(start_points: list[Path] | None = None) -> Path:
    """Repository root in an editable checkout, or the working directory in Docker."""
    if start_points is None:
        start_points = [Path(__file__).resolve(), Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in start_points:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "churn").is_dir():
                return candidate
    return Path.cwd().resolve()


def registry_dir() -> Path:
    """Champion artifacts. The environment variable wins over the repo default."""
    override = os.environ.get("CHURN_REGISTRY_DIR")
    if override:
        return Path(override)
    return project_root() / "model_registry"


PROJECT_ROOT = project_root()
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_SEED = 42
DEFAULT_ROWS = 8000
