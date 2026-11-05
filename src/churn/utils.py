"""Small JSON and time helpers shared by tracking, the registry, and reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    """ISO-8601 timestamp in UTC with second resolution."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    """Convert numpy and pathlib values into objects ``json.dumps`` can write."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)
