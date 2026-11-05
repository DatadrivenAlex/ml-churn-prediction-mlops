"""Experiment logging with a file backend and an optional MLflow backend.

``auto`` writes JSON files unless ``CHURN_TRACKING=mlflow``. If that variable
is set and MLflow is not installed, logging falls back to files so a laptop
and CI can both finish offline. Selecting ``backend="mlflow"`` directly is
stricter and raises when the package is missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

from churn.utils import jsonable, utc_now

logger = logging.getLogger(__name__)


class ExperimentTracker(Protocol):
    """Minimal tracking surface used by training."""

    def start_run(self, run_name: str) -> str: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def end_run(self) -> None: ...


class FileExperimentTracker:
    """One directory per run: ``meta.json``, ``params.json``, ``metrics.json``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.run_dir: Path | None = None
        self.run_id: str | None = None

    def start_run(self, run_name: str) -> str:
        stamp = utc_now().replace(":", "").replace("-", "").replace("+", "")
        suffix = uuid.uuid4().hex[:8]
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", run_name).strip("-") or "run"
        self.run_id = f"{stamp}-{safe_name}-{suffix}"
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._write_meta(
            {
                "run_id": self.run_id,
                "run_name": run_name,
                "backend": "file",
                "status": "running",
                "started_at": utc_now(),
            }
        )
        return self.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        self._require_open_run()
        assert self.run_dir is not None
        payload = json.dumps(jsonable(params), indent=2) + "\n"
        (self.run_dir / "params.json").write_text(payload, encoding="utf-8")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._require_open_run()
        assert self.run_dir is not None
        clean = {str(key): float(value) for key, value in metrics.items()}
        payload = json.dumps(clean, indent=2) + "\n"
        (self.run_dir / "metrics.json").write_text(payload, encoding="utf-8")

    def end_run(self) -> None:
        self._require_open_run()
        assert self.run_dir is not None
        meta = json.loads((self.run_dir / "meta.json").read_text(encoding="utf-8"))
        meta["status"] = "finished"
        meta["ended_at"] = utc_now()
        self._write_meta(meta)
        self.run_dir = None

    def _require_open_run(self) -> None:
        if self.run_dir is None:
            raise RuntimeError("start_run must be called before logging")

    def _write_meta(self, meta: dict) -> None:
        assert self.run_dir is not None
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )


class MLflowExperimentTracker:
    """Thin adapter. Pass ``mlflow_module`` in tests to avoid the dependency."""

    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "churn",
        mlflow_module: Any | None = None,
    ):
        if mlflow_module is None:
            try:
                import mlflow as mlflow_module
            except ImportError as exc:
                raise RuntimeError(
                    "MLflow is not installed. Use the file tracker, or install the "
                    "optional extra: pip install 'churn-mlops[mlflow]'."
                ) from exc
        self.mlflow = mlflow_module
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self._active = False

    def start_run(self, run_name: str) -> str:
        run = self.mlflow.start_run(run_name=run_name)
        self._active = True
        return str(run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        self._require_open_run()
        self.mlflow.log_params({str(key): _mlflow_param(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._require_open_run()
        self.mlflow.log_metrics({str(key): float(value) for key, value in metrics.items()})

    def end_run(self) -> None:
        self._require_open_run()
        self.mlflow.end_run()
        self._active = False

    def _require_open_run(self) -> None:
        if not self._active:
            raise RuntimeError("start_run must be called before logging")


def get_tracker(backend: str, experiments_dir: Path) -> ExperimentTracker:
    """Return the requested tracker. ``auto`` prefers files unless overridden."""
    if backend == "file":
        return FileExperimentTracker(experiments_dir)
    if backend == "mlflow":
        return MLflowExperimentTracker()
    if backend == "auto":
        choice = os.environ.get("CHURN_TRACKING", "file").strip().lower()
        if choice == "mlflow":
            try:
                return MLflowExperimentTracker()
            except RuntimeError:
                logger.warning("CHURN_TRACKING=mlflow but MLflow is unavailable; using files")
                return FileExperimentTracker(experiments_dir)
        if choice not in {"", "file"}:
            raise ValueError(f"Unknown CHURN_TRACKING value: {choice}")
        return FileExperimentTracker(experiments_dir)
    raise ValueError(f"Unknown tracking backend: {backend}")


def _mlflow_param(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(jsonable(value))
    return str(text)[:500]
