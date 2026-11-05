"""Model training and operating-point selection."""

from churn.training.metrics import ThresholdChoice, evaluate_at_threshold, select_threshold
from churn.training.train import TrainConfig, run_training

__all__ = [
    "ThresholdChoice",
    "TrainConfig",
    "evaluate_at_threshold",
    "run_training",
    "select_threshold",
]
