"""International football match outcome prediction package."""

from .modeling import (
    FEATURE_COLUMNS,
    OUTCOME_LABELS,
    build_team_stats,
    build_training_frame,
    load_artifacts,
    load_matches,
    predict_match,
    train_and_save,
)

__all__ = [
    "FEATURE_COLUMNS",
    "OUTCOME_LABELS",
    "build_team_stats",
    "build_training_frame",
    "load_artifacts",
    "load_matches",
    "predict_match",
    "train_and_save",
]

