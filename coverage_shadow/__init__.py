"""Coverage Shadow — quantifying how much catch window each defender erases."""
from .model import (
    load_week, throw_frame, time_to_ball, score_plays, leaderboard,
    fit_completion_model, add_v2_metrics, leaderboard_v2,
)
__all__ = [
    "load_week", "throw_frame", "time_to_ball", "score_plays", "leaderboard",
    "fit_completion_model", "add_v2_metrics", "leaderboard_v2",
]
