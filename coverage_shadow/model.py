"""
Coverage Shadow v1 — closing-time model.

Idea: at the moment the ball is released, every player is in a race to the
ball's landing spot. The targeted receiver's lead over the fastest-arriving
defender is the "catch window". A defender's Shadow on a play is how much
that window would grow if they were removed from the field.
"""
import numpy as np
import pandas as pd

MAX_SPEED = 9.0   # yards/sec, approximate NFL top speed
REACTION = 0.3    # seconds to redirect toward the ball


def load_week(path: str) -> pd.DataFrame:
    """Load one Big Data Bowl 2026 input_<season>_w<week>.csv file."""
    return pd.read_csv(path)


def throw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last tracked frame of each play (the ball-release moment)."""
    idx = df.groupby(["game_id", "play_id"])["frame_id"].idxmax()
    keys = df.loc[idx, ["game_id", "play_id", "frame_id"]]
    return df.merge(keys, on=["game_id", "play_id", "frame_id"]).copy()


def time_to_ball(snap: pd.DataFrame) -> pd.Series:
    """
    Estimated seconds for each player to reach the ball landing spot.
    Uses current closing speed (velocity component toward the ball) ramping
    up to MAX_SPEED, plus a fixed reaction penalty.
    """
    dx = snap["ball_land_x"] - snap["x"]
    dy = snap["ball_land_y"] - snap["y"]
    dist = np.hypot(dx, dy)
    rad = np.deg2rad(snap["dir"])          # NFL dir: 0 = +y, 90 = +x
    vx, vy = snap["s"] * np.sin(rad), snap["s"] * np.cos(rad)
    safe = np.where(dist > 0, dist, 1.0)
    toward = np.where(dist > 0, (vx * dx + vy * dy) / safe, snap["s"]).clip(min=0)
    return REACTION + dist / ((toward + MAX_SPEED) / 2)


def score_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (play, defender) with that defender's Shadow in seconds.
    v1: only the closest-arriving defender earns credit on a play.
    """
    snap = throw_frame(df)
    snap["ttb"] = time_to_ball(snap)
    rows = []
    for (g, p), grp in snap.groupby(["game_id", "play_id"]):
        tgt = grp[grp["player_role"] == "Targeted Receiver"]
        defs = grp[grp["player_side"] == "Defense"]
        if len(tgt) != 1 or defs.empty:
            continue
        t = tgt["ttb"].iloc[0]
        d = defs["ttb"].to_numpy()
        window = d.min() - t
        for i, (_, r) in enumerate(defs.iterrows()):
            others = np.delete(d, i)
            window_without = (others.min() if len(others) else 99.0) - t
            rows.append({
                "game_id": g, "play_id": p,
                "nfl_id": r["nfl_id"], "player_name": r["player_name"],
                "position": r["player_position"],
                "ttb": r["ttb"], "target_ttb": t,
                "catch_window": window,
                "shadow": window_without - window,
                "closest": bool(i == d.argmin()),
            })
    return pd.DataFrame(rows)


def leaderboard(scored: pd.DataFrame, min_plays: int = 10) -> pd.DataFrame:
    lb = (scored.groupby(["nfl_id", "player_name", "position"])
          .agg(plays=("shadow", "size"),
               total_shadow=("shadow", "sum"),
               avg_shadow=("shadow", "mean"),
               times_closest=("closest", "sum"))
          .reset_index())
    return (lb[lb["plays"] >= min_plays]
            .sort_values("total_shadow", ascending=False)
            .reset_index(drop=True))
