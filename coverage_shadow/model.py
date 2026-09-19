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


# ---------------------------------------------------------------------------
# v2: outcome-aware Shadow. Grades contests instead of just counting geometry.
# Requires a "pass_result" column already merged onto the v1 output.
# ---------------------------------------------------------------------------

def fit_completion_model(scored: pd.DataFrame, max_iter: int = 50):
    """
    Fit P(complete) = sigmoid(a + b * catch_window) on the closest-defender
    row per play, restricted to completions and incompletions. Returns (a, b).
    Uses Newton-Raphson so we don't need sklearn/scipy.
    """
    plays = scored[scored["closest"] & scored["pass_result"].isin(["C", "I"])]
    y = (plays["pass_result"] == "C").astype(float).to_numpy()
    x = plays["catch_window"].to_numpy()
    a, b = 0.0, 0.0
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(a + b * x)))
        g0, g1 = (p - y).sum(), ((p - y) * x).sum()
        w = p * (1 - p)
        h00, h01, h11 = w.sum(), (w * x).sum(), (w * x * x).sum()
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        step_a = (h11 * g0 - h01 * g1) / det
        step_b = (-h01 * g0 + h00 * g1) / det
        a -= step_a
        b -= step_b
        if abs(step_a) + abs(step_b) < 1e-8:
            break
    return a, b


def add_v2_metrics(scored: pd.DataFrame, coef=None) -> pd.DataFrame:
    """
    Enrich the v1 per-defender-per-play frame with:
      - completed:             1 for C, 0 for I, NaN otherwise
      - expected_completion:   sigmoid(a + b * catch_window), in [0, 1]
      - shadow_over_expected:  expected - completed on closest+resolved rows,
                               i.e. completions prevented above expectation
                               on that play. Unitless; summed over a season
                               it reads as total completions prevented.

    The completion model uses catch_window as its only feature; adding throw
    depth is future work.
    """
    out = scored.copy()
    out["completed"] = out["pass_result"].map({"C": 1.0, "I": 0.0})
    a, b = coef if coef is not None else fit_completion_model(out)
    out["expected_completion"] = 1.0 / (1.0 + np.exp(-(a + b * out["catch_window"])))
    mask = out["closest"] & out["completed"].notna()
    out["shadow_over_expected"] = np.where(
        mask, out["expected_completion"] - out["completed"], np.nan
    )
    return out


def leaderboard_v2(scored_v2: pd.DataFrame, min_plays: int = 100) -> pd.DataFrame:
    """
    Per-defender v2 board. `plays` is coverage snaps (same units as v1's
    min_plays filter); won/lost/SOE aggregate only over plays where the
    defender was closest with a resolved outcome.

    Units: `shadow_won` and `shadow_lost` are in seconds (sums of Shadow).
    `shadow_over_expected` is in completions prevented above expectation.
    """
    base = (scored_v2.groupby(["nfl_id", "player_name", "position"])
            .agg(plays=("shadow", "size"),
                 total_shadow=("shadow", "sum"))
            .reset_index())

    contested = scored_v2[scored_v2["closest"] & scored_v2["completed"].notna()]
    keys = ["nfl_id", "player_name", "position"]
    agg = (contested.assign(
                incomplete_shadow=lambda d: np.where(d["completed"] == 0, d["shadow"], 0.0),
                complete_shadow=lambda d: np.where(d["completed"] == 1, d["shadow"], 0.0))
           .groupby(keys)
           .agg(contests=("shadow", "size"),
                shadow_won=("incomplete_shadow", "sum"),
                shadow_lost=("complete_shadow", "sum"),
                shadow_over_expected=("shadow_over_expected", "sum"))
           .reset_index())

    lb = base.merge(agg, on=keys, how="left")
    for col in ["contests", "shadow_won", "shadow_lost", "shadow_over_expected"]:
        lb[col] = lb[col].fillna(0.0)
    denom = lb["shadow_won"] + lb["shadow_lost"]
    lb["win_rate"] = np.where(denom > 0, lb["shadow_won"] / denom, np.nan)

    return (lb[lb["plays"] >= min_plays]
            .sort_values("shadow_over_expected", ascending=False)
            .reset_index(drop=True))
