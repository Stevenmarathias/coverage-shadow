"""
Role controls for Coverage Shadow: alignment at the snap and team scheme.

Tracking frame 1 of each play is the snap (offensive players' median speed
there is ~0.02 yd/s). Route runners, the passer and coverage defenders are
tracked; linemen are not. From frame 1, per defender:

    slot     inside the widest route runner on his side of the ball by more
             than SLOT_MARGIN yards (lateral distance from the passer, who
             lines up behind the ball); no route runner on his side -> outside
    depth    yards off the line of scrimmage (absolute_yardline_number),
             toward his own end zone
    cushion  yards to the nearest route runner

Scheme (supplementary file, per play): team_coverage_man_zone,
team_coverage_type, defensive_team.

controls_residuals() regresses play-level Shadow on these (OLS) and returns
the residual per defender-play, so a defender's average residual is his
Shadow relative to what his alignment and scheme mix would predict.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .model import _quiet_matmul

SLOT_MARGIN = 1.0
ROUTE_ROLES = {"Targeted Receiver", "Other Route Runner"}


def snap_alignment(week_df: pd.DataFrame) -> pd.DataFrame:
    """Alignment of every coverage defender at frame 1 of each play."""
    f1 = week_df[week_df["frame_id"] == 1]
    out = []
    for (g, p), grp in f1.groupby(["game_id", "play_id"]):
        qb = grp[grp["player_role"] == "Passer"]
        defs = grp[grp["player_side"] == "Defense"]
        rr = grp[grp["player_role"].isin(ROUTE_ROLES)]
        if len(qb) != 1 or defs.empty or rr.empty:
            continue
        qy = qb["y"].iloc[0]
        los = grp["absolute_yardline_number"].iloc[0]
        sign = 1.0 if grp["play_direction"].iloc[0] == "right" else -1.0
        r_off = rr["y"].to_numpy() - qy
        rxy = rr[["x", "y"]].to_numpy()
        for _, d in defs.iterrows():
            off = d["y"] - qy
            same = r_off[np.sign(r_off) == np.sign(off)]
            widest = np.abs(same).max() if len(same) else None
            slot = bool(widest is not None and abs(off) < widest - SLOT_MARGIN)
            cushion = float(np.hypot(rxy[:, 0] - d["x"], rxy[:, 1] - d["y"]).min())
            out.append({"game_id": g, "play_id": p, "nfl_id": d["nfl_id"],
                        "slot": float(slot), "depth": float((d["x"] - los) * sign),
                        "cushion": cushion})
    return pd.DataFrame(out)


def load_alignment(folder: Path, season: int) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(folder).glob(f"input_{season}_w*.csv")):
        wk = pd.read_csv(f, usecols=["game_id", "play_id", "nfl_id", "frame_id", "player_side",
                                     "player_role", "x", "y", "absolute_yardline_number",
                                     "play_direction"])
        frames.append(snap_alignment(wk))
    return pd.concat(frames, ignore_index=True)


def attach_scheme(df: pd.DataFrame, supp_path: Path) -> pd.DataFrame:
    s = pd.read_csv(supp_path, usecols=["game_id", "play_id", "team_coverage_man_zone",
                                        "team_coverage_type", "defensive_team"])
    return df.merge(s, on=["game_id", "play_id"], how="left")


BLOCKS = {
    "alignment": ["slot", "depth", "depth2", "cushion"],
    "alignment+scheme": ["slot", "depth", "depth2", "cushion", "man", "cov:*"],
    "alignment+scheme+team": ["slot", "depth", "depth2", "cushion", "man", "cov:*", "team:*"],
}


def design(df: pd.DataFrame, block: str) -> np.ndarray:
    cols = [np.ones(len(df))]
    for term in BLOCKS[block]:
        if term == "depth2":
            cols.append(df["depth"].to_numpy() ** 2 / 10.0)
        elif term == "man":
            cols.append((df["team_coverage_man_zone"] == "MAN_COVERAGE").to_numpy(float))
        elif term.endswith(":*"):
            src = {"cov": "team_coverage_type", "team": "defensive_team"}[term[:-2]]
            dummies = pd.get_dummies(df[src].fillna("NA"), drop_first=True, dtype=float)
            cols.extend(dummies.to_numpy().T)
        else:
            cols.append(df[term].to_numpy(float))
    return np.column_stack(cols)


def ols_residuals(df: pd.DataFrame, block: str, y: str = "shadow"):
    """(residuals, play-level R^2) of y on the block's controls."""
    X = design(df, block)
    yy = df[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    with _quiet_matmul():
        res = yy - X @ beta
    if not np.all(np.isfinite(res)):
        raise FloatingPointError("non-finite OLS residuals")
    return res, float(1 - res.var() / yy.var())
