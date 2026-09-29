"""
Throw depth in the expected-completion model.

v2's completion model uses catch window alone. Short throws complete far
more often than deep ones at the same window, so this module tests adding
air yards (`pass_length` in the Big Data Bowl supplementary file: yards
past the line of scrimmage to the target, negative behind it).

Model specs (all logistic, fit on the closest-defender row of every
completion / incompletion):
    "window"         sigmoid(a + b*cw)                  -- v2 as published
    "window+depth"   sigmoid(a + b*cw + c*air)
    "window+depth2"  sigmoid(a + b*cw + c*air + d*air^2) -- curvature check
Air yards enter in tens of yards to keep Newton-Raphson well conditioned.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .model import fit_logit, log_loss, predict_logit

SPECS = ("window", "window+depth", "window+depth2")
# Six folds of three consecutive weeks: held-out weeks never share a game
# (or a week's opponents / weather) with the weeks the model was fit on.
FOLDS = [tuple(range(w, w + 3)) for w in range(1, 19, 3)]


def find_supplementary(folder: Path) -> Path:
    for p in (folder / "supplementary_data.csv", folder.parent / "supplementary_data.csv"):
        if p.exists():
            return p
    raise FileNotFoundError(f"supplementary_data.csv not in {folder} or its parent")


def attach_pass_length(scored: pd.DataFrame, supp_path: Path) -> pd.DataFrame:
    supp = pd.read_csv(supp_path, usecols=["game_id", "play_id", "pass_length"])
    return scored.merge(supp, on=["game_id", "play_id"], how="left")


def contested(scored: pd.DataFrame) -> pd.DataFrame:
    """Closest-defender rows of completions and incompletions (one per play)."""
    out = scored[scored["closest"] & scored["pass_result"].isin(["C", "I"])].copy()
    out["completed"] = (out["pass_result"] == "C").astype(float)
    return out


def design(df: pd.DataFrame, spec: str) -> np.ndarray:
    cols = [np.ones(len(df)), df["catch_window"].to_numpy()]
    if spec in ("window+depth", "window+depth2"):
        air = df["pass_length"].to_numpy() / 10.0
        cols.append(air)
        if spec == "window+depth2":
            cols.append(air ** 2)
    elif spec != "window":
        raise ValueError(spec)
    return np.column_stack(cols)


def cross_validate(plays: pd.DataFrame, specs=SPECS) -> pd.DataFrame:
    """Held-out log loss per fold and spec, plus an intercept-only baseline."""
    y = plays["completed"].to_numpy()
    wk = plays["week"].astype(int).to_numpy()
    rows = []
    for fold in FOLDS:
        te = np.isin(wk, fold)
        row = {"fold": f"W{fold[0]}-{fold[-1]}", "n_test": int(te.sum()),
               "baseline": log_loss(y[te], np.full(te.sum(), y[~te].mean()))}
        for spec in specs:
            X = design(plays, spec)
            beta = fit_logit(X[~te], y[~te])
            row[spec] = log_loss(y[te], predict_logit(X[te], beta))
        rows.append(row)
    return pd.DataFrame(rows)


def add_soe(scored: pd.DataFrame, spec: str, beta: np.ndarray, col: str) -> pd.DataFrame:
    """SOE under `spec`: expected - completed on closest, resolved rows."""
    out = scored.copy()
    done = out["pass_result"].map({"C": 1.0, "I": 0.0})
    mask = out["closest"] & done.notna()
    exp = predict_logit(design(out.fillna({"pass_length": 0.0}), spec), beta)
    out[col] = np.where(mask, exp - done, np.nan)
    return out
