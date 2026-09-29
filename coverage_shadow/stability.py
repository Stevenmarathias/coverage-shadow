"""
Stability of Coverage Shadow and SOE, and of the standard coverage stats.

A skill metric should agree with itself: a defender's value in one half of
the season should predict the other half. Split-half reliability compares
odd weeks with even weeks (9 each in 2023), and Spearman-Brown projects the
half-season correlation r to a full season: r_full = 2r / (1 + r).

Metrics per defender:
    avg_shadow       mean Shadow (s) over all coverage snaps       (v1)
    soe_per_contest  SOE / contests, contests = closest defender on a
                     completion or incompletion                     (v2)
Benchmark (nflverse PFR advanced stats, charted targets):
    cmp_pct_allowed        completions / targets
    passer_rating_allowed  NFL passer rating from summed components
"""
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
PFR_DEF = NFLVERSE + "/pfr_advstats/advstats_week_def_{season}.parquet"
ROSTER = NFLVERSE + "/rosters/roster_{season}.parquet"
CB_LABELS = {"CB", "LCB", "RCB", "NB"}


# ---------------------------------------------------------------- stats ----

def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def spearman(a, b) -> float:
    return pearson(pd.Series(a).rank(), pd.Series(b).rank())


def fisher_ci(r: float, n: int, z: float = 1.96):
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    f, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    return float(np.tanh(f - z * se)), float(np.tanh(f + z * se))


def spearman_brown(r: float, k: float = 2.0) -> float:
    """Reliability of a measure k times as long, from reliability r."""
    return k * r / (1 + (k - 1) * r)


def correlate(x, y) -> dict:
    n = len(x)
    r = pearson(x, y) if n > 2 else np.nan
    lo, hi = fisher_ci(r, n)
    return {"n": n, "r": r, "ci_lo": lo, "ci_hi": hi,
            "rho": spearman(x, y) if n > 2 else np.nan}


# ---------------------------------------------------- tracking metrics -----

def per_player(scored: pd.DataFrame, soe_col: str) -> pd.DataFrame:
    """avg_shadow and soe_per_contest per defender over `scored`'s rows."""
    keys = ["nfl_id", "player_name", "position"]
    base = (scored.groupby(keys)
            .agg(plays=("shadow", "size"), avg_shadow=("shadow", "mean"))
            .reset_index())
    c = scored[scored[soe_col].notna()]
    soe = (c.groupby(keys).agg(contests=(soe_col, "size"), soe=(soe_col, "sum"))
           .reset_index())
    out = base.merge(soe, on=keys, how="left").fillna({"contests": 0, "soe": 0.0})
    out["soe_per_contest"] = np.where(out["contests"] > 0,
                                      out["soe"] / out["contests"].clip(lower=1), np.nan)
    return out


def halves(scored: pd.DataFrame, soe_col: str):
    wk = scored["week"].astype(int)
    return (per_player(scored[wk % 2 == 1], soe_col),
            per_player(scored[wk % 2 == 0], soe_col))


def split_half(odd: pd.DataFrame, even: pd.DataFrame, metric: str, gate_col: str,
               gate: int, cb_only: bool) -> dict:
    """Odd- vs even-week correlation among defenders with >= gate of
    gate_col in BOTH halves, with the Spearman-Brown full-season value."""
    m = odd.merge(even, on=["nfl_id", "player_name", "position"], suffixes=("_o", "_e"))
    m = m[(m[f"{gate_col}_o"] >= gate) & (m[f"{gate_col}_e"] >= gate)]
    if cb_only:
        m = m[m["position"] == "CB"]
    m = m.dropna(subset=[f"{metric}_o", f"{metric}_e"])
    out = correlate(m[f"{metric}_o"], m[f"{metric}_e"])
    out["spearman_brown"] = spearman_brown(out["r"]) if np.isfinite(out["r"]) else np.nan
    out["pairs"] = m
    return out


# ------------------------------------------------------- PFR benchmark -----

def _cached(url: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / url.rsplit("/", 1)[-1]
    if not p.exists():
        p.write_bytes(urlopen(url, timeout=120).read())
    return p


def pfr_cb_weeks(season: int, cache: Path) -> pd.DataFrame:
    """Regular-season PFR weekly coverage rows for players the nflverse
    roster lists at CB that season."""
    d = pd.read_parquet(_cached(PFR_DEF.format(season=season), cache))
    d = d[(d["game_type"] == "REG") & d["pfr_player_id"].notna()]
    ro = pd.read_parquet(_cached(ROSTER.format(season=season), cache),
                         columns=["pfr_id", "position", "depth_chart_position"])
    cb = ro[ro["position"].isin(CB_LABELS) | ro["depth_chart_position"].isin(CB_LABELS)]
    return d[d["pfr_player_id"].isin(set(cb["pfr_id"].dropna()))].copy()


def _rating(c, a, y, t, i) -> np.ndarray:
    a = np.where(a > 0, a, np.nan)
    clip = lambda x: np.clip(x, 0, 2.375)  # noqa: E731
    return 100 * (clip((c / a - 0.3) * 5) + clip((y / a - 3) * 0.25)
                  + clip(t / a * 20) + clip(2.375 - i / a * 25)) / 6


def pfr_per_player(rows: pd.DataFrame) -> pd.DataFrame:
    g = (rows.groupby("pfr_player_id")
         .agg(name=("pfr_player_name", "first"),
              team=("team", lambda s: s.mode().iat[0]),
              teams=("team", "nunique"),
              targets=("def_targets", "sum"), cmp=("def_completions_allowed", "sum"),
              yds=("def_yards_allowed", "sum"), td=("def_receiving_td_allowed", "sum"),
              ints=("def_ints", "sum"))
         .reset_index())
    g["cmp_pct_allowed"] = np.where(g["targets"] > 0, g["cmp"] / g["targets"].clip(lower=1), np.nan)
    g["passer_rating_allowed"] = _rating(g["cmp"], g["targets"], g["yds"], g["td"], g["ints"])
    return g


def pfr_split_half(rows: pd.DataFrame, metric: str, gate: int) -> dict:
    o = pfr_per_player(rows[rows["week"] % 2 == 1])
    e = pfr_per_player(rows[rows["week"] % 2 == 0])
    m = o.merge(e, on="pfr_player_id", suffixes=("_o", "_e"))
    m = m[(m["targets_o"] >= gate) & (m["targets_e"] >= gate)].dropna(
        subset=[f"{metric}_o", f"{metric}_e"])
    out = correlate(m[f"{metric}_o"], m[f"{metric}_e"])
    out["spearman_brown"] = spearman_brown(out["r"]) if np.isfinite(out["r"]) else np.nan
    out["pairs"] = m
    return out


def pfr_year_over_year(a: pd.DataFrame, b: pd.DataFrame, metric: str, gate: int) -> dict:
    """Season-to-season correlation for CBs with >= gate targets in both,
    split by whether the player's (modal) team changed."""
    pa, pb = pfr_per_player(a), pfr_per_player(b)
    m = pa.merge(pb, on="pfr_player_id", suffixes=("_1", "_2"))
    m = m[(m["targets_1"] >= gate) & (m["targets_2"] >= gate)].dropna(
        subset=[f"{metric}_1", f"{metric}_2"])
    moved = (m["team_1"] != m["team_2"]) | (m["teams_1"] > 1) | (m["teams_2"] > 1)
    res = {"all": correlate(m[f"{metric}_1"], m[f"{metric}_2"])}
    res["same_team"] = correlate(m.loc[~moved, f"{metric}_1"], m.loc[~moved, f"{metric}_2"])
    res["changed_team"] = correlate(m.loc[moved, f"{metric}_1"], m.loc[moved, f"{metric}_2"])
    return res
