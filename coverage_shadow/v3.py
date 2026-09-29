"""
Coverage Shadow v3: split avg Shadow into involvement and effect.

Pre-registered (README, "v3") before any v3 number was computed:

    involvement  share of a defender's coverage snaps on which he is the
                 closest or second-closest defender to the landing spot at
                 release (by estimated time to ball, `ttb`)
    effect       mean window reduction on those involved plays, where a
                 defender's window reduction = the next defender's ttb minus
                 his own: how much sooner he gets there than the help behind
                 him. For the closest defender that is exactly his v1 Shadow;
                 for the second-closest it is the window he would erase if the
                 closest weren't there. (v1 Shadow alone is 0 for everyone but
                 the closest defender, so it can't grade second-closest plays.)
"""
import numpy as np
import pandas as pd


def add_v3(scored: pd.DataFrame) -> pd.DataFrame:
    """Adds ttb_rank (1 = closest), involved, and reduction per defender-play."""
    out = scored.sort_values(["game_id", "play_id", "ttb"]).copy()
    g = out.groupby(["game_id", "play_id"])["ttb"]
    out["ttb_rank"] = g.cumcount() + 1
    out["next_ttb"] = g.shift(-1)
    out["involved"] = out["ttb_rank"] <= 2
    # Every 2023 play has 3+ coverage defenders, so next_ttb always exists
    # for ranks 1-2; rank-1 reduction equals v1 Shadow exactly.
    out["reduction"] = np.where(out["involved"], out["next_ttb"] - out["ttb"], np.nan)
    return out


def per_player(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["nfl_id", "player_name", "position"]
    g = (df.groupby(keys)
         .agg(plays=("shadow", "size"), avg_shadow=("shadow", "mean"),
              involved=("involved", "sum"), involvement=("involved", "mean"))
         .reset_index())
    e = (df[df["involved"]].groupby(keys)
         .agg(effect=("reduction", "mean")).reset_index())
    return g.merge(e, on=keys, how="left")
