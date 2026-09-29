"""Does throw depth belong in the expected-completion model?

Usage:
    python run_expected_model.py <folder_of_input_csvs> [season]

Reads outputs/shadow_plays_<season>_season.csv (run_season.py) and the
Big Data Bowl supplementary file for pass_length (air yards). Compares
held-out log loss of the completion model with and without air yards
(six folds of three weeks), then scores SOE both ways.

Adopt depth only if it CLEARLY helps: lower held-out log loss in every
fold and a mean improvement of at least 0.005 nats (~1% of the loss).

Writes:
    outputs/completion_model_cv_<season>.csv
    outputs/shadow_leaderboard_<season>_season_v2_depth.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coverage_shadow import fit_logit, leaderboard_v2
from coverage_shadow.depth import (
    SPECS, add_soe, attach_pass_length, contested, cross_validate, design,
    find_supplementary,
)

CLEAR_GAIN = 0.005

folder = Path(sys.argv[1]).expanduser()
season = sys.argv[2] if len(sys.argv) > 2 else "2023"
scored = pd.read_csv(f"outputs/shadow_plays_{season}_season.csv")
scored = attach_pass_length(scored, find_supplementary(folder))
plays = contested(scored)
print(f"{season}: {len(plays):,} completions/incompletions, "
      f"{plays['pass_length'].isna().sum()} missing pass_length, "
      f"completion rate {plays['completed'].mean():.3f}")

cv = cross_validate(plays)
print("\nHeld-out log loss by fold (lower is better):")
print(cv.round(4).to_string(index=False))
means = cv[["baseline", *SPECS]].mean()
print("\nMean:", "  ".join(f"{k} {v:.4f}" for k, v in means.items()))

gain = cv["window"] - cv["window+depth"]
gain2 = cv["window+depth"] - cv["window+depth2"]
print(f"\nwindow -> window+depth: mean gain {gain.mean():.4f} nats, "
      f"better in {int((gain > 0).sum())}/{len(cv)} folds")
print(f"window+depth -> +depth^2: mean gain {gain2.mean():.4f} nats, "
      f"better in {int((gain2 > 0).sum())}/{len(cv)} folds")
adopt = bool((gain > 0).all() and gain.mean() >= CLEAR_GAIN)
print(f"Adopt depth (every fold better and mean gain >= {CLEAR_GAIN}): {adopt}")

# Full-season fits and SOE both ways.
y = plays["completed"].to_numpy()
betas = {s: fit_logit(design(plays, s), y) for s in ("window", "window+depth")}
for s, b in betas.items():
    terms = " + ".join(f"{v:+.3f}·{n}" for v, n in zip(b, ["1", "cw", "air/10"]))
    print(f"full-season {s:13s} logit = {terms}")

scored = add_soe(scored, "window+depth", betas["window+depth"], "soe_depth")
lb = leaderboard_v2(scored, min_plays=100)          # v2 as published (window only)
lb_d = leaderboard_v2(scored.drop(columns=["shadow_over_expected"])
                      .rename(columns={"soe_depth": "shadow_over_expected"}),
                      min_plays=100)
cols = ["player_name", "position", "plays", "contests", "shadow_over_expected"]
print("\nSOE top 10, window only (v2 as published):")
print(lb.head(10)[cols].round(2).to_string(index=False))
print("\nSOE top 10, window + depth:")
print(lb_d.head(10)[cols].round(2).to_string(index=False))

both = lb[["nfl_id", "player_name", "shadow_over_expected"]].merge(
    lb_d[["nfl_id", "shadow_over_expected"]], on="nfl_id", suffixes=("", "_depth"))
r = both["shadow_over_expected"].corr(both["shadow_over_expected_depth"])
rho = both["shadow_over_expected"].rank().corr(both["shadow_over_expected_depth"].rank())
top10 = len(set(lb.head(10)["nfl_id"]) & set(lb_d.head(10)["nfl_id"]))
print(f"\nqualified defenders {len(both)}: corr(SOE, SOE_depth) r={r:.3f}, "
      f"rho={rho:.3f}; top-10 overlap {top10}/10")

Path("outputs").mkdir(exist_ok=True)
cv.to_csv(f"outputs/completion_model_cv_{season}.csv", index=False)
lb_d.to_csv(f"outputs/shadow_leaderboard_{season}_season_v2_depth.csv", index=False)
