"""Score a full season and write one combined leaderboard.

Usage:
    python run_season.py <folder_of_input_csvs> [season]
"""
import sys
from pathlib import Path
import pandas as pd
from coverage_shadow import load_week, score_plays, leaderboard

folder = Path(sys.argv[1]).expanduser()
season = sys.argv[2] if len(sys.argv) > 2 else "2023"

files = sorted(folder.glob(f"input_{season}_w*.csv"))
if not files:
    sys.exit(f"No input_{season}_w*.csv files found in {folder}")

all_scored = []
for f in files:
    df = load_week(str(f))
    scored = score_plays(df)
    week = f.stem.split("_w")[-1]
    scored["week"] = week
    all_scored.append(scored)
    print(f"  week {week}: {scored.groupby(['game_id','play_id']).ngroups} plays")

season_scored = pd.concat(all_scored, ignore_index=True)

# Join pass outcomes: supplementary_data.csv lives in the folder or its parent.
supp_path = next((p for p in [folder / "supplementary_data.csv",
                              folder.parent / "supplementary_data.csv"]
                  if p.exists()), None)
if supp_path:
    supp = pd.read_csv(supp_path, usecols=["game_id", "play_id", "pass_result"])
    season_scored = season_scored.merge(supp, on=["game_id", "play_id"], how="left")

lb = leaderboard(season_scored, min_plays=30)

Path("outputs").mkdir(exist_ok=True)
season_scored.to_csv(f"outputs/shadow_plays_{season}_season.csv", index=False)
lb.to_csv(f"outputs/shadow_leaderboard_{season}_season.csv", index=False)

total = season_scored.groupby(["game_id","play_id"]).ngroups
print(f"\n{season} season: {len(files)} weeks, {total} plays, {len(season_scored)} defender-play rows")
print(f"\nTop 20 by total Shadow (min 30 coverage snaps):")
print(lb.head(20).round(2).to_string(index=False))

if supp_path:
    closest = (season_scored[season_scored["closest"]]
               .dropna(subset=["pass_result"]).copy())
    closest["completed"] = (closest["pass_result"] == "C").astype(int)
    q1, q3 = closest["shadow"].quantile([0.25, 0.75])
    top = closest[closest["shadow"] >= q3]
    bot = closest[closest["shadow"] <= q1]
    corr = closest[["catch_window", "completed"]].corr().iloc[0, 1]
    print(f"\nValidation ({len(closest)} plays with pass_result):")
    print(f"  Top-quartile Shadow    (>= {q3:.2f}s, n={len(top)}): "
          f"{top['completed'].mean():.1%} complete")
    print(f"  Bottom-quartile Shadow (<= {q1:.2f}s, n={len(bot)}): "
          f"{bot['completed'].mean():.1%} complete")
    print(f"  corr(catch_window, completed) = {corr:+.3f}")
