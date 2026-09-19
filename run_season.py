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
lb = leaderboard(season_scored, min_plays=30)

Path("outputs").mkdir(exist_ok=True)
season_scored.to_csv(f"outputs/shadow_plays_{season}_season.csv", index=False)
lb.to_csv(f"outputs/shadow_leaderboard_{season}_season.csv", index=False)

total = season_scored.groupby(["game_id","play_id"]).ngroups
print(f"\n{season} season: {len(files)} weeks, {total} plays, {len(season_scored)} defender-play rows")
print(f"\nTop 20 by total Shadow (min 30 coverage snaps):")
print(lb.head(20).round(2).to_string(index=False))
