"""Score one week of tracking data and write play-level scores + a leaderboard.

Usage:
    python run.py data/raw/input_2023_w01.csv
"""
import sys
from pathlib import Path
from coverage_shadow import load_week, score_plays, leaderboard

path = Path(sys.argv[1])
tag = path.stem.replace("input_", "")            # e.g. 2023_w01

df = load_week(path)
scored = score_plays(df)
lb = leaderboard(scored)

Path("outputs").mkdir(exist_ok=True)
scored.to_csv(f"outputs/shadow_plays_{tag}.csv", index=False)
lb.to_csv(f"outputs/shadow_leaderboard_{tag}.csv", index=False)

print(f"{tag}: {scored.groupby(['game_id','play_id']).ngroups} plays, {len(scored)} defender-play rows")
print(lb.head(15).round(2).to_string(index=False))
