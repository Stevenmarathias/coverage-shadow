"""Render the README figures.

Usage:
    python make_figures.py <folder_of_input_csvs> [season]

Reads outputs/shadow_plays_<season>_season.csv and
outputs/shadow_leaderboard_<season>_season_v2.csv (produced by
run_season.py) plus the raw week CSVs from the input folder.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from coverage_shadow import load_week
from coverage_shadow.plot import plot_play, plot_soe

folder = Path(sys.argv[1]).expanduser()
season = sys.argv[2] if len(sys.argv) > 2 else "2023"
outputs = Path("outputs")
figures = Path("figures")
figures.mkdir(exist_ok=True)

plays = pd.read_csv(outputs / f"shadow_plays_{season}_season.csv")
lb_v2 = pd.read_csv(outputs / f"shadow_leaderboard_{season}_season_v2.csv")

# Gilmore's best single-play shadow_won: closest defender, incompletion, max shadow.
gilmore = plays[(plays["player_name"] == "Stephon Gilmore")
                & (plays["closest"])
                & (plays["completed"] == 0)]
if gilmore.empty:
    sys.exit("No Gilmore closest+incomplete rows found in season plays")
best = gilmore.loc[gilmore["shadow"].idxmax()]
week = f"{int(best['week']):02d}"
raw_path = folder / f"input_{season}_w{week}.csv"
print(f"Example play: game {int(best['game_id'])} play {int(best['play_id'])} "
      f"(week {week}), shadow={best['shadow']:.2f}s")

tracking = load_week(str(raw_path))

fig, ax = plt.subplots(figsize=(13, 6.5))
plot_play(tracking, int(best["game_id"]), int(best["play_id"]), ax=ax,
          title=(f"Stephon Gilmore forces incomplete — "
                 f"Shadow {best['shadow']:.2f}s "
                 f"(catch window {best['catch_window']:.2f}s, "
                 f"expected completion {best['expected_completion']:.2f})"))
fig.tight_layout()
fig.savefig(figures / f"example_play_{season}.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6.5))
plot_soe(lb_v2, min_plays=100, ax=ax)
fig.tight_layout()
fig.savefig(figures / f"soe_scatter_{season}.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {figures}/example_play_{season}.png and "
      f"{figures}/soe_scatter_{season}.png")
