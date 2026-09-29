"""Stability of Coverage Shadow and SOE vs the standard coverage stats.

Usage:
    python run_stability.py <folder_of_input_csvs> [season]

Needs outputs/shadow_plays_<season>_season.csv (run_season.py) and the Big
Data Bowl supplementary file (pass_length, for the depth-aware SOE).
Downloads nflverse PFR advanced defense + roster files (2018-2025) into
data/raw/nflverse/ for the benchmark.

Only the 2023 season has tracking data in the Big Data Bowl 2026 release
(2024 appears only as play outcomes for weeks 14-18), so our metrics are
tested split-half within 2023: odd vs even weeks, Spearman-Brown to a full
season. PFR's stats get the same split-half test in 2023, plus their
2023 -> 2024 year-over-year correlation (and 2018-2025 pooled) as context.

Writes outputs/stability_<season>.csv and figures/stability_split_half_<season>.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coverage_shadow import fit_logit  # noqa: E402
from coverage_shadow.depth import (  # noqa: E402
    add_soe, attach_pass_length, contested, design, find_supplementary,
)
from coverage_shadow.stability import (  # noqa: E402
    halves, pfr_cb_weeks, pfr_split_half, pfr_year_over_year, split_half,
)

SNAP_GATES = [25, 50, 75, 100, 150]        # coverage snaps per half
TARGET_GATES = [10, 15, 20, 25, 30, 40]    # contests / targets per half
DEFAULT_SNAPS, DEFAULT_TARGETS = 50, 15    # ~ the season boards' minimums, halved
CACHE = Path("data/raw/nflverse")

folder = Path(sys.argv[1]).expanduser()
season = int(sys.argv[2]) if len(sys.argv) > 2 else 2023

scored = pd.read_csv(f"outputs/shadow_plays_{season}_season.csv")
scored = attach_pass_length(scored, find_supplementary(folder))
plays = contested(scored)
beta = fit_logit(design(plays, "window+depth"), plays["completed"].to_numpy())
scored = add_soe(scored, "window+depth", beta, "soe_depth")

rows = []


def record(metric, version, group, split, gate_kind, gate, res):
    rows.append({"metric": metric, "version": version, "group": group, "split": split,
                 "gate": f">={gate} {gate_kind}", **{k: res[k] for k in
                 ("n", "r", "ci_lo", "ci_hi", "rho")},
                 "spearman_brown": res.get("spearman_brown", np.nan)})


def fmt(res):
    sb = res.get("spearman_brown", np.nan)
    return (f"n={res['n']:3d}  r={res['r']:+.3f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]"
            f"  rho={res['rho']:+.3f}" + (f"  full-season (S-B) {sb:+.3f}" if np.isfinite(sb) else ""))


print(f"== Split-half (odd vs even weeks), {season} tracking ==")
figdata = {}
for soe_col, version in (("shadow_over_expected", "v2 window"), ("soe_depth", "v2 window+depth")):
    odd, even = halves(scored, soe_col)
    for cb in (True, False):
        grp = "CB" if cb else "all defenders"
        if version == "v2 window":
            for gate in SNAP_GATES:
                res = split_half(odd, even, "avg_shadow", "plays", gate, cb)
                record("avg_shadow", "v1", grp, "split-half", "snaps/half", gate, res)
                if gate == DEFAULT_SNAPS:
                    print(f"avg Shadow per snap   {grp:13s} >={gate} snaps/half: {fmt(res)}")
                    if cb:
                        figdata["shadow"] = res
        for gate in TARGET_GATES:
            res = split_half(odd, even, "soe_per_contest", "contests", gate, cb)
            record("soe_per_contest", version, grp, "split-half", "contests/half", gate, res)
            if gate == DEFAULT_TARGETS:
                print(f"SOE per contest ({version:15s}) {grp:13s} >={gate} contests/half: {fmt(res)}")
                if cb and version == "v2 window+depth":
                    figdata["soe"] = res

print(f"\n== PFR benchmark (CBs by nflverse roster), same split in {season} ==")
pfr = {s: pfr_cb_weeks(s, CACHE) for s in range(2018, 2026)}
for metric in ("cmp_pct_allowed", "passer_rating_allowed"):
    for gate in TARGET_GATES:
        res = pfr_split_half(pfr[season], metric, gate)
        record(metric, "PFR", "CB", "split-half", "targets/half", gate, res)
        if gate == DEFAULT_TARGETS:
            print(f"{metric:22s} >={gate} targets/half: {fmt(res)}")
            if metric == "cmp_pct_allowed":
                figdata["pfr"] = res

print(f"\n== PFR year over year, {season} -> {season + 1} (CBs, >=30 targets each season) ==")
for metric in ("cmp_pct_allowed", "passer_rating_allowed"):
    yoy = pfr_year_over_year(pfr[season], pfr[season + 1], metric, 30)
    for part, res in yoy.items():
        record(metric, "PFR", f"CB {part}", f"{season}->{season + 1}", "targets/season", 30, res)
        print(f"{metric:22s} {part:12s}: {fmt(res)}")
    for gate in (20, 40, 60):
        res = pfr_year_over_year(pfr[season], pfr[season + 1], metric, gate)["all"]
        record(metric, "PFR", "CB all", f"{season}->{season + 1}", "targets/season", gate, res)

print("\n== PFR year over year, pooled consecutive seasons 2018 -> 2025 (context) ==")
for metric in ("cmp_pct_allowed", "passer_rating_allowed"):
    rs, ns = [], []
    for s in range(2018, 2025):
        res = pfr_year_over_year(pfr[s], pfr[s + 1], metric, 30)["all"]
        rs.append(res["r"]); ns.append(res["n"])
    z = np.average(np.arctanh(rs), weights=np.array(ns) - 3)
    pooled = {"n": int(sum(ns)), "r": float(np.tanh(z)), "ci_lo": np.nan, "ci_hi": np.nan,
              "rho": np.nan}
    record(metric, "PFR", "CB all", "2018-2025 pooled", "targets/season", 30, pooled)
    print(f"{metric:22s} pooled r={np.tanh(z):+.3f} over {sum(ns)} CB pairs "
          f"(per pair of seasons: {', '.join(f'{r:+.2f}' for r in rs)})")

print("\n== Sample size: split-half r (Spearman-Brown full season), CBs ==")
t = pd.DataFrame(rows)
sh = t[(t["split"] == "split-half") & (t["group"] == "CB")]
for (metric, version), g in sh.groupby(["metric", "version"], sort=False):
    cells = "  ".join(f"{r.gate.split()[0]}: {r.r:+.2f} ({r.spearman_brown:+.2f}, n={r.n})"
                      for r in g.itertuples())
    print(f"{metric:22s} {version:16s} {cells}")

# Why SOE doesn't hold: compare the spread of SOE per contest across CBs
# with the spread pure completion luck would produce. Each contest's SOE is
# p - completed, with variance p(1 - p) under the model, so a CB's
# per-contest average has noise variance sum(p(1-p)) / contests^2.
print("\n== Diagnostics ==")
c = scored[scored["soe_depth"].notna() & (scored["position"] == "CB")].copy()
c["p"] = c["soe_depth"] + c["pass_result"].map({"C": 1.0, "I": 0.0})
g = (c.groupby("nfl_id").agg(n=("soe_depth", "size"), soe=("soe_depth", "sum"),
                             v=("p", lambda p: (p * (1 - p)).sum())))
g = g[g["n"] >= 30]
obs = (g["soe"] / g["n"]).var(ddof=1)
noise = (g["v"] / g["n"] ** 2).mean()
print(f"SOE per contest, CBs with >=30 contests (n={len(g)}): observed sd {obs ** .5:.3f}, "
      f"sd from completion luck alone {noise ** .5:.3f} -> skill share of variance "
      f"{max(0.0, 1 - noise / obs):.2f}")

# Is avg Shadow's stability just scheme? Man vs zone from the supplementary file.
supp = pd.read_csv(find_supplementary(folder), usecols=["game_id", "play_id", "team_coverage_man_zone"])
x = scored.merge(supp, on=["game_id", "play_id"], how="left")
x["man"] = (x["team_coverage_man_zone"] == "MAN_COVERAGE").astype(float)
x = x[x["position"] == "CB"]
wk = x["week"].astype(int)


def cb_half(mask):
    h = x[mask].groupby("nfl_id").agg(avg=("shadow", "mean"), man=("man", "mean"),
                                      plays=("shadow", "size"))
    h = h[h["plays"] >= DEFAULT_SNAPS]
    b = np.polyfit(h["man"], h["avg"], 1)
    h["resid"] = h["avg"] - np.polyval(b, h["man"])
    return h


full = x.groupby("nfl_id").agg(avg=("shadow", "mean"), man=("man", "mean"), plays=("shadow", "size"))
full = full[full["plays"] >= 100]
m = cb_half(wk % 2 == 1).join(cb_half(wk % 2 == 0), lsuffix="_o", rsuffix="_e", how="inner")
print(f"CB avg Shadow vs share of snaps in man coverage (>=100 snaps, n={len(full)}): "
      f"r={np.corrcoef(full['avg'], full['man'])[0, 1]:+.3f}")
print(f"CB split-half after removing man share (n={len(m)}): "
      f"r={np.corrcoef(m['resid_o'], m['resid_e'])[0, 1]:+.3f} "
      f"(raw {np.corrcoef(m['avg_o'], m['avg_e'])[0, 1]:+.3f})")

Path("outputs").mkdir(exist_ok=True)
t.to_csv(f"outputs/stability_{season}.csv", index=False)

# Figure: odd vs even weeks for CBs, our two metrics and PFR's completion % allowed.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
panels = [("shadow", "avg_shadow", f"Avg Coverage Shadow (s/snap)\n>= {DEFAULT_SNAPS} snaps per half"),
          ("soe", "soe_per_contest", f"SOE per contest (window + depth)\n>= {DEFAULT_TARGETS} contests per half"),
          ("pfr", "cmp_pct_allowed", f"PFR completion % allowed\n>= {DEFAULT_TARGETS} targets per half")]
for ax, (key, col, title) in zip(axes, panels):
    res = figdata[key]
    m = res["pairs"]
    x, y = m[f"{col}_o"], m[f"{col}_e"]
    ax.scatter(x, y, s=18, alpha=0.7, color="#1f5f99", edgecolor="white", linewidth=0.4)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"odd weeks ({season})")
    ax.set_ylabel(f"even weeks ({season})")
    ax.text(0.03, 0.97, f"r = {res['r']:+.2f}  (n = {res['n']})\n"
            f"full-season reliability ~ {res['spearman_brown']:+.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
fig.suptitle(f"Split-half stability, {season} cornerbacks: does a CB's first-half number "
             f"predict his second half?", fontsize=12)
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
out = f"figures/stability_split_half_{season}.png"
fig.savefig(out, dpi=150)
print(f"\nwrote outputs/stability_{season}.csv and {out}")
