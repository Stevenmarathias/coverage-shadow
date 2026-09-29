"""Two checks on avg Coverage Shadow for 2023 cornerbacks.

Usage:
    python run_role_checks.py <folder_of_input_csvs> [season]

1. Role confound. Regress play-level Shadow on alignment at the snap (slot,
   depth off the ball, cushion; coverage_shadow/role.py), then also on team
   scheme (man/zone, coverage type) and defensive team. Does split-half
   reliability survive on the residuals, and how much of CBs' season avg
   Shadow do these explain?
2. Predictive validity. Does first-half avg Shadow predict second-half PFR
   completion % allowed and yards per target allowed better than
   first-half completion % allowed does? Splits: weeks 1-9 -> 10-18, and
   odd -> even weeks. Shadow is sign-flipped (more Shadow should mean
   fewer completions) so every predictor's "right" direction is positive.

Writes outputs/role_confound_<season>.csv, outputs/predictive_validity_<season>.csv
and figures/predictive_validity_<season>.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coverage_shadow.depth import find_supplementary  # noqa: E402
from coverage_shadow.model import _quiet_matmul  # noqa: E402
from coverage_shadow.role import BLOCKS, attach_scheme, load_alignment, ols_residuals  # noqa: E402
from coverage_shadow.stability import (  # noqa: E402
    bootstrap_r_diff, compare_dependent_r, correlate, nfl_to_pfr, pearson, pfr_cb_weeks,
    pfr_per_player, spearman_brown,
)

SNAPS_HALF, TARGETS_HALF, SNAPS_SEASON = 50, 15, 100
CACHE = Path("data/raw/nflverse")

folder = Path(sys.argv[1]).expanduser()
season = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
supp = find_supplementary(folder)

scored = pd.read_csv(f"outputs/shadow_plays_{season}_season.csv",
                     usecols=["game_id", "play_id", "nfl_id", "player_name", "position",
                              "shadow", "week"])
cb = scored[scored["position"] == "CB"].merge(
    load_alignment(folder, season), on=["game_id", "play_id", "nfl_id"], how="inner")
cb = attach_scheme(cb, supp)
cb["week"] = cb["week"].astype(int)
print(f"{season} CB coverage snaps with alignment: {len(cb):,} "
      f"(slot {cb['slot'].mean():.0%}, median depth {cb['depth'].median():.1f} yd, "
      f"median cushion {cb['cushion'].median():.1f} yd)")

# ---------------------------------------------------------------- 1 ----
rows = []
print("\n== 1. Role confound ==")
full = cb.groupby("nfl_id").agg(avg=("shadow", "mean"), plays=("shadow", "size"),
                               slot=("slot", "mean"), depth=("depth", "mean"),
                               cushion=("cushion", "mean"),
                               man=("team_coverage_man_zone", lambda s: (s == "MAN_COVERAGE").mean()),
                               team=("defensive_team", lambda s: s.mode().iat[0]))
full = full[full["plays"] >= SNAPS_SEASON]
covmix = (pd.get_dummies(cb["team_coverage_type"].fillna("NA"), dtype=float)
          .groupby(cb["nfl_id"]).mean().loc[full.index])


def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    with _quiet_matmul():
        res = y - X @ b
    k = X.shape[1] - 1
    raw = 1 - res.var() / y.var()
    return raw, 1 - (1 - raw) * (len(y) - 1) / (len(y) - k - 1)


y = full["avg"].to_numpy()
player_level = {
    "slot share": full[["slot"]],
    "alignment (slot, depth, cushion)": full[["slot", "depth", "cushion"]],
    "alignment + scheme (man, coverage mix)": pd.concat(
        [full[["slot", "depth", "cushion", "man"]], covmix.iloc[:, 1:]], axis=1),
    "alignment + scheme + team": pd.concat(
        [full[["slot", "depth", "cushion", "man"]], covmix.iloc[:, 1:],
         pd.get_dummies(full["team"], drop_first=True, dtype=float)], axis=1),
}
print(f"Share of CB season avg Shadow variance explained (>= {SNAPS_SEASON} snaps, n={len(full)}):")
for name, X in player_level.items():
    raw, adj = r2(X.to_numpy(float), y)
    rows.append({"check": "variance explained", "controls": name, "n": len(full),
                 "r2": raw, "adj_r2": adj})
    print(f"  {name:42s} R^2 {raw:.3f}  adjusted {adj:.3f}")
for c in ("slot", "depth", "cushion", "man"):
    print(f"  corr(avg Shadow, {c:7s}) {pearson(full['avg'], full[c]):+.3f}")


def half_avgs(df, block):
    if block == "none":
        df = df.assign(res=df["shadow"])
        play_r2 = 0.0
    else:
        res, play_r2 = ols_residuals(df, block)
        df = df.assign(res=res)
    g = df.groupby("nfl_id").agg(v=("res", "mean"), plays=("res", "size"))
    return g[g["plays"] >= SNAPS_HALF], play_r2


print(f"\nSplit-half (odd vs even weeks), CBs >= {SNAPS_HALF} snaps per half, "
      "controls fit within each half:")
for block in ["none", *BLOCKS]:
    o, r2o = half_avgs(cb[cb["week"] % 2 == 1], block)
    e, r2e = half_avgs(cb[cb["week"] % 2 == 0], block)
    m = o.join(e, lsuffix="_o", rsuffix="_e", how="inner")
    res = correlate(m["v_o"], m["v_e"])
    sb = spearman_brown(res["r"])
    rows.append({"check": "split-half", "controls": block, "n": res["n"], "r": res["r"],
                 "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "spearman_brown": sb,
                 "play_r2": (r2o + r2e) / 2})
    print(f"  {block:24s} n={res['n']}  r={res['r']:+.3f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]"
          f"  full-season {sb:+.3f}  (play-level R^2 of controls {(r2o + r2e) / 2:.3f})")

Path("outputs").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(f"outputs/role_confound_{season}.csv", index=False)

# ---------------------------------------------------------------- 2 ----
print("\n== 2. Predictive validity (PFR outcomes) ==")
to_pfr = nfl_to_pfr(CACHE)
pfr = pfr_cb_weeks(season, CACHE)          # CBs per nflverse roster
pfr_all = pd.read_parquet(CACHE / f"advstats_week_def_{season}.parquet")
pfr_all = pfr_all[(pfr_all["game_type"] == "REG") & pfr_all["pfr_player_id"].notna()]
bdb_cbs = cb["nfl_id"].unique()
mapped = {n: to_pfr.get(int(n)) for n in bdb_cbs}
print(f"BDB CBs: {len(bdb_cbs)}, mapped to a PFR id: {sum(v is not None for v in mapped.values())}")


def pfr_half(weeks):
    g = pfr_per_player(pfr_all[pfr_all["week"].isin(weeks)])
    g["ypt"] = np.where(g["targets"] > 0, g["yds"] / g["targets"].clip(lower=1), np.nan)
    return g.set_index("pfr_player_id")


def shadow_half(weeks, block=None):
    d = cb[cb["week"].isin(weeks)]
    if block:
        d = d.assign(shadow=ols_residuals(d, block)[0])
    g = d.groupby("nfl_id").agg(shadow=("shadow", "mean"), plays=("shadow", "size"))
    g = g[g["plays"] >= SNAPS_HALF]
    g["pfr_player_id"] = [mapped.get(n) for n in g.index]
    return g.dropna(subset=["pfr_player_id"]).set_index("pfr_player_id")


splits = {"weeks 1-9 -> 10-18": (list(range(1, 10)), list(range(10, 19))),
          "odd -> even weeks": (list(range(1, 19, 2)), list(range(2, 19, 2)))}
prows, fig_rows = [], []
for split, (w1, w2) in splits.items():
    s1 = shadow_half(w1)
    s1_adj = shadow_half(w1, "alignment+scheme")["shadow"].rename("shadow_adj")
    p1, p2 = pfr_half(w1), pfr_half(w2)
    d = (s1.join(s1_adj, how="inner")
         .join(p1[["targets", "cmp_pct_allowed", "ypt"]].add_suffix("_1"), how="inner")
         .join(p2[["targets", "cmp_pct_allowed", "ypt"]].add_suffix("_2"), how="inner"))
    d = d[(d["targets_1"] >= TARGETS_HALF) & (d["targets_2"] >= TARGETS_HALF)]
    n = len(d)
    print(f"\n{split}: CBs with >= {SNAPS_HALF} snaps and >= {TARGETS_HALF} PFR targets "
          f"in the first half and >= {TARGETS_HALF} targets in the second: n={n}")
    preds = {"-avg Shadow (1st half)": -d["shadow"],
             "-role-adjusted Shadow (1st half)": -d["shadow_adj"],
             "cmp % allowed (1st half)": d["cmp_pct_allowed_1"],
             "yds/target allowed (1st half)": d["ypt_1"]}
    for outcome, label in (("cmp_pct_allowed_2", "2nd-half cmp % allowed"),
                           ("ypt_2", "2nd-half yds/target allowed")):
        yv = d[outcome]
        base_name = ("cmp % allowed (1st half)" if outcome == "cmp_pct_allowed_2"
                     else "yds/target allowed (1st half)")
        print(f"  predicting {label}:")
        for pname, x in preds.items():
            res = correlate(x, yv)
            prows.append({"split": split, "outcome": label, "predictor": pname, **res})
            print(f"    {pname:34s} r={res['r']:+.3f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]")
            fig_rows.append((split, label, pname, res["r"], res["ci_lo"], res["ci_hi"]))
        # Shadow vs the stat itself (and vs cmp% for the yds/target outcome).
        for rival in {base_name, "cmp % allowed (1st half)"}:
            rs = pearson(preds["-avg Shadow (1st half)"], yv)
            rb = pearson(preds[rival], yv)
            rx = pearson(preds["-avg Shadow (1st half)"], preds[rival])
            z, p = compare_dependent_r(rs, rb, rx, n)
            lo, hi = bootstrap_r_diff(yv, preds["-avg Shadow (1st half)"], preds[rival])
            print(f"    Shadow minus {rival}: {rs - rb:+.3f} (bootstrap 95% [{lo:+.2f}, {hi:+.2f}]), "
                  f"Meng-Rosenthal-Rubin z={z:+.2f}, p={p:.2f}; corr between predictors {rx:+.2f}")
            prows.append({"split": split, "outcome": label,
                          "predictor": f"diff: Shadow - {rival}", "n": n, "r": rs - rb,
                          "ci_lo": lo, "ci_hi": hi, "rho": np.nan, "z": z, "p": p})
        # Does Shadow add anything on top of the stat?
        X1 = np.column_stack([np.ones(n), preds[base_name]])
        X2 = np.column_stack([X1, preds["-avg Shadow (1st half)"]])
        with _quiet_matmul():
            r2_1 = 1 - np.var(yv - X1 @ np.linalg.lstsq(X1, yv, rcond=None)[0]) / np.var(yv)
            r2_2 = 1 - np.var(yv - X2 @ np.linalg.lstsq(X2, yv, rcond=None)[0]) / np.var(yv)
        print(f"    R^2: {base_name} alone {r2_1:.3f}; + Shadow {r2_2:.3f}")
        prows.append({"split": split, "outcome": label, "predictor": f"R2 {base_name} (+Shadow)",
                      "n": n, "r": r2_1, "ci_lo": r2_2})

# Why doesn't Shadow forecast completion %? Completion % only counts targets,
# so check whether Shadow changes who gets thrown at: second-half PFR targets
# per coverage snap, and share of snaps as the defender closest to the throw.
closest = pd.read_csv(f"outputs/shadow_plays_{season}_season.csv",
                      usecols=["game_id", "play_id", "nfl_id", "closest"])
cbc = cb.merge(closest, on=["game_id", "play_id", "nfl_id"], how="left")
print("\nInvolvement: does first-half Shadow change how often a CB is thrown at later?")
for split, (w1, w2) in splits.items():
    a = cbc[cbc["week"].isin(w1)].groupby("nfl_id").agg(sh=("shadow", "mean"), n1=("shadow", "size"))
    b = cbc[cbc["week"].isin(w2)].groupby("nfl_id").agg(n2=("shadow", "size"),
                                                       closest2=("closest", "mean"))
    d = a.join(b, how="inner")
    d = d[(d["n1"] >= SNAPS_HALF) & (d["n2"] >= SNAPS_HALF)]
    d["pfr"] = [mapped.get(i) for i in d.index]
    t2 = pfr_all[pfr_all["week"].isin(w2)].groupby("pfr_player_id")["def_targets"].sum()
    d = d.join(t2, on="pfr").dropna()
    d["tps"] = d["def_targets"] / d["n2"]
    for col, label in (("tps", "2nd-half PFR targets per coverage snap"),
                       ("closest2", "2nd-half share of snaps as closest defender")):
        res = correlate(d["sh"], d[col])
        prows.append({"split": split, "outcome": label, "predictor": "avg Shadow (1st half)", **res})
        print(f"  {split:20s} corr(1st-half Shadow, {label}) n={res['n']} r={res['r']:+.3f} "
              f"[{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]")

pd.DataFrame(prows).to_csv(f"outputs/predictive_validity_{season}.csv", index=False)

# Figure: correlation with the second-half outcome, per predictor, both splits.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
order = ["-avg Shadow (1st half)", "-role-adjusted Shadow (1st half)",
         "cmp % allowed (1st half)", "yds/target allowed (1st half)"]
names = ["Avg Shadow", "Role-adjusted Shadow", "Cmp % allowed", "Yds/target allowed"]
colors = {"weeks 1-9 -> 10-18": "#1f5f99", "odd -> even weeks": "#e08a2e"}
for ax, label in zip(axes, ("2nd-half cmp % allowed", "2nd-half yds/target allowed")):
    for k, split in enumerate(splits):
        for i, pname in enumerate(order):
            r = next(f for f in fig_rows if f[0] == split and f[1] == label and f[2] == pname)
            yv = i + (k - 0.5) * 0.3
            ax.errorbar(r[3], yv, xerr=[[r[3] - r[4]], [r[5] - r[3]]], fmt="o",
                        color=colors[split], capsize=3, label=split if i == 0 else None)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(order)), names)
    ax.invert_yaxis()
    ax.set_title(f"Predicting {label}", fontsize=11)
    ax.set_xlabel("correlation with the second-half outcome (95% CI)\n"
                  "Shadow sign-flipped: right of 0 = predicts in the expected direction")
axes[0].legend(loc="upper right", fontsize=9)
fig.suptitle(f"{season} cornerbacks: does first-half Shadow forecast second-half coverage results?",
             fontsize=12)
fig.tight_layout()
out = f"figures/predictive_validity_{season}.png"
fig.savefig(out, dpi=150)
print(f"\nwrote outputs/role_confound_{season}.csv, outputs/predictive_validity_{season}.csv, {out}")
