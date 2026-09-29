"""Coverage Shadow v3: involvement and effect, tested against a pre-registered bar.

Usage:
    python run_v3.py <folder_of_input_csvs> [season]

Definitions: coverage_shadow/v3.py. Bar (README "v3", fixed before running):
effect passes only if
  1. its split-half reliability (odd vs even weeks, Spearman-Brown full
     season) is >= 0.40, AND
  2. in BOTH splits (weeks 1-9 -> 10-18, odd -> even), first-half effect
     predicts second-half PFR completion % allowed at least as well as
     first-half completion % allowed, within noise: not significantly worse,
     Meng-Rosenthal-Rubin two-sided p >= 0.05 (bootstrap interval reported).
Involvement is reported against the same bar. CBs only; >= 50 coverage
snaps and >= 15 involved plays per half; >= 15 PFR targets per half.

Writes outputs/v3_<season>.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coverage_shadow.stability import (
    bootstrap_r_diff, compare_dependent_r, correlate, nfl_to_pfr, pearson, pfr_per_player,
    spearman_brown,
)
from coverage_shadow.v3 import add_v3, per_player

SNAPS_HALF, INVOLVED_HALF, TARGETS_HALF = 50, 15, 15
RELIABILITY_BAR, P_BAR = 0.40, 0.05
CACHE = Path("data/raw/nflverse")

season = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
scored = add_v3(pd.read_csv(f"outputs/shadow_plays_{season}_season.csv"))
cb = scored[scored["position"] == "CB"].copy()
cb["week"] = cb["week"].astype(int)
rows = []


def half(weeks):
    p = per_player(cb[cb["week"].isin(weeks)]).set_index("nfl_id")
    return p[(p["plays"] >= SNAPS_HALF) & (p["involved"] >= INVOLVED_HALF)]


full = per_player(cb)
full = full[full["plays"] >= 100]
print(f"{season} CBs with 100+ snaps (n={len(full)}): involvement median "
      f"{full['involvement'].median():.3f}, effect median {full['effect'].median():.3f} s; "
      f"corr(involvement, effect) {pearson(full['involvement'], full['effect']):+.3f}; "
      f"corr with avg Shadow: involvement {pearson(full['involvement'], full['avg_shadow']):+.3f}, "
      f"effect {pearson(full['effect'], full['avg_shadow']):+.3f}")

# ---- bar part 1: split-half reliability ----
print(f"\n== Split-half reliability, odd vs even weeks (>= {SNAPS_HALF} snaps, "
      f">= {INVOLVED_HALF} involved plays per half) ==")
o, e = half(range(1, 19, 2)), half(range(2, 19, 2))
m = o.join(e, lsuffix="_o", rsuffix="_e", how="inner")
reliability = {}
for comp in ("involvement", "effect", "avg_shadow"):
    res = correlate(m[f"{comp}_o"], m[f"{comp}_e"])
    sb = spearman_brown(res["r"])
    reliability[comp] = sb
    rows.append({"test": "split-half", "component": comp, "n": res["n"], "r": res["r"],
                 "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "spearman_brown": sb})
    print(f"  {comp:12s} n={res['n']}  half r={res['r']:+.3f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]"
          f"  full-season reliability {sb:+.3f}  -> {'meets' if sb >= RELIABILITY_BAR else 'below'} 0.40")

# ---- bar part 2: prediction of second-half completion % allowed ----
to_pfr = nfl_to_pfr(CACHE)
pfr = pd.read_parquet(CACHE / f"advstats_week_def_{season}.parquet")
pfr = pfr[(pfr["game_type"] == "REG") & pfr["pfr_player_id"].notna()]
splits = {"weeks 1-9 -> 10-18": (range(1, 10), range(10, 19)),
          "odd -> even weeks": (range(1, 19, 2), range(2, 19, 2))}
prediction_ok = {"involvement": True, "effect": True}
print("\n== Predicting second-half PFR completion % allowed "
      "(components sign-flipped: more involvement / effect should mean fewer completions) ==")
for split, (w1, w2) in splits.items():
    s1 = half(w1)
    s1["pfr"] = [to_pfr.get(int(i)) for i in s1.index]
    p1 = pfr_per_player(pfr[pfr["week"].isin(w1)]).set_index("pfr_player_id")
    p2 = pfr_per_player(pfr[pfr["week"].isin(w2)]).set_index("pfr_player_id")
    d = (s1.dropna(subset=["pfr"]).set_index("pfr")
         .join(p1[["targets", "cmp_pct_allowed"]].add_suffix("_1"), how="inner")
         .join(p2[["targets", "cmp_pct_allowed"]].add_suffix("_2"), how="inner"))
    d = d[(d["targets_1"] >= TARGETS_HALF) & (d["targets_2"] >= TARGETS_HALF)]
    n, y = len(d), d["cmp_pct_allowed_2"]
    rb = pearson(d["cmp_pct_allowed_1"], y)
    base = correlate(d["cmp_pct_allowed_1"], y)
    print(f"  {split} (n={n}): 1st-half cmp % allowed r={rb:+.3f} "
          f"[{base['ci_lo']:+.2f}, {base['ci_hi']:+.2f}]")
    rows.append({"test": f"predict 2nd-half cmp% ({split})", "component": "cmp_pct_allowed",
                 **{k: base[k] for k in ("n", "r", "ci_lo", "ci_hi")}})
    for comp in ("involvement", "effect", "avg_shadow"):
        x = -d[comp]
        res = correlate(x, y)
        rx = pearson(x, d["cmp_pct_allowed_1"])
        z, p = compare_dependent_r(res["r"], rb, rx, n)
        lo, hi = bootstrap_r_diff(y, x, d["cmp_pct_allowed_1"])
        ok = p >= P_BAR or res["r"] >= rb
        if comp in prediction_ok:
            prediction_ok[comp] &= ok
        rows.append({"test": f"predict 2nd-half cmp% ({split})", "component": comp,
                     **{k: res[k] for k in ("n", "r", "ci_lo", "ci_hi")},
                     "diff_vs_cmp": res["r"] - rb, "diff_lo": lo, "diff_hi": hi, "z": z, "p": p})
        print(f"    -{comp:12s} r={res['r']:+.3f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]  "
              f"minus cmp%: {res['r'] - rb:+.3f} [{lo:+.2f}, {hi:+.2f}], MRR p={p:.2f}"
              f"  -> {'not worse' if ok else 'WORSE'}")

# ---- does involvement explain the target-rate finding? ----
print("\n== Target rate: first-half component vs second-half PFR targets per coverage snap ==")
for split, (w1, w2) in splits.items():
    s1 = per_player(cb[cb["week"].isin(w1)]).set_index("nfl_id")
    s1 = s1[s1["plays"] >= SNAPS_HALF]
    s2 = cb[cb["week"].isin(w2)].groupby("nfl_id").size().rename("plays2")
    d = s1.join(s2, how="inner")
    d = d[d["plays2"] >= SNAPS_HALF]
    d["pfr"] = [to_pfr.get(int(i)) for i in d.index]
    t2 = pfr[pfr["week"].isin(w2)].groupby("pfr_player_id")["def_targets"].sum()
    d = d.join(t2, on="pfr").dropna(subset=["def_targets", "effect"])
    d["tps"] = d["def_targets"] / d["plays2"]
    out = []
    for comp in ("avg_shadow", "involvement", "effect"):
        res = correlate(d[comp], d["tps"])
        rows.append({"test": f"target rate ({split})", "component": comp,
                     **{k: res[k] for k in ("n", "r", "ci_lo", "ci_hi")}})
        out.append(f"{comp} {res['r']:+.3f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]")
    # Shadow's link to target rate once involvement is held fixed.
    X = np.column_stack([np.ones(len(d)), d["involvement"]])
    ra = d["avg_shadow"] - X @ np.linalg.lstsq(X, d["avg_shadow"], rcond=None)[0]
    rt = d["tps"] - X @ np.linalg.lstsq(X, d["tps"], rcond=None)[0]
    partial = pearson(ra, rt)
    rows.append({"test": f"target rate ({split})", "component": "avg_shadow | involvement",
                 "n": len(d), "r": partial})
    print(f"  {split} (n={len(d)}): " + "; ".join(out)
          + f"; avg Shadow partial on involvement {partial:+.3f}")

# ---- verdict ----
print("\n== Verdict against the pre-registered bar ==")
for comp in ("effect", "involvement"):
    rel_ok = reliability[comp] >= RELIABILITY_BAR
    verdict = "PASS" if rel_ok and prediction_ok[comp] else "FAIL"
    print(f"  {comp:12s} reliability {reliability[comp]:.2f} ({'ok' if rel_ok else 'below 0.40'}); "
          f"prediction {'not worse in both splits' if prediction_ok[comp] else 'worse in at least one split'}"
          f"  -> {verdict}")

Path("outputs").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(f"outputs/v3_{season}.csv", index=False)
print(f"\nwrote outputs/v3_{season}.csv")
