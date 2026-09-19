"""Visuals for Coverage Shadow. Two entry points:

    plot_play(tracking_df, game_id, play_id, ax=None)
    plot_soe(lb_v2, min_plays=100, ax=None)
"""
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .model import throw_frame, time_to_ball

FIELD_LEN = 120.0
FIELD_WID = 53.3

_FIELD_GREEN = "#3c7d3c"
_ENDZONE_GREEN = "#2b5f2b"
_LINE = "white"
_DEF = "#dc143c"
_TARGET = "#ffd166"
_OFF = "#89c2ff"
_PASSER = "white"
_BALL = "black"


def _draw_field(ax):
    ax.add_patch(Rectangle((0, 0), FIELD_LEN, FIELD_WID, fc=_FIELD_GREEN, ec="black"))
    ax.add_patch(Rectangle((0, 0), 10, FIELD_WID, fc=_ENDZONE_GREEN, ec=_LINE))
    ax.add_patch(Rectangle((110, 0), 10, FIELD_WID, fc=_ENDZONE_GREEN, ec=_LINE))
    for x in range(10, 111, 10):
        ax.axvline(x, color=_LINE, lw=0.8, alpha=0.55, zorder=1)
    for x in range(20, 101, 10):
        label = str(x - 10 if x <= 50 else 110 - x)
        ax.text(x, 5, label, color=_LINE, ha="center", va="center",
                fontsize=8, alpha=0.7, zorder=1)
        ax.text(x, FIELD_WID - 5, label, color=_LINE, ha="center", va="center",
                fontsize=8, alpha=0.7, zorder=1)
    ax.set_xlim(-2, FIELD_LEN + 2)
    ax.set_ylim(-2, FIELD_WID + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_play(tracking_df: pd.DataFrame, game_id, play_id, ax=None,
              title: Optional[str] = None):
    """Draw one play at the throw frame with defender times-to-ball labeled."""
    play = tracking_df[(tracking_df["game_id"] == game_id) &
                       (tracking_df["play_id"] == play_id)]
    if play.empty:
        raise ValueError(f"No tracking rows for game_id={game_id} play_id={play_id}")

    snap = throw_frame(play).copy()
    snap["ttb"] = time_to_ball(snap)

    # Orient so offense always moves left-to-right.
    if snap["play_direction"].iloc[0] == "left":
        for col in ("x", "ball_land_x"):
            snap[col] = FIELD_LEN - snap[col]

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6.5))
    _draw_field(ax)

    land_x = snap["ball_land_x"].iloc[0]
    land_y = snap["ball_land_y"].iloc[0]
    ax.plot(land_x, land_y, marker="X", color=_BALL, markersize=16,
            markeredgecolor=_LINE, markeredgewidth=1.5, zorder=5)
    ax.annotate("ball", (land_x, land_y), xytext=(6, 6),
                textcoords="offset points", color=_LINE, fontsize=8, zorder=6)

    defs = snap[snap["player_side"] == "Defense"].copy()
    closest_id = defs.loc[defs["ttb"].idxmin(), "nfl_id"] if not defs.empty else None
    for _, d in defs.iterrows():
        is_closest = d["nfl_id"] == closest_id
        if is_closest:
            ax.plot(d["x"], d["y"], marker="o", color=_DEF, markersize=17,
                    markeredgecolor=_LINE, markeredgewidth=2.5, zorder=6)
            ax.annotate(f"{d['player_name']}  {d['ttb']:.2f}s",
                        (d["x"], d["y"]), xytext=(18, 18),
                        textcoords="offset points", color="black", fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=_LINE,
                                  ec=_DEF, lw=1.5, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=_DEF, lw=1.2),
                        zorder=7)
        else:
            ax.plot(d["x"], d["y"], marker="o", color=_DEF, markersize=10,
                    markeredgecolor=_DEF, zorder=4)
            ax.annotate(f"{d['ttb']:.2f}s", (d["x"], d["y"]), xytext=(7, -2),
                        textcoords="offset points", color=_LINE, fontsize=7,
                        zorder=5)

    tgt = snap[snap["player_role"] == "Targeted Receiver"]
    for _, r in tgt.iterrows():
        ax.plot(r["x"], r["y"], marker="o", color=_TARGET, markersize=13,
                markeredgecolor="black", markeredgewidth=1, zorder=4)
        ax.annotate(f"target ({r['ttb']:.2f}s)", (r["x"], r["y"]),
                    xytext=(7, 6), textcoords="offset points",
                    color="black", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=_TARGET, ec="none",
                              alpha=0.85), zorder=5)

    others = snap[snap["player_role"] == "Other Route Runner"]
    ax.plot(others["x"], others["y"], marker="o", color=_OFF, markersize=8,
            linestyle="", zorder=3)

    passer = snap[snap["player_role"] == "Passer"]
    ax.plot(passer["x"], passer["y"], marker="s", color=_PASSER, markersize=10,
            markeredgecolor="black", markeredgewidth=1, zorder=4)

    ax.set_title(title or f"game {game_id} play {play_id} — throw frame",
                 fontsize=11)
    # Compact legend.
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=_PASSER,
                   markeredgecolor="black", markersize=9, label="passer"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_TARGET,
                   markeredgecolor="black", markersize=9, label="target"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_OFF,
                   markersize=8, label="other receiver"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_DEF,
                   markersize=9, label="defender"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_DEF,
                   markeredgecolor=_LINE, markeredgewidth=2, markersize=12,
                   label="closest defender"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=_BALL,
                   markeredgecolor=_LINE, markersize=11, label="ball landing"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, frameon=False)
    return ax


def plot_soe(lb_v2: pd.DataFrame, min_plays: int = 100, ax=None,
             n_labels: int = 5, title: Optional[str] = None):
    """Scatter contests vs Shadow Over Expected; label top and bottom `n_labels`."""
    df = lb_v2[lb_v2["plays"] >= min_plays].copy()
    if df.empty:
        raise ValueError("Leaderboard has no rows above min_plays")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))

    ax.axhline(0, color="#888", lw=0.8, ls="--", zorder=1)
    ax.scatter(df["contests"], df["shadow_over_expected"],
               s=28, c="#2a4d7a", alpha=0.65, edgecolors="white",
               linewidths=0.5, zorder=2)

    ranked = df.sort_values("shadow_over_expected", ascending=False)
    highlight = pd.concat([ranked.head(n_labels), ranked.tail(n_labels)])
    for _, r in highlight.iterrows():
        color = "#0a7d2a" if r["shadow_over_expected"] > 0 else "#a30000"
        ax.scatter(r["contests"], r["shadow_over_expected"],
                   s=55, c=color, edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(r["player_name"],
                    (r["contests"], r["shadow_over_expected"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=color, zorder=4)

    ax.set_xlabel("Contests (closest defender on a resolved pass)")
    ax.set_ylabel("Shadow Over Expected (completions prevented)")
    ax.set_title(title or f"Shadow Over Expected — 2023 season (min {min_plays} snaps)",
                 fontsize=11)
    ax.grid(True, alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax
