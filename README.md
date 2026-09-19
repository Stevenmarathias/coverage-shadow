# Coverage Shadow

An NFL player-tracking metric that quantifies how much catch window each defender erases on a pass.

Built on NFL Big Data Bowl 2026 tracking data (2023–2024 seasons). Companion to
[Catch Radius Pressure](https://github.com/Stevenmarathias) — CRP measures the receiver's side of the
catch point; Coverage Shadow measures the defense's.

## The idea

At the moment the ball is released, every player is racing to where the ball will land.
Using each player's position, speed and direction, the model estimates their time to reach
the landing spot. The targeted receiver's lead over the fastest-arriving defender is the
**catch window**. A defender's **Shadow** on a play is how much that window would grow if
they were removed from the field — the space they personally took away.

Aggregate over a season and you get a leaderboard of who actually contests throws.

## 2023 season (v1)

18 weeks, 14,107 plays. Total Shadow is a volume stat that rewards heavily targeted
corners; average Shadow is a per-play rate. Total board uses a 30-snap minimum;
avg board uses 100 to filter small-sample noise.

| # | By total Shadow | Total (s) | Plays | | By avg Shadow (≥100 snaps) | Avg (s) | Plays |
|---|---|---|---|---|---|---|---|
| 1 | Deonte Banks | 87.94 | 333 | | Deonte Banks | 0.26 | 333 |
| 2 | Benjamin St-Juste | 86.92 | 419 | | Josh Jobe | 0.25 | 117 |
| 3 | Ahkello Witherspoon | 86.15 | 443 | | Emmanuel Forbes | 0.23 | 203 |
| 4 | Tyrique Stevenson | 82.28 | 393 | | J.C. Jackson | 0.23 | 221 |
| 5 | Brandon Stephens | 78.14 | 444 | | Tre Avery | 0.23 | 135 |
| 6 | Zyon McCollum | 75.57 | 337 | | Ronald Darby | 0.23 | 189 |
| 7 | Charvarius Ward | 74.28 | 415 | | Darrell Baker Jr. | 0.23 | 186 |
| 8 | D.J. Reed | 72.71 | 329 | | Zyon McCollum | 0.22 | 337 |
| 9 | Michael Davis | 71.48 | 380 | | Montaric Brown | 0.22 | 208 |
| 10 | James Bradberry | 68.36 | 464 | | Shaun Wade | 0.22 | 142 |

## Validation

Does Shadow actually track pass outcomes? Joining `pass_result` from the Big Data Bowl
supplementary file, restricted to completions and incompletions (13,770 plays;
interceptions excluded as a distinct outcome):

- **Top-quartile closest-defender Shadow (≥ 1.03s):** 62.5% completion (n = 3,443)
- **Bottom-quartile closest-defender Shadow (≤ 0.26s):** 75.0% completion (n = 3,443)
- **corr(catch_window, completed) = +0.332**

Passes where the nearest defender erases the most window complete ~12.5 points less often,
and catch window itself is moderately correlated with completion in the expected direction.
v1 is a real signal, not noise.

## Run it

    pip install -r requirements.txt
    python run.py data/raw/input_2023_w01.csv

Outputs land in `outputs/` as play-level scores and a leaderboard.

## Limitations

v1 Shadow measures **contest opportunity, not contest quality.** An isolated corner
with no safety help earns a large Shadow whether or not he actually wins the rep —
the metric rewards the geometry of being the nearest defender, not the result of the
throw. It also can't distinguish a tight-window completion from a tight-window PBU:
both look the same at the release frame. v2 will split Shadow by outcome (completed,
incomplete, PBU, INT) so quality separates from opportunity.

## Roadmap

- **v1 (current):** closing-time model; only the closest-arriving defender earns credit
- **v2:** soft credit so a second defender closing hard earns partial Shadow
- **v3:** ball-in-the-air extension — how the window collapses frame by frame
- **Validation:** does Shadow predict incompletions and pass breakups?
- **Visuals:** field heatmaps of each defender's shadow

## Data & license

Tracking data is from the NFL Big Data Bowl 2026 on Kaggle, licensed CC BY-NC 4.0.
Raw data is not committed; see `data/README.md`. This project is non-commercial.
