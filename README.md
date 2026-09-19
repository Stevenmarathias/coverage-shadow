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

## 2023 season (v2 — Shadow Over Expected)

v2 grades contests instead of just counting them. Every play with a resolved outcome
gets an **expected completion** from a logistic regression of completion on catch
window (fit across all 13,770 completions/incompletions in 2023:
`P(C) = sigmoid(+0.158 + 1.755 · catch_window)`). A defender's **Shadow Over Expected**
(SOE) on a play is the model's expected completion minus what actually happened — so
forcing an incompletion on a wide-open target earns big positive credit, while getting
beaten on a tight window barely dents the score. Summed over the season (closest
defender per play, min 100 coverage snaps) it rewards defenders who beat the geometry
their own Shadow assigned them.

| # | Player | Pos | Plays | Contests | Won (s) | Lost (s) | Win rate | SOE |
|---|---|---|---|---|---|---|---|---|
| 1 | Stephon Gilmore | CB | 372 | 72 | 24.47 | 34.24 | 41.7% | +10.03 |
| 2 | Kendall Fuller | CB | 402 | 54 | 22.06 | 19.87 | 52.6% | +9.53 |
| 3 | Levi Wallace | CB | 314 | 61 | 29.61 | 27.04 | 52.3% | +8.81 |
| 4 | Paulson Adebo | CB | 379 | 75 | 25.13 | 18.15 | 58.1% | +8.50 |
| 5 | Ahkello Witherspoon | CB | 443 | 85 | 38.54 | 43.24 | 47.1% | +8.40 |
| 6 | Greg Newsome II | CB | 282 | 55 | 25.84 | 16.87 | 60.5% | +7.76 |
| 7 | Devon Witherspoon | CB | 333 | 66 | 22.31 | 35.18 | 38.8% | +7.31 |
| 8 | Zyon McCollum | CB | 337 | 72 | 36.70 | 38.87 | 48.6% | +7.29 |
| 9 | Ja'Sir Taylor | CB | 250 | 41 | 19.51 | 15.19 | 56.2% | +7.23 |
| 10 | Darious Williams | CB | 465 | 77 | 26.33 | 33.23 | 44.2% | +7.21 |

`shadow_won` and `shadow_lost` sum a defender's Shadow across the incompletions and
completions they contested; `win_rate = won / (won + lost)`. Note how SOE reshuffles
the board: Deonte Banks, the v1 volume leader, drops to 19th — he contests a lot but
converts about as often as the geometry would predict.

## Run it

    pip install -r requirements.txt
    python run.py data/raw/input_2023_w01.csv

Outputs land in `outputs/` as play-level scores and a leaderboard.

## Limitations

Even v2 uses only the release-frame snapshot: the ball hasn't left the QB's hand yet
in the model's view of the world, so a tight-window completion and a tight-window PBU
still look identical at scoring time. v2 also gives all credit to the single closest
defender; a second defender closing hard gets nothing. Both are on the roadmap.

## Roadmap

- **v1 (done):** closing-time model; only the closest-arriving defender earns credit
- **v2 (done):** outcome-aware Shadow — win/loss splits and Shadow Over Expected
- **v3:** soft credit so a second defender closing hard earns partial Shadow
- **v4:** ball-in-the-air extension — how the window collapses frame by frame
- **Visuals:** field heatmaps of each defender's shadow

## Data & license

Tracking data is from the NFL Big Data Bowl 2026 on Kaggle, licensed CC BY-NC 4.0.
Raw data is not committed; see `data/README.md`. This project is non-commercial.
