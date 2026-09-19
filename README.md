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

Top 10 by total Shadow, minimum 30 coverage snaps, across all 18 weeks (14,107 plays).

| Player | Pos | Plays | Total Shadow (s) | Avg |
|---|---|---|---|---|
| Deonte Banks | CB | 333 | 87.94 | 0.26 |
| Benjamin St-Juste | CB | 419 | 86.92 | 0.21 |
| Ahkello Witherspoon | CB | 443 | 86.15 | 0.19 |
| Tyrique Stevenson | CB | 393 | 82.28 | 0.21 |
| Brandon Stephens | CB | 444 | 78.14 | 0.18 |
| Zyon McCollum | CB | 337 | 75.57 | 0.22 |
| Charvarius Ward | CB | 415 | 74.28 | 0.18 |
| D.J. Reed | CB | 329 | 72.71 | 0.22 |
| Michael Davis | CB | 380 | 71.48 | 0.19 |
| James Bradberry | CB | 464 | 68.36 | 0.15 |

## Run it

    pip install -r requirements.txt
    python run.py data/raw/input_2023_w01.csv

Outputs land in `outputs/` as play-level scores and a leaderboard.

## Roadmap

- **v1 (current):** closing-time model; only the closest-arriving defender earns credit
- **v2:** soft credit so a second defender closing hard earns partial Shadow
- **v3:** ball-in-the-air extension — how the window collapses frame by frame
- **Validation:** does Shadow predict incompletions and pass breakups?
- **Visuals:** field heatmaps of each defender's shadow

## Data & license

Tracking data is from the NFL Big Data Bowl 2026 on Kaggle, licensed CC BY-NC 4.0.
Raw data is not committed; see `data/README.md`. This project is non-commercial.
