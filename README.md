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

## Week 1, 2023 (v1)

| Player | Pos | Plays | Total Shadow (s) | Avg |
|---|---|---|---|---|
| Denzel Ward | CB | 21 | 10.73 | 0.51 |
| D.J. Reed | CB | 31 | 9.30 | 0.30 |
| A.J. Terrell | CB | 27 | 8.87 | 0.33 |
| Dre Greenlaw | ILB | 34 | 8.86 | 0.26 |
| Christian Gonzalez | CB | 26 | 7.94 | 0.31 |

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
