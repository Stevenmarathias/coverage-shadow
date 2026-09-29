# I Built a Cornerback Metric, Then Tried to Break It

*Steven Marathias · September 2026*

I spent a few weeks building a metric to measure how well cornerbacks cover. It passed its first test, looked great on a leaderboard, and then failed the tests that actually mattered.

This is the story of that metric, why it failed, and why I think the failure is the most useful thing I built. If you evaluate players for a living, or want to, the lesson applies well beyond cornerbacks: a metric that looks right and a metric that measures skill are not the same thing, and the only way to tell them apart is to try to break your own work.

## What I built

The data is the NFL's player tracking from the 2026 Big Data Bowl: the position, speed and direction of every player, ten times a second, for the 2023 season.

The idea behind **Coverage Shadow** is simple. At the moment the quarterback releases the ball, I estimate how long it would take each player to reach the spot where it lands, using his position, speed and direction. The gap between the targeted receiver's arrival and the nearest defender's arrival is the catch window. A defender's Shadow is how much that window would grow if he weren't on the field. In plain terms, it's how much catchable space he takes away.

The first version had an obvious flaw. Summed over a season, it mostly rewarded corners who got targeted a lot. So I built a second version, **Shadow Over Expected (SOE)**. I fit a model of completion probability from the catch window, then credited each defender with the completions he prevented relative to that expectation.

The early evidence looked strong. On passes where the top-quartile Shadow defender was in coverage, quarterbacks completed 62.5% of throws. In the bottom quartile, 75.0%. That's a 12.5-point gap across nearly 14,000 plays. The SOE leaderboard was topped by names that made sense, like Stephon Gilmore and Kendall Fuller.

It looked like I had something. I didn't know yet what that something was.

## Test one: is it skill or luck?

A stat that measures skill should repeat. If a corner is genuinely good, his number in one half of the season should look like his number in the other half. So I split 2023 into odd and even weeks and checked how well each metric agreed with itself. The result is reliability on a 0 to 1 scale, where higher means more repeatable.

| Metric (2023 cornerbacks) | Full-season reliability |
| --- | --- |
| Coverage Shadow, average per snap | 0.60 |
| PFR completion % allowed | 0.61 |
| PFR passer rating allowed | 0.28 |
| Shadow Over Expected | 0.10 |

![Split-half reliability, 2023 cornerbacks](figures/stability_split_half_2023.png)

SOE was essentially noise. The spread in SOE across corners was almost exactly what you'd get if every completion were a coin flip weighted by the model. The leaderboard that looked so sensible was mostly ranking luck.

Coverage Shadow, though, held up. It was as repeatable as completion percentage allowed, which is the standard coverage stat. For context, coverage stats are famously unstable: from one season to the next, completion percentage allowed correlates at only about 0.19 across 2018 to 2025. Matching it was a real result.

So SOE was out, and Shadow was in. Or so I thought.

## Test two: does it measure the player or his job?

A stable number isn't automatically a skill. It could be stable because a corner's role is stable. A slot corner and an outside corner do different jobs every week, and so do corners in different schemes.

So I controlled for alignment (slot or outside, depth, cushion from the receiver) and for scheme. Alignment mattered: slot corners scored noticeably lower, and role explained about a quarter of Shadow's reliability. But after removing it, reliability stayed around 0.50. Something player-specific was still there.

Then came the test I should have run first. If Shadow measures coverage skill, a corner's first-half Shadow should predict how he performs in the second half. I compared it against the obvious baseline: his own first-half completion percentage allowed.

| Predicting second-half completion % allowed | Weeks 1–9 → 10–18 | Odd → even weeks |
| --- | --- | --- |
| First-half completion % allowed | +0.27 | +0.36 |
| First-half Coverage Shadow | +0.15 | +0.09 |

![Predictive validity, 2023 cornerbacks](figures/predictive_validity_2023.png)

Shadow lost. Adding it on top of completion percentage improved the prediction by less than one percentage point of explained variance.

One more check explained why. Corners with high Shadow weren't avoided by quarterbacks. They were thrown at *more*. Shadow was strongly tied to how often a corner was the closest defender to the throw.

That's the real finding. Coverage Shadow is a stable measure of how often a corner is around the ball. That's a genuine part of his job, but it isn't the same as making those throws fail. It also explains the promising early result: the 62.5% versus 75.0% gap came from the same plays the metric was built on, so part of it was baked in. The predictive test is the stricter one, and Shadow didn't pass it.

## One last try, with the rules set first

There was one obvious fix left. Split Shadow into two parts: **involvement**, how often a corner is near the throw, and **effect**, how much window he takes away when he is. The second part is the actual skill question.

This is exactly the moment where it's easy to fool yourself. With enough versions, one will eventually look good by chance. So before computing a single number, I wrote down the pass/fail bar and committed the definitions and test code to the repo. Effect would pass only if it was reliable (at least 0.40) and predicted second-half completion percentage at least as well as completion percentage itself, in both splits.

Both parts failed.

- **Involvement** was reliable (0.47) but predicted nothing about completions. It did predict how often a corner got targeted, which accounts for most of the original Shadow's stability.
- **Effect** was the best outcome predictor I found, but its reliability was only 0.22. Each corner had fewer than 50 involved plays per half-season, which isn't enough to separate skill from noise. Even a second full season of tracking data would only lift it to about 0.36, still short of the bar.

I stopped there. A fourth version would have been the start of tuning the test until something passed.

## What I learned

The honest conclusion is narrower than "the metric failed." With one season of tracking data, a corner's involvement around the throw is measurable and stable. How much he actually shrinks the window when he's there is not measurable yet, because there aren't enough plays per player to separate skill from noise. That's a finding about the limits of coverage evaluation, not just about my metric.

Three things I'd tell anyone building player metrics:

1. **Stable isn't the same as skill.** A number can repeat because the player's role repeats. Control for the job before crediting the player.
2. **Test prediction, not just description.** A metric that explains the plays it was built from will always look good. Ask whether it predicts the next ones.
3. **Set the bar before you look.** Writing down pass/fail criteria first is the only protection against trying versions until one works.

I've retired Shadow Over Expected entirely and relabeled Coverage Shadow on my site, [Catch Point Analytics](https://catch-point-analytics.fly.dev), as an involvement measure rather than a coverage grade. The full analysis, including every table, the code and the committed test definitions, is in the [coverage-shadow repo](https://github.com/Stevenmarathias/coverage-shadow).

Building a metric is the easy part. Finding out what it really measures is the work.
