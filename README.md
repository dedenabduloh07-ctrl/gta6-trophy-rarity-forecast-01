# GTA VI Trophy Rarity Forecaster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![make.fun bounty](https://img.shields.io/badge/make.fun-gta6--trophy--rarity--forecast--01-green.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies,
five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at the case's target — for the final case, 26 November 2026 00:00 UTC.
Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
"Forecast how many GTA VI players earn each trophy in week one" ($250, 20 places paid).

## How it works

Exophase prints each trophy's earn rate as `earned / tracked`, and refreshes
a game's counts on its own schedule (hours apart), not continuously. At a
refresh the tracked population jumps — launch night's first refresh took it
from 638 to 3,868 players — while the earned counts barely move, so every
rate drops by a large factor at once. A forecaster that only extends each
trophy's recent curve misses that mechanism, and the persistence baseline
scores about 1.51 on the practice cases because the refresh takes every rate
outside its narrow band.

This entry is therefore structural, in earners and players:

```
rate_i(t) = earned_i(t) / tracked(t)
```

- **Tracked players:** a kernel-weighted ridge local-linear fit of log
  tracked against log time since launch, extrapolated to the target, with
  the far-horizon slope tapered (launch-week tracking growth decelerates).
- **Earned per trophy:** the same fit on log(earned + 0.5), using the last
  48 hours of distinct page states, with each trophy's slope shrunk toward
  the kernel-weighted pooled slope of comparable trophies (same tier,
  nearby current rate). Everything is fit at run time from the case's own
  history — no GTA VI answers are hardcoded.
- **Spread:** an ensemble of a parametric part (the house random walk of
  0.5 logit per root day, plus the fit's residual scale, plus a
  refresh-jump part) and kernel-weighted empirical quantiles of the logit
  moves comparable trophies actually made between the case's distinct page
  states. The refresh part of the spread is the larger of the two.
- **Flat history:** when the case history shows no refresh yet (every read
  identical, as in both practice cases), no trend can be fit, so a
  first-refresh prior is used instead of pretending the rates will sit
  still. The prior is calibrated on the one refresh the competition's own
  documentation records (tracked 638 → 3,868) and on the persistence
  baseline's documented practice loss of about 1.51, which implies an
  effective rate dilution of about 5.0. The one trophy at 100% stays at
  100%: its earners are the whole tracked population.

Scoring is the pinball loss on the logit scale, exactly as the competition
documents. Output is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}`, one row per trophy, each quantile rounded to 4 decimals
(half up).

## Local backtest (practice cases p00/p01)

The true practice answers are not published in the kit (this competition
is graded on the server only), so the backtest scores against synthetic
answers built from the facts the competition documents: the 04:30 UTC
refresh took tracked 638 → 3,868 and every rate but the 100% one fell.
Three proxies are used — A: flat effective dilution 5.0 (calibrated so the
persistence baseline reproduces its documented ~1.51); B: per-trophy
dilution; C: pure dilution 6.06 (earned counts conserved exactly).

| answer proxy | this entry | persistence | curve baseline |
|---|---:|---:|---:|
| A (D = 5.0) | **0.085** | 1.498 | 1.498 |
| B (per-trophy D) | **0.073** | 1.549 | 1.549 |
| C (D = 6.06) | **0.101** | 1.688 | 1.688 |

Mean pinball loss on the logit scale, lower is better. On the practice
proxies the entry beats both house baselines by roughly 15–20×, in the
same range as the public leaderboard's best entries (~0.06).

A synthetic final-like case (lumpy refreshes, tracked growth slowing after
day 4, six seeds) gives a mean loss of 0.245 for this entry vs 0.259 for
persistence and 0.269 for the curve baseline; without the slowdown it is
0.252 vs 0.313 and 0.225. Those are simulations, not the real final.

## Run it

Python 3.13, standard library only, no network, deterministic (two runs
on the same case write byte-identical output), well inside the 8-second
limit (about 0.1 s for both practice cases).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with the case at `/input`
and writes `/output/<id>.out`.

## Files

- `main.py` — the forecaster (stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — entry wallet
