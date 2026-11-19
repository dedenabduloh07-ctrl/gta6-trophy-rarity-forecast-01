"""gta6-trophy-rarity-forecast-01: refresh-aware earners/players forecaster.

Forecasts, for each of GTA VI's 55 PlayStation 5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the case's target.
Python 3.13, standard library only, deterministic, offline.

Why this shape, and not a plain trend extrapolation
---------------------------------------------------
Exophase prints each trophy's earn rate as earned / tracked. It refreshes a
game's counts on its own schedule (hours apart), not continuously. At a
refresh the tracked population jumps -- launch night's first refresh took it
from 638 to 3,868 players -- while the earned counts barely move, so every
rate drops by a large factor at once. A forecaster that only extends each
trophy's recent curve (the house curve baseline) misses that mechanism; the
persistence baseline (the kit's starter) misses it too, and scores about
1.51 on the practice cases because the refresh takes every rate outside its
narrow band.

The model here is therefore structural, in earners and players:

  rate_i(t) = earned_i(t) / tracked(t)

* tracked(t): a kernel-weighted ridge local-linear fit of log tracked
  against log time since launch, extrapolated to the target.
* earned_i(t): the same fit per trophy on log(earned + 0.5), with each
  trophy's slope shrunk toward the kernel-weighted pooled slope of
  comparable trophies (same tier, nearby current rate). Comparables are the
  case's own other trophies, so everything is fit at run time from the
  case's own history.
* The median forecast is earned_i(target) / tracked(target).

The spread is an ensemble of (a) a parametric part -- the house random
walk (0.5 logit per root day) plus the fit's residual scale plus a
refresh-jump part -- and (b) kernel-weighted empirical quantiles of the
logit moves comparable trophies actually made between the case's distinct
page states, scaled by the refreshes expected before the target. The
refresh part of the spread is the larger of the parametric allowance and
the empirical one. With fewer than 3 comparables showing a move, (b) is
skipped.

When the case history shows no refresh yet (every read identical, as in
both practice cases), no trend can be fit, so a first-refresh prior is
used instead of pretending the rates will sit still. The prior is
calibrated on the one refresh the kit's own documentation records:
launch night, tracked 638 -> 3,868 (a factor of 6.06), while the
persistence baseline's documented practice loss of about 1.51 implies an
effective rate dilution of about 5.0 (the new players had earned some
trophies too). The prior therefore dilutes each rate by

  D_i = 6.06 / (1 + 0.35 * r_i / r_max)

so the earliest, most-earned trophies are diluted least, and centres the
median on r_i / D_i. The one trophy at 100% stays at 100%: its earners
are the whole tracked population, so a refresh cannot dilute it.

Scoring is the pinball loss on the logit scale, exactly as the kit
documents: loss = 2 * max(tau * d, (tau - 1) * d) with
d = logit(y) - logit(q), rates clamped to 0.01..99.99 before the logit.

Output format (identical to the starter): {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}, one row per trophy, each q five
finite numbers from 0 to 100, non-decreasing, rounded to 4 decimals
(half up, as JavaScript's Math.round rounds it).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (the
# house random-walk component, same as the persistence baseline).
SIGMA_RW = 0.5

# --- first-refresh prior (flat history only; see the module docstring) ---
# Launch night's first refresh, as the kit's competition.yaml records it:
# players tracked 638 -> 3,868.
FIRST_REFRESH_RATIO = 3868.0 / 638.0  # 6.0627
# Effective dilution implied by the persistence baseline's documented
# practice loss of about 1.51: the new players had earned some trophies,
# so rates fell by ~5.0, not by the full tracked ratio. The per-trophy
# factor below averages to that effective dilution.
PRIOR_BETA = 0.35
# Logit spread of one refresh's magnitude (the refresh size is uncertain).
SIGMA_REFRESH_ONE = 0.35
# Logit spread per refresh expected before the target. Refresh jumps are
# largely predictable through the tracked/earned extrapolation, and with
# many refreshes before a multi-day target their deviations partly average
# out, so this is deliberately smaller than SIGMA_REFRESH_ONE.
SIGMA_REFRESH_EACH = 0.15

# Launch's time origin, the same one the house curve baseline uses.
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# A trophy at (essentially) 100% in the last read stays there: its earners
# are the whole tracked population.
UNIVERSAL_RATE = 99.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_t0(moment: datetime) -> float:
    return (moment - T0).total_seconds() / 3600.0


def x_of(hours: float) -> float:
    """The house's time transform: log time since launch."""
    return math.log1p(max(0.0, hours))


def distinct_states(case: dict) -> list:
    """Collapse the history into distinct page states.

    Exophase refreshes hours apart, so consecutive reads are usually the
    same page state. Each distinct state is one observation: its slot
    time, players tracked, and rates.
    """
    states = []
    prev_key = None
    for read in case["history"]:
        key = (read.get("players_tracked"), tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        states.append(
            {
                "t": hours_since_t0(parse_time(read["slot"])),
                "tracked": read["players_tracked"],
                "rates": list(read["rates"]),
            }
        )
    return states


def ridge_line(xs, ys, ws, lam, prior_slope=0.0):
    """Weighted least squares line y ~ a + b*x, ridge-shrunk toward prior_slope.

    Returns (a, b, resid_std). Closed form, deterministic.
    """
    sw = sum(ws)
    if sw <= 0:
        return (ys[0] if ys else 0.0, prior_slope, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = (sxy + lam * prior_slope) / (sxx + lam) if (sxx + lam) > 0 else prior_slope
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw if sw > 0 else 0.0
    return (a, b, math.sqrt(max(0.0, var)))


def kernel_similarity(rate_a: float, rate_b: float, tier_a: str, tier_b: str) -> float:
    """Similarity between two trophies: nearby logit rate, same tier preferred."""
    d = abs(logit(rate_a) - logit(rate_b))
    w = math.exp(-0.5 * (d / 1.0) ** 2)
    if tier_a == tier_b:
        w *= 1.5
    return w


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    last = case["history"][-1]
    last_rates = [r if r is not None else 0.0 for r in last["rates"]]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = hours_since_t0(parse_time(case["target"]))
    x_target = x_of(target_hours)
    sigma_rw = SIGMA_RW * math.sqrt(horizon_days)

    states = distinct_states(case)

    # --- medians and spreads, per trophy ---
    medians = [0.0] * n
    sigmas = [0.0] * n

    if len(states) < 2:
        # Flat history: no trend can be fit. Use the first-refresh prior.
        r_max = max([r for r in last_rates if r < 100.0] or [1.0])
        sigma_prior = math.sqrt(sigma_rw * sigma_rw + SIGMA_REFRESH_ONE ** 2)
        for i in range(n):
            r = last_rates[i]
            if r >= UNIVERSAL_RATE:
                medians[i] = r
                sigmas[i] = 0.10
                continue
            s = r / r_max if r_max > 0 else 1.0
            dilution = FIRST_REFRESH_RATIO / (1.0 + PRIOR_BETA * s)
            medians[i] = clamp(r / dilution)
            sigmas[i] = sigma_prior
    else:
        # --- structural fit on the distinct states ---
        xs = [x_of(s["t"]) for s in states]
        m = len(states)
        # Recency kernel: later states count more (the page's recent
        # behaviour is the best guide to its near future), but gently:
        # a steep kernel plus the ridge below would shrink the fitted
        # slopes toward zero and bias the extrapolation.
        ws = [0.85 ** (m - 1 - j) for j in range(m)]

        # Tracked players: ridge line of log tracked on log time.
        log_n = [math.log(max(1, s["tracked"])) for s in states]
        a_n, b_n, _ = ridge_line(xs, log_n, ws, lam=0.1, prior_slope=0.0)
        # Taper the far-horizon extrapolation: launch-week tracking growth
        # decelerates (the rush fades), so the fitted slope is applied at
        # full strength only for the first 0.5 units of log-time beyond the
        # last state, then at 60% strength. This is a fixed prior, not
        # tuned to any case: it guards against over-diluting the rates when
        # the tracked population grows more slowly than the launch trend.
        dx = x_target - xs[-1]
        dx_eff = 0.5 * dx if dx <= 0.5 else 0.5 + 0.6 * (dx - 0.5)
        log_n_target = a_n + b_n * (xs[-1] + dx_eff)
        n_last = states[-1]["tracked"]
        # Cap the extrapolation: tracked cannot fall, and cannot plausibly
        # grow more than twenty-fold over one horizon.
        log_n_target = min(math.log(n_last * 20.0), max(math.log(n_last), log_n_target))
        n_target = math.exp(log_n_target)

        # Refresh cadence from the distinct states (hours between refreshes).
        gaps = [states[j + 1]["t"] - states[j]["t"] for j in range(m - 1)]
        gaps = sorted(g for g in gaps if g > 0)
        cadence = gaps[len(gaps) // 2] if gaps else 24.0
        cadence = min(48.0, max(1.0, cadence))
        horizon_hours = case["horizon_minutes"] / 60.0
        expected_refreshes = min(4.0, horizon_hours / cadence)
        sigma_refresh = SIGMA_REFRESH_EACH * math.sqrt(expected_refreshes)

        # Per-trophy earned counts, and each trophy's own slope. The earned
        # fits use only the last 48 hours of states: early states have tiny
        # earned counts (a handful of players), where log(earned + 0.5) is
        # noisy and the continuity correction distorts the slope. (If that
        # leaves fewer than 3 states, all states are used.)
        t_last = states[-1]["t"]
        recent = [j for j in range(m) if states[j]["t"] >= t_last - 48.0]
        if len(recent) < 3:
            recent = list(range(m))
        xs_e = [xs[j] for j in recent]
        ws_e = [0.85 ** (len(recent) - 1 - k) for k in range(len(recent))]
        own_slope = [0.0] * n
        own_resid = [0.0] * n
        for i in range(n):
            ys = []
            for j in recent:
                s = states[j]
                r = s["rates"][i]
                e = (r if r is not None else 0.0) * s["tracked"] / 100.0
                ys.append(math.log(e + 0.5))
            _, b, resid = ridge_line(xs_e, ys, ws_e, lam=0.1, prior_slope=0.0)
            own_slope[i] = b
            own_resid[i] = resid

        # Pooled slope per trophy: kernel-weighted mean of the other
        # trophies' slopes, weighted by similarity (tier + current rate).
        pooled_resid = sorted(own_resid)[len(own_resid) // 2] if own_resid else 0.2
        sigma_fit = min(0.8, max(0.15, pooled_resid))
        for i in range(n):
            r = last_rates[i]
            if r >= UNIVERSAL_RATE:
                medians[i] = r
                sigmas[i] = 0.10
                continue
            num = 0.0
            den = 0.0
            for j in range(n):
                if j == i:
                    continue
                w = kernel_similarity(r, last_rates[j], trophies[i]["tier"], trophies[j]["tier"])
                num += w * own_slope[j]
                den += w
            pooled = num / den if den > 0 else 0.0
            # Shrink the own slope toward the pooled slope: with few
            # recent states the pooled slope dominates; with many, the
            # trophy's own trajectory does.
            kappa = 2.0
            m_e = len(recent)
            slope = (m_e * own_slope[i] + kappa * pooled) / (m_e + kappa)
            e_last = r * n_last / 100.0
            log_e_target = math.log(e_last + 0.5) + slope * (x_target - xs[-1])
            # Earned cannot exceed the tracked population, nor go below 0.5.
            log_e_target = min(math.log(n_target), max(math.log(0.5), log_e_target))
            medians[i] = clamp(100.0 * math.exp(log_e_target) / n_target)
            sigmas[i] = math.sqrt(sigma_rw * sigma_rw + sigma_fit * sigma_fit + sigma_refresh * sigma_refresh)

        # --- (b) empirical-quantile spread from comparable trophies' moves ---
        # For each trophy, collect the logit moves comparable trophies made
        # between consecutive distinct states (those moves are the
        # refresh jumps). Their weighted spread, scaled by the number of
        # refreshes expected before the target, is an empirical spread
        # estimate; it is averaged with the parametric spread. The median
        # is NOT shifted by it: the structural fit already accounts for
        # the expected dilution. Skipped when fewer than 3 moves exist.
        moves = []  # (trophy index, logit move)
        for j in range(m - 1):
            for i in range(n):
                r0 = states[j]["rates"][i]
                r1 = states[j + 1]["rates"][i]
                if r0 is None or r1 is None:
                    continue
                d = logit(r1) - logit(r0)
                if d == 0.0:
                    continue
                moves.append((i, d))
        if len(moves) >= 3:
            for i in range(n):
                if last_rates[i] >= UNIVERSAL_RATE:
                    continue
                num = 0.0
                den = 0.0
                for (j, d) in moves:
                    w = kernel_similarity(last_rates[i], last_rates[j], trophies[i]["tier"], trophies[j]["tier"])
                    num += w * d
                    den += w
                if den <= 0:
                    continue
                mean = num / den
                var = 0.0
                for (j, d) in moves:
                    w = kernel_similarity(last_rates[i], last_rates[j], trophies[i]["tier"], trophies[j]["tier"])
                    var += w * (d - mean) ** 2
                s_emp = math.sqrt(var / den) if den > 0 else 0.0
                sigma_emp = s_emp * math.sqrt(expected_refreshes)
                # The refresh part of the spread is the larger of the
                # parametric allowance and the empirical one: observed
                # refreshes can only widen the honest spread, never narrow it.
                sigma_ref = max(sigma_refresh, sigma_emp)
                sigmas[i] = math.sqrt(
                    sigma_rw * sigma_rw + sigma_fit * sigma_fit + sigma_ref * sigma_ref
                )

    # --- assemble rows ---
    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        q = [round4(clamp(inv_logit(lg + z * sigmas[i]))) for z in Z]
        # Rounding can only keep the order (the underlying values are
        # strictly increasing); enforce non-decreasing defensively.
        q = [max(q[k], q[k - 1]) if k > 0 else q[k] for k in range(len(q))]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
