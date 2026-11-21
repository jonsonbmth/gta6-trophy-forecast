"""Refresh-inversion trophy rarity forecaster for gta6-trophy-rarity-forecast-01.

Forecasts every GTA VI PS5 trophy's Exophase earn rate at the case target as five
quantiles, from the case's history of Exophase page reads.

Model (all standard library, deterministic, no network):

1.  Refresh states.  Exophase refreshes a game's counts on its own schedule, so
    consecutive reads are usually identical.  The history is collapsed to its
    distinct states (players tracked, earn rates).  Every fit below runs on
    states, never on the frozen reads between refreshes.

2.  Player growth.  Players tracked P(t) is forecast to the target as the
    geometric mean of three views: a power law in log-time fitted on recent
    states, a saturating Michaelis-Menten curve fitted on all states, and the
    last observed growth rate held with a decay.  The spread of the three views
    measures player-count uncertainty, which moves every trophy's rate together.

3.  Per-trophy earners.  Two views, blended in logit space:
    (a) a trend view: logit(rate) linear in ln(1 + hours since launch), fitted
        on states, anchored at the last observed rate, with its slope damped
        toward zero when states are few (a two-state slope mostly measures one
        big refresh, not the going trend);
    (b) a structural view (only when the history holds >= 3 refresh
        transitions): each transition's new earners are split into incumbents
        earning at a hazard h and a new cohort arriving already holding the
        trophy at rate c, by least squares over the transitions; the forecast
        then grows incumbents at h and adds new cohorts at c on top of the
        player forecast.  This inverts the refresh mechanics instead of
        extrapolating the diluted rate.
    Rare trophies (few earners) get a wider band: a handful of players'
    progress can double their rate.  A count view grows earners at each
    trophy's own measured per-hour rate (decayed over the horizon) and divides
    by the player forecast, so a trophy whose rate was flat only because player
    growth masked its earners' growth is still forecast to rise as player
    growth slows; it is blended with the trend view in logit space.

4.  Count consistency.  Medians are capped by the story chain (a later chapter
    cannot exceed an earlier one, the platinum cannot exceed any trophy) via an
    isotonic pass per quantile level, and every rate is clamped to [0.01, 99.99].

5.  Spreads.  Each trophy's logit spread combines player-count uncertainty, its
    own fit residual, and a horizon random-walk term, so the 5%-95% band is
    honest about both the common dilution risk and the trophy's own noise.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg > 30.0:
        return 99.99
    if lg < -30.0:
        return 0.01
    return clamp_pct(100.0 / (1.0 + math.exp(-lg)))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t):
    return (t - LAUNCH).total_seconds() / 3600.0


def log_time(t):
    """ln(1 + hours since launch); pre-launch reads sit just below zero."""
    h = hours_since_launch(t)
    return math.log(max(0.05, 1.0 + h))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def least_squares(xs, ys):
    """Slope and intercept of y ~ a + b*x. Returns (b, a)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def isotonic_decreasing(values):
    """Pool-adjacent-violators for a non-increasing sequence (least squares)."""
    # Walk left to right, pooling any block that rises above its left neighbour.
    stack = []  # each entry: [sum, count]
    for v in values:
        s, c = v, 1.0
        while stack and stack[-1][0] / stack[-1][1] < s / c:
            ps, pc = stack.pop()
            s += ps
            c += pc
        stack.append([s, c])
    res = []
    for s, c in stack:
        res.extend([s / c] * int(c))
    return res


def extract_states(case):
    """Collapse the history to distinct refresh states.

    A state is (time, players, [rates]); consecutive identical reads merge into
    one state whose time is the first read showing the new values.  (The
    refresh happened before that read, but the read time is what the forecaster
    honestly knows; for the dense final history the two differ by minutes.)
    """
    hist = case["history"]
    states = []
    for read in hist:
        at = parse_time(read["at"])
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1][2] == key:
            continue
        states.append((at, read["players_tracked"], key, list(read["rates"])))
    return states


def forecast_players(states, target_t):
    """Forecast players tracked at the target; returns (P, sigma_log).

    Geometric mean of three views: recent power law, saturating curve, and the
    last growth rate decayed.  Sigma is the spread of the views in log space.
    """
    times = [s[0] for s in states]
    ps = [float(s[1]) for s in states]
    last_t, last_p = times[-1], ps[-1]
    dt_h = max(0.0, (target_t - last_t).total_seconds() / 3600.0)
    if dt_h <= 0.0 or len(states) < 2:
        return last_p, 0.10
    x_t = log_time(target_t)
    views = []

    # View 1: power law ln P ~ a + b * ln(1 + hours), fitted on recent states,
    # anchored at the last state, slope capped: launch-week growth is concave
    # in log-log (it decelerates), so an uncapped fit on the launch jump
    # wildly over-extrapolates.
    recent = [(s[0], s[1]) for s in states
              if (last_t - s[0]).total_seconds() <= 24 * 3600]
    if len(recent) < 2:
        recent = [(s[0], s[1]) for s in states[-4:]]
    if len(recent) >= 2:
        xs = [log_time(t) for t, _ in recent]
        ys = [math.log(max(1.0, p)) for _, p in recent]
        b, _a = least_squares(xs, ys)
        b = min(max(b, 0.0), 0.7)
        views.append(math.log(max(1.0, last_p)) + b * (x_t - log_time(last_t)))

    # View 2: saturating Michaelis-Menten P = Pinf * t / (t + tau), grid on tau.
    best = None
    for tau in (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0):
        pinfs = []
        for t, p in zip(times, ps):
            h = max(0.5, hours_since_launch(t))
            pinfs.append(p * (h + tau) / h)
        pinf = median(pinfs)
        err = sum((math.log(max(1.0, p)) - math.log(max(1.0, pinf * max(0.5, hours_since_launch(t)) / (max(0.5, hours_since_launch(t)) + tau)))) ** 2
                  for t, p in zip(times, ps))
        if best is None or err < best[0]:
            best = (err, pinf, tau)
    _, pinf, tau = best
    h_t = max(0.5, hours_since_launch(target_t))
    views.append(math.log(max(1.0, pinf * h_t / (h_t + tau))))

    # View 3: last growth rate, capped and decayed (launch spikes slow down).
    # If the last interval is long, its average rate overstates the current one.
    dt_last = (times[-1] - times[-2]).total_seconds() / 3600.0
    if dt_last > 0:
        g = (math.log(max(1.0, ps[-1])) - math.log(max(1.0, ps[-2]))) / dt_last
        g = min(max(g, 0.0), 0.03)
        if dt_last > 12.0:
            g *= 0.5
        # Integrate g * exp(-u / 72h) over the horizon.
        grown = g * 72.0 * (1.0 - math.exp(-dt_h / 72.0))
        views.append(math.log(max(1.0, last_p)) + grown)

    if not views:
        return last_p, 0.10
    # Median of the views in log space: robust when one view extrapolates badly.
    mean_log = median(views)
    if len(views) > 1:
        var = sum((v - mean_log) ** 2 for v in views) / len(views)
        sigma = math.sqrt(var)
    else:
        sigma = 0.15
    sigma = min(max(sigma, 0.10), 0.50)
    return math.exp(mean_log), sigma


def structural_fit(transitions):
    """Fit (hazard h, cohort rate c) for one trophy over its transitions.

    Each transition: dE = h * (P_prev - E_prev) * dt + c * dP.
    Grid search, both parameters >= 0.  Returns (h, c) or None.
    """
    usable = []
    for (dt, dp, de, pool) in transitions:
        if dt > 0 and dp > 0:
            usable.append((dt, dp, de, pool))
    if len(usable) < 3:
        return None
    h_grid = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    c_grid = [0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
    best = None
    for h in h_grid:
        for c in c_grid:
            err = 0.0
            for dt, dp, de, pool in usable:
                pred = h * pool * dt + c * dp
                err += (pred - de) ** 2
            if best is None or err < best[0]:
                best = (err, h, c)
    return best[1], best[2]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    target_t = parse_time(case["target"])
    states = extract_states(case)
    last = states[-1]
    last_t = last[0]
    last_rates = last[3]
    last_p = float(last[1])
    horizon_days = max(0.05, case["horizon_minutes"] / 1440.0)
    dt_target_h = max(0.0, (target_t - last_t).total_seconds() / 3600.0)
    x_last = log_time(last_t)
    x_target = log_time(target_t)

    p_forecast, sigma_p = forecast_players(states, target_t)

    # Per-state earners per trophy (rate * players / 100).
    state_times = [s[0] for s in states]
    state_ps = [float(s[1]) for s in states]
    state_xs = [log_time(t) for t in state_times]
    earners = []  # earners[i][k]
    for i in range(n):
        col = []
        for k, s in enumerate(states):
            r = s[3][i]
            col.append(None if r is None else r * state_ps[k] / 100.0)
        earners.append(col)

    # Transitions for the structural fit: (dt_h, dP, dE_i, pool_i).
    transitions_per_trophy = []
    for i in range(n):
        trs = []
        for k in range(1, len(states)):
            e0, e1 = earners[i][k - 1], earners[i][k]
            if e0 is None or e1 is None:
                continue
            dt = (state_times[k] - state_times[k - 1]).total_seconds() / 3600.0
            dp = state_ps[k] - state_ps[k - 1]
            trs.append((dt, dp, e1 - e0, max(0.0, state_ps[k - 1] - e0)))
        transitions_per_trophy.append(trs)

    # Trend fits per trophy: logit(rate) ~ a + b * x.  Slopes are damped toward
    # zero (persistence) when states are few: a two-state slope mostly measures
    # one big refresh, not the going trend.  (Shrinking toward a tier median
    # was tried and rejected: risers and fallers share tiers, so the median
    # drags rising trophies down and falling ones up.)
    tier_of = [t.get("tier", "bronze") for t in trophies]
    betas = {}
    resid_sd = {}
    for i in range(n):
        xs, ys = [], []
        for k, s in enumerate(states):
            r = s[3][i]
            if r is None:
                continue
            xs.append(state_xs[k])
            ys.append(logit(r))
        if len(xs) >= 2 and max(xs) - min(xs) > 1e-9:
            b, a = least_squares(xs, ys)
            betas[i] = b
            res = [y - (a + b * x) for x, y in zip(xs, ys)]
            resid_sd[i] = math.sqrt(sum(v * v for v in res) / len(res)) if res else 0.15
        else:
            betas[i] = 0.0
            resid_sd[i] = 0.15

    n_states = len(states)
    shrink_w = n_states / (n_states + 3.0)  # weight on the trophy's own fit
    # Decay of the measured earners-growth rate over the forecast horizon:
    # launch-window growth mixes a flooding cohort inflow with incumbents
    # progressing, and both slow down; 1 / (1 + dt/48h) halves it in two days.
    growth_decay = 1.0 / (1.0 + dt_target_h / 48.0)

    medians = []
    sigmas = []
    for i in range(n):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.01
        tier = tier_of[i]
        e_last = earners[i][-1] if earners[i][-1] is not None else r_last * last_p / 100.0

        # --- trend view (anchored at the last observed rate) ---
        beta = shrink_w * betas[i]
        beta = min(max(beta, -1.5), 1.5)
        trend_logit = logit(r_last) + beta * (x_target - x_last)

        # --- structural view (refresh inversion) ---
        struct_logit = None
        sc = structural_fit(transitions_per_trophy[i])
        if sc is not None:
            h, c = sc
            pool = max(0.0, last_p - e_last)
            e_fc = e_last + pool * (1.0 - math.exp(-h * dt_target_h)) + c * max(0.0, p_forecast - last_p)
            e_fc = min(max(e_fc, 0.0), p_forecast)
            struct_logit = logit(100.0 * e_fc / p_forecast) if p_forecast > 0 else trend_logit

        # --- count view (earners growth in count space, all trophies) ---
        # The rate trend above is blind to the regime change: when player growth
        # slows, earners keep growing and rates rise even though the rate trend
        # was flat.  The count view grows earners at the trophy's own measured
        # per-hour rate (decayed over the horizon) and divides by the player
        # forecast.  No tier shrinkage: growth rates span orders of magnitude
        # across trophies, so shrinking toward a tier median crushes the fast
        # growers that matter most.
        count_logit = None
        col = [(state_times[k], e) for k, e in enumerate(earners[i]) if e is not None]
        if len(col) >= 2:
            span_h = (col[-1][0] - col[0][0]).total_seconds() / 3600.0
            g_own = (col[-1][1] - col[0][1]) / span_h if span_h > 0 else 0.0
            g_fwd = max(0.0, g_own) * growth_decay
            e_fc = e_last + g_fwd * dt_target_h
            e_fc = min(max(e_fc, 0.0), p_forecast)
            count_logit = logit(100.0 * e_fc / p_forecast) if p_forecast > 0 else trend_logit

        # --- blend ---
        parts = [trend_logit, count_logit] if count_logit is not None else [trend_logit]
        if struct_logit is not None:
            parts.append(struct_logit)
        med_logit = sum(parts) / len(parts)

        # Near-100% trophies stay near 100%: everyone holds them, newcomers too.
        if r_last >= 99.99:
            med_logit = logit(99.99)

        medians.append(inv_logit(med_logit))
        sig = math.sqrt(sigma_p ** 2 + min(max(resid_sd[i], 0.12), 0.6) ** 2
                        + (0.25 * math.sqrt(horizon_days)) ** 2)
        if r_last >= 99.99:
            sig = 0.05
        sigmas.append(min(max(sig, 0.15), 1.2))

    # --- count consistency: story chain + platinum cap, per quantile level ---
    # Chain order from descriptions: Chapter 1..N, then the final mission, then
    # the platinum (needs every trophy, so it caps everything).
    chain_idx = []
    for i, t in enumerate(trophies):
        desc = (t.get("description") or "") + " " + (t.get("name") or "")
        m = re.search(r"Chapter (\d+)", desc, re.IGNORECASE)
        if m:
            chain_idx.append((int(m.group(1)), i))
    chain_idx.sort()
    chain = [i for _, i in chain_idx]
    final_idx = None
    for i, t in enumerate(trophies):
        desc = ((t.get("description") or "") + " " + (t.get("name") or "")).lower()
        if "final mission" in desc and tier_of[i] in ("gold", "silver"):
            final_idx = i
            break
    if final_idx is not None and final_idx not in chain:
        chain.append(final_idx)
    plat_idx = None
    for i, t in enumerate(trophies):
        if tier_of[i] == "platinum":
            plat_idx = i
            break

    # Build quantile rows.
    q_rows = []
    for i in range(n):
        lg = logit(medians[i])
        row = [inv_logit(lg + z * sigmas[i]) for z in Z]
        q_rows.append(row)
    if len(chain) >= 2:
        for lvl in range(5):
            vals = [math.log(max(1e-9, q_rows[i][lvl])) for i in chain]
            fixed = isotonic_decreasing(vals)
            for pos, i in enumerate(chain):
                q_rows[i][lvl] = math.exp(fixed[pos])
    if plat_idx is not None:
        for lvl in range(5):
            cap = min(q_rows[i][lvl] for i in range(n) if i != plat_idx)
            q_rows[plat_idx][lvl] = min(q_rows[plat_idx][lvl], cap)

    rows = []
    for i, t in enumerate(trophies):
        q = [round4(min(100.0, max(0.0, v))) for v in q_rows[i]]
        # Keep the row non-decreasing after rounding.
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
