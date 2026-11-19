#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI PlayStation 5 trophy, the earn rate on Exophase
at a case's target instant, as five quantiles (0.05/0.25/0.5/0.75/0.95) of
the rate in percent. Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (else
/input/) and writes one ``<id>.out`` per case to $MAKEFUN_OUTPUT_DIR (else
/output/). Standard library only, no network, deterministic.

Model: earners-over-players decomposition
-----------------------------------------
An earn rate is a ratio: earners E(t) over tracked players P(t). The two
move very differently in launch week -- P explodes as players sync their
profiles (a refresh can multiply it overnight), while E grows only as
players actually reach each trophy. Forecasting the ratio directly confuses
the two; this forecaster models each side separately and divides.

* Players P(t): log P fitted on log(1 + hours since release) over the
  distinct page states in the last 24 hours of the case history, anchored
  on the last observed value, with the slope damped for long horizons.
  Consecutive identical reads are one stale page, not many data points,
  so the fit uses distinct states only.
* Earners E(t): per trophy, log E fitted the same way, with each slope
  shrunk toward its trophy type's median slope (more shrinkage for
  trophies with few earners, where the fit is mostly noise). Slopes are
  damped for long horizons and each trophy's rate is capped by a generous
  type ceiling, so no extrapolation runs away.
* Flat history (no trend signal at all): earners grow only from the new
  cohort -- E += f_type * last_rate * (P_target - P_last) -- where
  f_type is the share of the launch-night rate at which newly tracked
  players earn that kind of trophy (early activities ~0.6, first chapter
  ~0.5, late story ~0). P is then forecast from a refresh prior: about
  900 newly tracked players per hour, plus the backlog a long-stale page
  holds, which the next refresh clears.
* The opening mission's trophy is pinned near 100%: every tracked player
  has done it, so its earners track P. The logit scale is brutal near
  100%, so it still gets a real (if narrow) band.
* Story-chain consistency: earners along intro -> chapter 1 -> ... ->
  final mission -> 100% completion -> platinum must be non-increasing
  (you cannot hold a later trophy without the earlier ones), enforced
  with isotonic regression (pool-adjacent-violators).

Uncertainty: each trophy's quantiles are a logit-normal band around its
median rate. The spread combines the fit residual, a refresh term that
grows with the expected player growth, and a horizon term (0.30 *
sqrt(days)) that is deliberately NOT shrunk -- a 4.5-day forecast is
genuinely less certain than a 10-hour one, and pinball loss punishes
under-dispersion harder than over-dispersion when the median is off.
The fit/refresh components carry a 0.3 multiplier tuned on a genuine
out-of-sample backtest (see README).

Trophy types are classified at runtime from each trophy's name and
description (keywords), never from hardcoded ids, so the program works
on any trophy list of this shape.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from statistics import median

# Quantile levels (from the case) and their standard-normal z-scores.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Fallback release instant if the case carries no usable release_date.
FALLBACK_RELEASE = "2026-11-19T00:00:00Z"

# Generous per-type rate ceilings (percent) -- guards against runaway
# extrapolation, never binding for a sane forecast.
CEILING = {
    "intro": 99.99, "platinum": 2.0, "complete100": 3.0,
    "story_ch1": 60.0, "story_ch2": 40.0, "story_ch3": 25.0,
    "story_ch4": 15.0, "story_final": 10.0, "story_mission": 20.0,
    "strand": 10.0, "collection": 15.0, "collection_early": 20.0,
    "early_fun": 30.0, "grind": 15.0, "misc": 20.0,
}

# New-cohort earn fraction by type, for flat histories only: the share of
# the launch-night rate at which newly tracked players earn a trophy of
# this kind during the horizon. Calibrated on the launch-night refresh
# (638 -> ~12,000 tracked players on 19 Nov 2026): early activities and
# the first chapter are earned quickly by newcomers, late-story trophies
# and grinds essentially not at all in the first hours.
COHORT_F = {
    "intro": 1.0, "platinum": 0.0, "complete100": 0.0,
    "story_ch1": 0.5, "story_ch2": 0.05, "story_ch3": 0.0, "story_ch4": 0.0,
    "story_final": 0.0, "story_mission": 0.02, "strand": 0.03,
    "collection": 0.01, "collection_early": 0.15, "early_fun": 0.6,
    "grind": 0.02, "misc": 0.05,
}
# The earliest story mission (lowest id) is reachable in the first hours;
# later missions are not, so it gets a much higher cohort fraction.
STORY_MISSION_FIRST_F = 0.2

# Story chain order for the isotonic constraint (earners non-increasing).
CHAIN = ["intro", "story_ch1", "story_ch2", "story_ch3", "story_ch4",
         "story_final", "complete100", "platinum"]

# Newly tracked players per hour, for the flat-history refresh prior.
REFRESH_RATE_PER_HOUR = 900.0
# The stale backlog the next refresh can clear, capped at 6 hours' worth.
MAX_BACKLOG_HOURS = 6.0
# Backtest-tuned multiplier on the fit/refresh spread components.
SPREAD_MULT = 0.3
# Horizon spread coefficient (logit sigma per sqrt(day)), unshrunk.
HORIZON_SPREAD = 0.30
# Intro trophy's logit spread (pinned near 100%, but not a point mass).
INTRO_SPREAD = 0.6


def parse_time(s):
    """Parse an ISO-8601 timestamp to epoch seconds. A timestamp with no
    timezone (e.g. a bare release date) is read as UTC, never local time,
    so the result does not depend on the container's timezone."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def logit(p):
    p = min(99.99, max(0.01, p)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (intercept, slope, resid std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        s = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        s = 0.0
    return a, b, s


def classify(desc, name):
    """Trophy type from its name and description (keywords, no ids)."""
    text = (name + " " + desc).lower()
    if "gone fishing" in text:
        return "intro"
    if "platinum" in text or "leonida legend" in text:
        return "platinum"
    if "100% completion" in text:
        return "complete100"
    m = re.search(r"chapter (\d)", text)
    if m:
        return "story_ch%d" % int(m.group(1))
    if "complete the final mission for" in text:
        return "strand"
    if "complete the final mission" in text:
        return "story_final"
    if "complete" in text and ("heist" in text or "run" in text
                               or "play" in text or "headliners" in text
                               or "takedown" in text or "operation" in text):
        return "story_mission"
    if "fishing challenge" in text or "arcade game" in text:
        return "early_fun"
    if "discover" in text:
        return "collection_early"
    if any(k in text for k in ["collectible card", "waink", "meditate",
                               "explore all", "every spot", "every location"]):
        return "collection"
    if any(k in text for k in ["250 stunts", "50 challenges",
                               "20 different races", "earn $", "spend $",
                               "purchase all", "gold medal"]):
        return "grind"
    if "strangers" in text:
        return "strand"
    return "misc"


def pava_decreasing(values):
    """Isotonic regression (non-increasing) by pool-adjacent-violators."""
    stack_v, stack_w = [], []
    for v in values:
        cv, cw = v, 1.0
        while stack_v and stack_v[-1] < cv:
            pv, pw = stack_v.pop(), stack_w.pop()
            cv = (cv * cw + pv * pw) / (cw + pw)
            cw += pw
        stack_v.append(cv)
        stack_w.append(cw)
    out = []
    for v, w in zip(stack_v, stack_w):
        out.extend([v] * int(round(w)))
    return out


def distinct_reads(hist):
    """Keep reads where the page state changed (plus the first). A stale
    page contributes one state, however many reads reprint it."""
    out = [hist[0]]
    for h in hist[1:]:
        prev = out[-1]
        if (h.get("players_tracked") != prev.get("players_tracked")
                or h.get("rates") != prev.get("rates")):
            out.append(h)
    return out


def forecast_case(case):
    """Forecast one case; returns {trophy_id: [q05, q25, q50, q75, q95]}."""
    hist = case["history"]
    trophies = case["trophies"]
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, (target - at) / 86400.0)
    horizon_hours = max(0.0, (target - at) / 3600.0)

    release = FALLBACK_RELEASE
    game = case.get("game") or {}
    if isinstance(game.get("release_date"), str):
        release = game["release_date"]
    try:
        release_epoch = parse_time(release)
    except (ValueError, TypeError):
        release_epoch = parse_time(FALLBACK_RELEASE)

    def x_of(epoch):
        return math.log1p(max(0.0, (epoch - release_epoch) / 3600.0))

    x_t = x_of(target)
    # Slope damping: extrapolations decay over long horizons.
    damp = 1.0 / (1.0 + 0.15 * max(0.0, horizon_days - 1.0))

    states = distinct_reads(hist)
    flat = len(states) <= 1
    times = [parse_time(h["at"]) for h in hist]
    last_change = times[0]
    for j in range(1, len(hist)):
        if (hist[j].get("players_tracked") != hist[j - 1].get("players_tracked")
                or hist[j].get("rates") != hist[j - 1].get("rates")):
            last_change = times[j]
    staleness_h = max(0.0, (at - last_change) / 3600.0)

    p_last = hist[-1]["players_tracked"]
    types = [classify(t.get("description", ""), t.get("name", ""))
             for t in trophies]

    # ---- players P(t) ----
    if flat:
        # No trend: refresh prior -- ~900 newly tracked players per hour,
        # plus the stale backlog the next refresh clears.
        p_tgt = min(p_last * 30.0,
                    p_last + REFRESH_RATE_PER_HOUR
                    * (horizon_hours + min(staleness_h, MAX_BACKLOG_HOURS)))
    else:
        win = [h for h in states if parse_time(h["at"]) >= at - 24 * 3600]
        if len(win) < 2:
            win = states
        xs = [x_of(parse_time(h["at"])) for h in win]
        ys = [math.log(max(1, h["players_tracked"])) for h in win]
        _, b, _ = ols(xs, ys)
        b = max(0.0, min(1.5, b))
        p_tgt = math.exp(ys[-1] + b * damp * (x_t - xs[-1]))
        p_tgt = max(float(p_last), p_tgt)
    p_log_growth = math.log(max(1.0, p_tgt / max(1, p_last)))

    # ---- earners E(t) ----
    last_rates = hist[-1]["rates"]
    last_e = []
    for i in range(len(trophies)):
        r = last_rates[i]
        last_e.append(max(0.05, (r if r is not None else 0.01)
                          * p_last / 100.0))

    sm_ids = [t["id"] for i, t in enumerate(trophies)
              if types[i] == "story_mission"]
    first_sm = min(sm_ids) if sm_ids else None

    e_tgt = []
    resid_s = []
    if flat:
        for i, t in enumerate(trophies):
            ty = types[i]
            if ty == "intro":
                e_tgt.append(1.0 * p_tgt)
                resid_s.append(0.0)
                continue
            r_last = last_rates[i] if last_rates[i] is not None else 0.01
            f = COHORT_F.get(ty, 0.05)
            if ty == "story_mission" and t["id"] == first_sm:
                f = STORY_MISSION_FIRST_F
            e = last_e[i] + f * (r_last / 100.0) * (p_tgt - p_last)
            cap_e = CEILING.get(ty, 20.0) / 100.0 * p_tgt
            e_tgt.append(max(0.05, min(e, cap_e)))
            resid_s.append(0.0)
    else:
        win = [h for h in states if parse_time(h["at"]) >= at - 24 * 3600]
        if len(win) < 2:
            win = states
        xs = [x_of(parse_time(h["at"])) for h in win]
        slopes, resids = [], []
        for i in range(len(trophies)):
            xs_i, ys_i = [], []
            for h, x in zip(win, xs):
                r = h["rates"][i]
                if r is None:
                    continue
                xs_i.append(x)
                ys_i.append(math.log(max(
                    0.05, r * h["players_tracked"] / 100.0)))
            if len(xs_i) < 2:
                slopes.append(0.0)
                resids.append(0.0)
                continue
            _, b, s = ols(xs_i, ys_i)
            slopes.append(b)
            resids.append(s)
        type_slopes = {}
        for ty in set(types):
            vals = [slopes[i] for i in range(len(trophies)) if types[i] == ty]
            type_slopes[ty] = median(vals) if vals else 0.0
        for i, t in enumerate(trophies):
            ty = types[i]
            if ty == "intro":
                e_tgt.append(1.0 * p_tgt)
                resid_s.append(0.0)
                continue
            # Shrink toward the type median: few earners => noisy fit.
            w = min(0.85, max(0.15,
                              math.log10(max(1.0, last_e[i]) + 1.0) / 2.0))
            b = w * slopes[i] + (1.0 - w) * type_slopes[ty]
            b = max(-0.5, min(2.5, b))
            e = math.exp(math.log(last_e[i]) + b * damp * (x_t - xs[-1]))
            cap_e = CEILING.get(ty, 20.0) / 100.0 * p_tgt
            e_tgt.append(max(0.05, min(e, cap_e)))
            resid_s.append(resids[i])

    # ---- isotonic: story-chain earners non-increasing ----
    chain_steps = []
    for ty in CHAIN:
        idxs = [i for i in range(len(trophies)) if types[i] == ty]
        if idxs:
            chain_steps.append(idxs)
    if len(chain_steps) >= 2:
        step_vals = [max(e_tgt[i] for i in idxs) for idxs in chain_steps]
        fixed = pava_decreasing(step_vals)
        for idxs, fv in zip(chain_steps, fixed):
            for i in idxs:
                if e_tgt[i] > fv:
                    e_tgt[i] = max(fv, 0.05)

    # ---- quantiles ----
    s_refresh = 0.35 * p_log_growth + (0.45 if flat else 0.0)
    s_horizon = HORIZON_SPREAD * math.sqrt(horizon_days)
    out = {}
    for i, t in enumerate(trophies):
        rate = 100.0 * e_tgt[i] / p_tgt
        rate = min(99.99, max(0.01, rate))
        if types[i] == "intro":
            s = INTRO_SPREAD
        else:
            s_fit = min(1.0, max(0.15, resid_s[i]))
            s = math.sqrt((s_fit ** 2 + s_refresh ** 2) * SPREAD_MULT ** 2
                          + s_horizon ** 2)
        l = logit(rate)
        qs = [round(min(100.0, max(0.0, inv_logit(l + z * s))), 4)
              for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out[t["id"]] = qs
    return out


def persistence_fallback(case):
    """Last-rate fallback (valid output even if the model fails)."""
    last_rates = case["history"][-1]["rates"]
    out = {}
    for t, r in zip(case["trophies"], last_rates):
        rate = min(99.99, max(0.01, r if r is not None else 0.01))
        l = logit(rate)
        qs = [round(min(100.0, max(0.0, inv_logit(l + z * 1.0))), 4)
              for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out[t["id"]] = qs
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(in_dir, name), encoding="utf-8") as f:
                case = json.load(f)
        except Exception:
            # Unreadable case file: nothing to forecast from. Skip it
            # rather than killing the other cases in this run.
            continue
        try:
            fc = forecast_case(case)
        except Exception:
            try:
                fc = persistence_fallback(case)
            except Exception:
                continue
        try:
            rows = [{"id": t["id"], "q": fc[t["id"]]}
                    for t in case["trophies"]]
        except Exception:
            continue
        payload = json.dumps({"trophies": rows}, separators=(",", ":"))
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
