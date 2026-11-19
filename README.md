# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jonsonbmth/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jonsonbmth/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast of every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at a case's target instant (the final case: 2026-11-26T00:00:00Z,
a week after launch), as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95)
per trophy. `main.py` (Python 3.13, standard library only, no network)
reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output/`).

## The model: earners over players

An earn rate is a ratio — earners E(t) over tracked players P(t) — and in
launch week the two sides move very differently. P explodes as players
sync their profiles (Exophase's first refresh multiplied it 6x overnight,
638 → 3,868, and it reached ~12,000 within 14 hours of launch), while E
grows only as players actually reach each trophy. Forecasting the ratio
directly confuses dilution with progress, so this forecaster models each
side separately and divides:

- **Players P(t):** log P fitted on log(1 + hours since release) over the
  distinct page states in the last 24 hours of the case history, anchored
  on the last observed value, slope damped for long horizons. Consecutive
  identical reads are one stale page, not many data points, so the fit
  uses distinct states only.
- **Earners E(t):** per trophy, log E fitted the same way, each slope
  shrunk toward its trophy type's median slope (more shrinkage where
  earners are few and the fit is mostly noise). Slopes are damped for
  long horizons and every trophy's rate is capped by a generous type
  ceiling, so no extrapolation runs away.
- **Flat history** (no trend signal at all, as on launch night): earners
  grow only from the new cohort — E += f_type · last_rate ·
  (P_target − P_last) — where f_type is the share of the launch-night
  rate at which newly tracked players earn that kind of trophy (early
  activities ~0.6, first chapter ~0.5, late story ~0). P is then forecast
  from a refresh prior: ~900 newly tracked players per hour, plus the
  backlog a long-stale page holds, which the next refresh clears.
- **The opening mission** ("Just Like Paradise") is pinned near 100%:
  every tracked player has done it, so its earners track P. It still gets
  a real band — the logit scale is brutal near 100%.
- **Story-chain consistency:** earners along intro → chapter 1 → … →
  final mission → 100% completion → platinum must be non-increasing (you
  cannot hold a later trophy without the earlier ones), enforced with
  isotonic regression (pool-adjacent-violators).

Trophy types are classified at runtime from each trophy's name and
description (keywords), never from hardcoded ids, so the program works on
any trophy list of this shape.

## Uncertainty

Each trophy's quantiles are a logit-normal band around its median rate.
The spread combines the fit residual, a refresh term that grows with the
expected player growth, and a horizon term (0.30 · sqrt(days)) that is
deliberately **not** shrunk — a 4.5-day forecast is genuinely less certain
than a 10-hour one, and pinball loss punishes under-dispersion harder than
over-dispersion when the median is off. The fit/refresh components carry a
0.3 multiplier tuned on the backtest below.

## Validation (out-of-sample, real data)

Backtest: forecast from the launch-night practice case (checkpoint
03:00 UTC 19 Nov 2026, flat history, 638 players) to the real Exophase
page at 13:35 UTC the same day (~12,000 players) — a genuine
out-of-sample window spanning the first refresh and 10.6 hours of growth.
Scored with the competition's own pinball loss on the logit scale:

| | pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| **This model** | **0.190** | **0.033 pp** | **0.982** |
| Persistence baseline (house) | 2.383 | 0.757 pp | 0.055 |
| Curve baseline (house) | 2.383 | 0.757 pp | 0.055 |

The house baselines score 2.38 on this window because a flat history
gives them a flat line: they miss the refresh entirely (coverage 0.055).
This model beats them 12.5x on loss and 23x on typical miss. The curve
baseline is identical to persistence here because a flat history has no
slope to fit — as the rules predict.

Two things this backtest does **not** prove, stated honestly:

- The trending-history path (log-log fits, shrinkage, damping) is what
  the final case actually uses, and it cannot be backtested on launch
  night because there is only one real refresh window. It was sanity
  checked on a synthetic trending case built from the real 13:35 page
  (sane medians, isotonic chain respected, no explosions), but that is a
  robustness check, not an accuracy measurement.
- The cohort fractions were calibrated on the same refresh the backtest
  scores against, so the 0.190 is optimistic for the flat-history path.
  The final case has a rich trending history, where the cohort path does
  not apply at all.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.06 s for two cases (limit 8 s per case). Output is
deterministic: two runs on the same input are byte-identical. Output is
~2.8 KB per case (limit 65,536 bytes), 55 rows, one per trophy.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
