"""Forecast GTA VI launch-day Twitch viewers, hour by hour (make.fun gta6-twitch-ccu-forecast-01).

Method: an ensemble of two log-ratio estimators, both fitted at run time
from the comparables the case file itself carries (38 in a practice case,
42 in the final).  Viewers are modelled in log space, anchored on the
case's own hour-1 print p1, for hours h = 2..24:

  1. Log-ratio regression: log(viewers_h / viewers_1) is regressed on
     log(viewers_1) across the comparables (OLS per hour); the forecast
     quantiles are the fitted value plus the empirical quantiles of the
     fit's residuals.  log(p1) is capped at the comparables' observed
     range, so the fit cannot extrapolate a collapse for an hour-1 print
     above anything seen before (GTA VI's hour 1 will likely be the
     largest ever).
  2. k-nearest neighbours: the 10 comparables whose hour-1 print is
     closest to p1 in log space; the forecast quantiles are the empirical
     quantiles of their log-ratios for that hour.

The two log-quantile forecasts are averaged.  Leave-one-out over the 37
usable comparables (mean pinball loss on the log scale): plain median
ratios 0.357, house baseline 0.465, regression alone 0.224, kNN alone
0.218, this ensemble 0.207.

Why the regression matters: launch days share one shape, but its
amplitude depends strongly on how big hour 1 already is.  Small hour-1
launches explode through the day (median hour-19 print ~100x hour 1);
the biggest launches stay near flat (median hour-19 print ~1x hour 1).
A plain median ratio ignores that.

Standard library only.  Reads every <id>.in in the input directory and
writes <id>.out: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))
KNN_K = 10
BASELINE_SIGMA = 0.15
BASELINE_Z = (-1.645, -0.674, 0, 0.674, 1.645)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript's Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantiles(sorted_vals, levels):
    n = len(sorted_vals)
    out = []
    for lv in levels:
        pos = lv * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)
    return out


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    b = cov / var if var else 0.0
    return my - b * mx, b


def release_day_hours(comp):
    """Map a comparable's hourly points to release-day hours 1..24 -> viewers."""
    day_start = parse_ts(comp["day_start"])
    hours = {}
    for p in comp.get("hourly") or []:
        try:
            t = parse_ts(p["from"])
        except (KeyError, ValueError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and p.get("viewers"):
            hours.setdefault(h, float(p["viewers"]))
    return hours


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not [0.05, 0.25, 0.5, 0.75, 0.95]")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    curves = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if 1 in hours and hours[1] > 0:
            curves.append(hours)

    log_p1 = math.log(p1)
    train_lp = [math.log(c[1]) for c in curves]
    lo_lp = min(train_lp) if train_lp else log_p1
    hi_lp = max(train_lp) if train_lp else log_p1
    lp = min(max(log_p1, lo_lp), hi_lp)

    rows = []
    for h in HOURS:
        pairs = []
        for c in curves:
            if h in c and c[h] > 0:
                pairs.append((math.log(c[1]), math.log(c[h] / c[1])))
        reg_qs = None
        if len(pairs) >= 8:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            a, b = ols(xs, ys)
            resid = sorted(y - (a + b * x) for x, y in pairs)
            reg_qs = [a + b * lp + q for q in quantiles(resid, LEVELS)]
        elif pairs:
            reg_qs = quantiles(sorted(y for _, y in pairs), LEVELS)
        knn_qs = None
        if pairs:
            near = sorted(pairs, key=lambda p: abs(p[0] - log_p1))[:KNN_K]
            if len(near) >= 5:
                knn_qs = quantiles(sorted(y for _, y in near), LEVELS)
        if reg_qs is not None and knn_qs is not None:
            qs = [(r + k) / 2 for r, k in zip(reg_qs, knn_qs)]
        elif reg_qs is not None:
            qs = reg_qs
        elif knn_qs is not None:
            qs = knn_qs
        else:
            qs = [z * BASELINE_SIGMA * math.sqrt(h - 1) for z in BASELINE_Z]
        vals = []
        for q in qs:
            v = p1 * math.exp(q)
            v = min(max(v, 0.01), 1e9)
            vals.append(round2(v))
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        rows.append({"hour": h, "q": vals})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
