# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vasm-696/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vasm-696/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour. The entry is a
single Python program (`main.py`, standard library only) that reads one
case file and writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for each
of hours 2..24 of the release day, anchored on the case's hour-1 print.

Method: an ensemble of two log-ratio estimators, fitted at run time from
the comparables the case file itself carries (38 in a practice case, 42
in the final). One is an OLS regression of log(viewers_h / viewers_1) on
log(viewers_1) across past launches, with empirical residual quantiles;
log(hour 1) is capped at the comparables' range so a record hour-1 print
cannot extrapolate a collapse. The other is the 10 comparables whose
hour-1 print is nearest in log space, with empirical quantiles of their
log-ratios. The two forecasts are averaged in log space. Leave-one-out
over the 37 usable comparables: mean pinball loss on the log scale 0.207,
against 0.357 for plain median ratios and 0.465 for the house baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9tb8shvj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
