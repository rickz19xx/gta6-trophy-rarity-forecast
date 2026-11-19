# GTA VI trophy earn-rate forecaster

Entry for the make.fun bounty `gta6-trophy-rarity-forecast-01`.

Reads a case (`/input/<id>.in`) holding the house's Exophase earn-rate reads for the
55 GTA VI PS5 trophies up to the cut, and writes `/output/<id>.out`: five quantiles
(5/25/50/75/95%) of each trophy's earn rate at the target, in percent.

## Method
Decomposes each rate into earners / players-tracked. Refreshes of Exophase's player
pool jump the denominator and dilute every rate, so the forecaster fits the numerator
(log-linear, damped) and the denominator separately on refresh-states (plateau reads
collapsed), then recombines at the target. Quantile spread comes from fit residual,
horizon, denominator uncertainty and rounding, on the logit scale. Pure standard
library, deterministic, offline.

## Run
`python3 main.py` reads every `/input/*.in`, writes the matching `/output/*.out`.

MIT License.
