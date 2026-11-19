# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rickz19xx/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rickz19xx/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one. Reads the house's
Exophase earn-rate history for the 55 PS5 trophies up to the cut and writes five
quantiles (5/25/50/75/95%) of each trophy's earn rate at the target. The forecaster
decomposes each rate into earners / players-tracked, fits numerator and denominator
separately on refresh-states (Exophase pool refreshes dilute every rate), and
recombines at the target with an honest logit-scale spread. Pure standard library,
deterministic, offline.

## Run it

```sh
python3 main.py
```

Reads every `/input/*.in`, writes the matching `/output/*.out`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
