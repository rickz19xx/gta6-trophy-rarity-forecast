"""gta6-trophy-rarity-forecast-01 entry: dilution-aware earners/players forecaster.

Reads each <id>.in case, forecasts every trophy's earn rate at the target as five
quantiles on the logit scale, writes <id>.out. Standard library only, deterministic.

The model forecasts the numerator (earners) and denominator (players tracked)
separately, because an Exophase refresh jumps the player count and dilutes almost
every rate; extrapolating the rate alone mistakes that dilution for a trend.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = model.forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
