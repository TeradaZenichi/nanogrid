"""Evaluate the forecasting models offline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.evaluate_prototype import run_evaluation
from opt.utils import load_sized_parameters

DAYS = 365
EVERY_MIN = 60
WITH_LSTM = True

def main() -> None:
    params = load_sized_parameters()
    run_evaluation(days=DAYS, every_min=EVERY_MIN, with_lstm=WITH_LSTM, params=params)


if __name__ == "__main__":
    main()
