"""Offline forecast evaluation: prototypes vs naive baselines vs LSTM.

Outputs go to Results/forecasting/. Run from the repo root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.evaluate_prototype import run_evaluation

DAYS = 365
EVERY_MIN = 60
WITH_LSTM = True

if __name__ == "__main__":
    run_evaluation(days=DAYS, every_min=EVERY_MIN, with_lstm=WITH_LSTM)
