"""Causal timestamp rules shared by the hourly forecasting models.

The hourly LSTM models were trained with left-labelled hourly aggregates: the
row labelled 11:00 contains the 5-min observations from 11:00 through 11:55.
Consequently, at an arbitrary decision time only rows whose complete source
window has already elapsed may enter the model.
"""

from __future__ import annotations

import pandas as pd


FORECAST_ASSEMBLY_VERSION = "causal-completed-hour-v1"


def last_completed_hour(t_now) -> pd.Timestamp:
    """Return the label of the latest fully observed left-labelled hour."""
    return pd.Timestamp(t_now).floor("h") - pd.Timedelta(hours=1)


def hourly_forecast_start(anchor_hour) -> pd.Timestamp:
    """Return the timestamp represented by the first hourly model output."""
    return pd.Timestamp(anchor_hour).floor("h") + pd.Timedelta(hours=1)


def hourly_source_end(anchor_hour, sample_minutes: int = 5) -> pd.Timestamp:
    """Return the final native timestamp used by an hourly aggregate row."""
    sample_minutes = int(sample_minutes)
    if sample_minutes <= 0 or 60 % sample_minutes:
        raise ValueError("sample_minutes must be a positive divisor of 60")
    return (
        pd.Timestamp(anchor_hour).floor("h")
        + pd.Timedelta(hours=1)
        - pd.Timedelta(minutes=sample_minutes)
    )
