"""Causal timestamp rules for left-labelled hourly forecasts.

An hourly row contains samples from its label through 55 minutes later, so it
becomes observable only after the hour has ended.
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
    return pd.Timestamp(anchor_hour).floor("h") + pd.Timedelta(hours=1, minutes=-sample_minutes)
