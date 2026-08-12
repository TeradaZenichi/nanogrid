import unittest

import numpy as np
import pandas as pd

from forecasting.causal import (
    FORECAST_ASSEMBLY_VERSION,
    hourly_forecast_start,
    hourly_source_end,
    last_completed_hour,
)
from forecasting.load_forecast import _add_cyc_hourly, _build_st_input


class ForecastCausalityTests(unittest.TestCase):
    def test_hourly_anchor_is_complete_at_every_control_minute(self):
        for minute in range(0, 60, 5):
            now = pd.Timestamp(2009, 5, 1, 12, minute)
            anchor = last_completed_hour(now)
            self.assertEqual(anchor, pd.Timestamp("2009-05-01 11:00:00"))
            self.assertLessEqual(hourly_source_end(anchor), now)
            self.assertEqual(
                hourly_forecast_start(anchor),
                pd.Timestamp("2009-05-01 12:00:00"),
            )

    def test_future_perturbations_do_not_change_hourly_lstm_input(self):
        index = pd.date_range("2009-05-01 00:00:00", periods=13 * 12, freq="5min")
        now = pd.Timestamp("2009-05-01 12:05:00")
        observed = pd.DataFrame({"p_norm": np.arange(len(index), dtype=float)}, index=index)
        perturbed = observed.copy()
        perturbed.loc[perturbed.index > now, "p_norm"] = 1e9

        hourly_observed = _add_cyc_hourly(observed.resample("h").mean())
        hourly_perturbed = _add_cyc_hourly(perturbed.resample("h").mean())
        anchor = last_completed_hour(now)
        x_observed = _build_st_input(hourly_observed, anchor, lookback_steps=8)
        x_perturbed = _build_st_input(hourly_perturbed, anchor, lookback_steps=8)

        np.testing.assert_array_equal(x_observed, x_perturbed)
        self.assertEqual(FORECAST_ASSEMBLY_VERSION, "causal-completed-hour-v1")

    def test_invalid_native_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            hourly_source_end("2009-05-01 11:00:00", sample_minutes=7)


if __name__ == "__main__":
    unittest.main()
