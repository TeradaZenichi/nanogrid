# Expose a minimal, TensorFlow-free API surface.
# ForecastMPC (LSTM, needs TF) is imported explicitly from
# forecasting.get_forecasting to keep this package import light.
from .load_forecast import load
from .pv_forecast import pv
from .prototype_forecast import HybridForecast, PerfectForecast, PrototypeForecast

__all__ = ["load", "pv", "PrototypeForecast", "PerfectForecast", "HybridForecast"]
