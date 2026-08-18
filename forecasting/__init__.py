"""TensorFlow-free forecasting interface."""

from .load_forecast import load
from .pv_forecast import pv
from .prototype_forecast import PerfectForecast, PrototypeForecast

__all__ = ["load", "pv", "PrototypeForecast", "PerfectForecast"]
