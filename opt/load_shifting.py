"""Time-of-use load-shifting operational baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import BaselineObservation, bounded_power, validate_reserve_fraction
from .closed_loop import ControlAction


@dataclass(frozen=True)
class LoadShiftingController:
    """Charge at the lowest tariff and serve local load at the highest tariff."""

    offpeak_tariff: float
    peak_tariff: float
    reserve_fraction: float = 0.0
    name = "load_shifting"
    version = "tou-load-shifting-v1"

    def __post_init__(self) -> None:
        offpeak = float(self.offpeak_tariff)
        peak = float(self.peak_tariff)
        if offpeak >= peak:
            raise ValueError("offpeak_tariff must be lower than peak_tariff")
        object.__setattr__(self, "offpeak_tariff", offpeak)
        object.__setattr__(self, "peak_tariff", peak)
        object.__setattr__(self, "reserve_fraction", validate_reserve_fraction(self.reserve_fraction))

    def action(self, observation: BaselineObservation) -> ControlAction:
        tariff = observation.tariff_per_kwh
        if tariff <= self.offpeak_tariff + 1e-12:
            if observation.energy_kwh >= observation.energy_max_kwh - 1e-12:
                return ControlAction()
            return ControlAction(p_bess_kw=-observation.power_max_kw)

        reserve = observation.reserve_energy_kwh(self.reserve_fraction)
        if tariff >= self.peak_tariff - 1e-12 and observation.energy_kwh > reserve + 1e-12:
            request = min(max(observation.net_load_kw, 0.0), observation.power_max_kw)
            return ControlAction(p_bess_kw=bounded_power(request, observation.power_max_kw))
        return ControlAction()

    def configuration(self) -> dict:
        return {
            "offpeak_tariff": self.offpeak_tariff,
            "peak_tariff": self.peak_tariff,
            "reserve_fraction": self.reserve_fraction,
        }


__all__ = ["LoadShiftingController"]
