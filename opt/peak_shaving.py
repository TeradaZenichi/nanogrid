"""Fixed-threshold peak-shaving operational baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import BaselineObservation, bounded_power, validate_reserve_fraction
from .closed_loop import ControlAction


@dataclass(frozen=True)
class PeakShavingController:
    """Limit net grid demand and recharge only from instantaneous PV surplus."""

    import_limit_kw: float
    reserve_fraction: float = 0.0
    name = "peak_shaving"
    version = "fixed-threshold-peak-shaving-v1"

    def __post_init__(self) -> None:
        limit = float(self.import_limit_kw)
        if limit < 0.0:
            raise ValueError("import_limit_kw must be nonnegative")
        object.__setattr__(self, "import_limit_kw", limit)
        object.__setattr__(self, "reserve_fraction", validate_reserve_fraction(self.reserve_fraction))

    def action(self, observation: BaselineObservation) -> ControlAction:
        net_load = observation.net_load_kw
        reserve = observation.reserve_energy_kwh(self.reserve_fraction)
        if net_load > self.import_limit_kw and observation.energy_kwh > reserve + 1e-12:
            request = net_load - self.import_limit_kw
            return ControlAction(p_bess_kw=bounded_power(request, observation.power_max_kw))
        if net_load < 0.0 and observation.energy_kwh < observation.energy_max_kwh - 1e-12:
            return ControlAction(p_bess_kw=bounded_power(net_load, observation.power_max_kw))
        return ControlAction()

    def configuration(self) -> dict:
        return {
            "import_limit_kw": self.import_limit_kw,
            "reserve_fraction": self.reserve_fraction,
            "recharge_source": "instantaneous PV surplus only",
        }


__all__ = ["PeakShavingController"]
