"""Greedy self-consumption operational baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import BaselineObservation, bounded_power, validate_reserve_fraction
from .closed_loop import ControlAction


@dataclass(frozen=True)
class SelfConsumptionController:
    """Charge from PV surplus and discharge against the current local deficit."""

    reserve_fraction: float = 0.0
    name = "self_consumption"
    version = "greedy-self-consumption-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reserve_fraction", validate_reserve_fraction(self.reserve_fraction))

    def action(self, observation: BaselineObservation) -> ControlAction:
        net_load = observation.net_load_kw
        reserve = observation.reserve_energy_kwh(self.reserve_fraction)
        if net_load > 0.0 and observation.energy_kwh <= reserve + 1e-12:
            return ControlAction()
        if net_load < 0.0 and observation.energy_kwh >= observation.energy_max_kwh - 1e-12:
            return ControlAction()
        return ControlAction(p_bess_kw=bounded_power(net_load, observation.power_max_kw))

    def configuration(self) -> dict:
        return {"reserve_fraction": self.reserve_fraction}


__all__ = ["SelfConsumptionController"]
