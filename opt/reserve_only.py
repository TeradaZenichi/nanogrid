"""BESS reserve-only operational baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import BaselineObservation
from .closed_loop import ControlAction


@dataclass(frozen=True)
class ReserveOnlyController:
    """Keep the BESS idle on-grid and release it only through islanded control."""

    name = "reserve_only"
    version = "reserve-only-v1"

    def action(self, observation: BaselineObservation) -> ControlAction:
        return ControlAction()

    def configuration(self) -> dict:
        return {"policy": "zero on-grid command; GridEnv emergency control off-grid"}


__all__ = ["ReserveOnlyController"]
