"""Common interface and closed-loop runner for rule-based controllers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import pandas as pd

from .closed_loop import ControlAction


BASELINE_MODEL_VERSION = "2026.08-causal-rule-based-v1"


@dataclass(frozen=True)
class BaselineObservation:
    """Measurements available to a baseline at one physical instant."""

    timestamp: pd.Timestamp
    load_kw: float
    pv_kw: float
    energy_kwh: float
    energy_min_kwh: float
    energy_max_kwh: float
    power_max_kw: float
    previous_bess_kw: float
    tariff_per_kwh: float

    @property
    def net_load_kw(self) -> float:
        return self.load_kw - self.pv_kw

    def reserve_energy_kwh(self, reserve_fraction: float) -> float:
        usable = self.energy_max_kwh - self.energy_min_kwh
        return self.energy_min_kwh + reserve_fraction * usable


class BaselineController(Protocol):
    name: str
    version: str

    def action(self, observation: BaselineObservation) -> ControlAction:
        """Return the command for the current physical step."""

    def configuration(self) -> dict:
        """Return the serializable controller configuration."""


@dataclass(frozen=True)
class BaselineLoopResult:
    n_ongrid_steps: int
    n_offgrid_steps: int
    total_action_time_s: float
    total_time_s: float

    @property
    def average_action_time_s(self) -> float | None:
        if not self.n_ongrid_steps:
            return None
        return self.total_action_time_s / self.n_ongrid_steps


def validate_reserve_fraction(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("reserve_fraction must be between zero and one")
    return value


def bounded_power(value: float, power_max_kw: float) -> float:
    limit = max(0.0, float(power_max_kw))
    return min(max(float(value), -limit), limit)


def _validate_action(action: ControlAction) -> None:
    values = (action.p_bess_kw, action.load_shed_fraction, action.pv_curtail_fraction)
    if any(value is not None and not math.isfinite(float(value)) for value in values):
        raise ValueError("Baseline returned a non-finite command")


def _observation(env, previous_bess_kw: float) -> BaselineObservation:
    timestamp = pd.Timestamp(env.timestamp)
    hour = timestamp.strftime("%H:00")
    return BaselineObservation(
        timestamp=timestamp,
        load_kw=float(env.load_kw_s.get(timestamp, 0.0)),
        pv_kw=float(env.pv_kw_s.get(timestamp, 0.0)),
        energy_kwh=float(env.E_meas),
        energy_min_kwh=float(env.bess["E_min"]),
        energy_max_kwh=float(env.bess["E_max"]),
        power_max_kw=float(env.bess["P_max"]),
        previous_bess_kw=float(previous_bess_kw),
        tariff_per_kwh=float(env.costs["TOU"].get(hour, 0.0)),
    )


def run_baseline_closed_loop(
    *, env, controller: BaselineController, n_iters: int,
    progress_every: int = 288,
) -> BaselineLoopResult:
    """Run a causal rule-based controller against the physical environment."""
    previous_bess_kw = 0.0
    n_ongrid = n_offgrid = 0
    action_time = 0.0
    run_t0 = time.perf_counter()

    while not env.done():
        loop_t0 = time.perf_counter()
        if env.mode == "offgrid":
            n_offgrid += 1
            action = ControlAction()
        else:
            n_ongrid += 1
            action_t0 = time.perf_counter()
            action = controller.action(_observation(env, previous_bess_kw))
            action_time += time.perf_counter() - action_t0
            _validate_action(action)

        realized, done = env.step(
            P_bess_kw=action.p_bess_kw,
            X_L=action.load_shed_fraction,
            X_PV=action.pv_curtail_fraction,
            obj=action.objective,
            exec_time_sec=time.perf_counter() - loop_t0,
        )
        previous_bess_kw = float(realized.get("P_bess_kw", action.p_bess_kw))

        if env.iter_k % max(1, progress_every) == 0 or done:
            elapsed = time.perf_counter() - run_t0
            print(
                f"[baseline] {controller.name} {env.iter_k}/{n_iters} | "
                f"elapsed={elapsed:.1f}s offgrid={n_offgrid}",
                flush=True,
            )
        if done:
            break

    return BaselineLoopResult(
        n_ongrid_steps=n_ongrid,
        n_offgrid_steps=n_offgrid,
        total_action_time_s=float(action_time),
        total_time_s=float(time.perf_counter() - run_t0),
    )


__all__ = [
    "BASELINE_MODEL_VERSION",
    "BaselineController",
    "BaselineLoopResult",
    "BaselineObservation",
    "bounded_power",
    "run_baseline_closed_loop",
    "validate_reserve_fraction",
]
