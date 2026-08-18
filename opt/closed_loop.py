"""Closed-loop execution routines."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class ControlAction:
    """Controller command for one physical step."""

    p_bess_kw: float = 0.0
    load_shed_fraction: float | None = None
    pv_curtail_fraction: float | None = None
    objective: float | None = None

    @classmethod
    def from_mpc_step(cls, step: Mapping[str, Any] | None) -> "ControlAction":
        if step is None:
            return cls()
        return cls(
            p_bess_kw=float(step.get("P_bess_kw", 0.0)),
            load_shed_fraction=float(step.get("X_L", 0.0)),
            pv_curtail_fraction=float(step.get("X_PV", 0.0)),
            objective=float(step.get("obj", 0.0)),
        )


@dataclass(frozen=True)
class ClosedLoopResult:
    """Closed-loop execution counters."""

    n_ongrid_steps: int
    n_offgrid_steps: int
    n_solve_ok: int
    n_solve_fail: int
    total_solve_time_s: float
    total_time_s: float
    max_simultaneous_ch_dis_kw: float

    @property
    def average_solve_time_s(self) -> float | None:
        if not self.n_solve_ok:
            return None
        return self.total_solve_time_s / self.n_solve_ok


def run_mpc_closed_loop(*, env, mpc, forecaster, solver_opts: Mapping[str, Any],
                        n_iters: int, controller_label: str, progress_every: int = 288) -> ClosedLoopResult:
    """Run receding-horizon MPC, applying one action per physical step.

    GridEnv replaces the zero external command with local emergency control
    while the system is off-grid.
    """
    previous_bess_kw = 0.0
    n_ongrid = n_offgrid = n_solve_ok = n_solve_fail = 0
    solve_time_total = 0.0
    max_simultaneity_kw = 0.0
    run_t0 = time.perf_counter()

    while not env.done():
        loop_t0 = time.perf_counter()
        now = env.timestamp

        if env.mode == "offgrid":
            n_offgrid += 1
            first_step = None
        else:
            n_ongrid += 1
            try:
                forecasts = forecaster.get_forecasts(
                    start_dt0=now, intervals=None, dt_min=env.dt_min, include_actuals=False
                )
                if forecasts is None:
                    raise RuntimeError("forecast window out of data range")

                solve_t0 = time.perf_counter()
                mpc.build(
                    start_dt=now.to_pydatetime(),
                    forecasts=forecasts,
                    E_hat_kwh=float(env.E_meas),
                    P_bess_hat_kw=previous_bess_kw,
                )
                mpc.solve(tee=False, **dict(solver_opts))
                solve_time_total += time.perf_counter() - solve_t0
                n_solve_ok += 1
                max_simultaneity_kw = max(
                    max_simultaneity_kw, mpc.max_simultaneous_charge_discharge_kw()
                )
                first_step = mpc.extract_first_step(scenario="c0")
            except Exception as error:
                n_solve_fail += 1
                print(f"[mpc] WARN: solve failed at {now}: {error}. Using safe fallback.")
                first_step = None

        action = ControlAction.from_mpc_step(first_step)
        realized, done = env.step(
            P_bess_kw=action.p_bess_kw, X_L=action.load_shed_fraction,
            X_PV=action.pv_curtail_fraction, obj=action.objective,
            exec_time_sec=time.perf_counter() - loop_t0,
        )

        if isinstance(realized, dict):
            previous_bess_kw = float(realized.get("P_bess_kw", action.p_bess_kw))
        else:
            previous_bess_kw = action.p_bess_kw

        if env.iter_k % max(1, progress_every) == 0 or done:
            elapsed = time.perf_counter() - run_t0
            average = solve_time_total / n_solve_ok if n_solve_ok else 0.0
            print(
                f"[mpc] {controller_label} {env.iter_k}/{n_iters} | "
                f"elapsed={elapsed:.0f}s avg_solve={average:.2f}s "
                f"ok={n_solve_ok} fail={n_solve_fail} offgrid={n_offgrid}"
            )
        if done:
            break

    return ClosedLoopResult(
        n_ongrid_steps=n_ongrid,
        n_offgrid_steps=n_offgrid,
        n_solve_ok=n_solve_ok,
        n_solve_fail=n_solve_fail,
        total_solve_time_s=float(solve_time_total),
        total_time_s=float(time.perf_counter() - run_t0),
        max_simultaneous_ch_dis_kw=float(max_simultaneity_kw),
    )


__all__ = ["ClosedLoopResult", "ControlAction", "run_mpc_closed_loop"]
