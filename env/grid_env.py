"""Physical nanogrid environment for grid-connected and islanded operation."""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from opt.utils import build_dt_vector, load_series_scaled


def _get_nested(d: Dict[str, Any], path_list: List[str], default: Any = None):
    """Read the first available alternative path from a nested dictionary."""
    for path in path_list:
        cur = d
        ok = True
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _resolve_dt_vector(params: Dict[str, Any]) -> List[float]:
    """Resolve interval durations from the unflattened parameters."""
    for key in ["dt_vector", "dt_minutes", "dt_min_vector", "dt_min_list"]:
        if key in params and isinstance(params[key], (list, tuple)):
            vec = params[key]
            if all(isinstance(x, (int, float)) for x in vec):
                return list(vec)

    horizon_hours = _get_nested(params, ["time.horizon_hours", "horizon_hours", "mpc.horizon_hours"], 24)
    outage_hours = _get_nested(params, ["EDS.outage_duration_hours", "outage_duration_hours", "time.outage_duration_hours"], 0)
    t1 = _get_nested(params, ["time.timestep_1_min", "timestep_1_min", "mpc.timestep_1_min", "timestep_min"], 5)
    t2_paths = ["time.timestep_2_min", "timestep_2_min", "mpc.timestep_2_min", "dt_min2"]
    t2 = _get_nested(params, t2_paths, t1)

    h = int(horizon_hours)
    Hod = int(outage_hours)
    dt1 = int(t1)
    dt2 = int(t2)

    return build_dt_vector(h, Hod, dt1, dt2)


def _hour_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.hour):02d}:00"


def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / b if abs(b) > eps else 0.0


def _derive_scaling_params(params: Dict[str, Any]) -> Dict[str, float]:
    """Extract nominal load and PV powers from the unflattened parameters."""
    P_L_max = _get_nested(params, [
        "P_L_nom_kw", "Load.Pmax_kw", "Load.Pmax", "load.Pmax_kw", "load.Pmax"
    ], None)
    P_PV_max = _get_nested(params, [
        "P_PV_nom_kw", "PV.Pmax_kw", "PV.Pmax", "pv.Pmax_kw", "pv.Pmax"
    ], None)
    if P_L_max is None or P_PV_max is None:
        raise KeyError("Could not get P_L_nom_kw/P_PV_nom_kw or Load.Pmax_kw/PV.Pmax_kw from JSON.")
    return {"P_L_nom_kw": float(P_L_max), "P_PV_nom_kw": float(P_PV_max)}


class GridEnv:
    def __init__(
        self,
        params: Dict[str, Any],
        load_csv: str,
        pv_csv: str,
        start_dt0: pd.Timestamp,
        n_iters: int = 288,
        traces_path: str = "outputs/step_traces.json",
        debug: bool = False,
        clamp_soc_pct: bool = True,
        tol_kw: float = 1e-6,
    ):
        self.p = params
        self.load_csv = load_csv
        self.pv_csv = pv_csv
        self.n_iters = n_iters
        self.traces_path = traces_path
        self.mode = "ongrid"
        self.debug = debug
        self.clamp_soc_pct = clamp_soc_pct
        self.tol_kw = tol_kw

        self.dt_h = self.p.get("time", {}).get("timestep", 5) / 60.0

        self.start_dt0 = pd.Timestamp(start_dt0)

        EDS = self.p.get("EDS", {})
        self.outage_prob_daily = float(EDS.get("outage_probability_pct", 0.0)) / 100.0
        self.mean_outage_duration_h = float(EDS.get("outage_duration_hours", 4.0))
        std_dev_frac = float(EDS.get("outage_duration_std_dev_frac", 0.3))
        self.std_dev_outage_duration_h = self.mean_outage_duration_h * std_dev_frac

        self.outage_seed = EDS.get("seed", None)
        if self.outage_prob_daily > 0.0 and self.outage_seed is None:
            raise ValueError(
                "EDS.seed is required when EDS.outage_probability_pct > 0 "
                "(outage sampling must be reproducible)."
            )

        self.outage_active = False
        self.outage_end_time = None
        self._outage_calendar: List[Dict[str, Any]] = []

        scaling = _derive_scaling_params(self.p)
        self.pv_scaling = scaling["P_PV_nom_kw"]
        self.load_scaling = scaling["P_L_nom_kw"]
        self.load_kw_s, self.pv_kw_s = load_series_scaled(scaling, self.load_csv, self.pv_csv)
        self.dt_min = _resolve_dt_vector(self.p)

        self._parse_constraints_and_costs()
        self._rows: List[Dict[str, Any]] = []

        self.reset()

    def _parse_constraints_and_costs(self):
        """Parse BESS, grid caps, noise, RNGs, and costs from params."""
        B = self.p.get("BESS", {})
        E_nom = float(B.get("Emax_kwh", self.p.get("E_nom_kwh", 0.0)) or 0.0)
        E_nom = max(E_nom, 1e-9)
        DoD = float(B.get("DoD_frac", self.p.get("DoD_frac", 1.0)))
        DoD = min(max(0.0, DoD), 1.0)
        soc_min = 1.0 - DoD
        E_min = E_nom * soc_min
        E_max = E_nom

        P_max = float(B.get("Pmax_kw", self.p.get("P_bess_max_kw", float("+inf"))))
        if not (P_max >= 0.0):
            P_max = float("+inf")
        ramp = B.get("ramp_kw_per_step", self.p.get("ramp_kw_per_step", None))
        ramp = float(ramp) if ramp is not None else None
        if ramp is not None and ramp < 0:
            ramp = None

        eta_c = float(B.get("eta_c", self.p.get("eta_c", 1.0)))
        eta_d = float(B.get("eta_d", self.p.get("eta_d", 1.0)))
        eta_c = min(max(eta_c, 1e-6), 1.0)
        eta_d = min(max(eta_d, 1e-6), 1.0)

        E_init = B.get("E_init_kwh", self.p.get("E_init_kwh", None))
        E_init = None if E_init is None else float(E_init)

        noisy_flag = bool(B.get("noisy", False))
        noise_dict = B.get("noise", {}) if isinstance(B.get("noise", {}), dict) else {}
        self.noise = {
            "enabled": noisy_flag,
            "type": noise_dict.get("type", "gauss"),
            "std_frac": float(noise_dict.get("std_frac", 0.05)),
            "std_kw": float(noise_dict.get("std_kw", 0.0)),
            "seed": noise_dict.get("seed", None),
        }
        if self.noise["enabled"] and self.noise["seed"] is None:
            raise ValueError(
                "BESS.noise.seed is required when BESS.noisy is true "
                "(actuator noise must be reproducible)."
            )
        self._rng = random.Random(self.noise["seed"]) if self.noise["enabled"] else None

        self.bess = {
            "E_nom": E_nom, "E_min": E_min, "E_max": E_max, "soc_min": soc_min,
            "P_max": P_max, "ramp": ramp, "eta_c": eta_c, "eta_d": eta_d, "E_init": E_init,
        }

        EDS = self.p.get("EDS", {})
        P_import_max = float(EDS.get("Pmax_kw", EDS.get("Pmax", float("+inf"))))
        P_export_max = float(EDS.get("Pmin", 0.0))

        self.grid_caps = {
            "P_import_max": max(0.0, P_import_max),
            "P_export_max": max(0.0, P_export_max),
        }

        C = self.p.get("costs", {})
        self.costs = {
            "c_shed": float(C.get("c_shed_per_kwh", 0.0)),
            "c_curt": float(C.get("c_pv_curt_per_kwh", 0.0)),
            "TOU": C.get("EDS", {}) if isinstance(C.get("EDS", {}), dict) else {},
        }

    def _log(self, msg: str):
        if self.debug:
            print(f"[GridEnv] {msg}")

    def reset(self):
        """Reset environment state for a fresh run."""
        self.iter_k = 0
        self.timestamp = self.start_dt0

        if self.bess["E_init"] is not None:
            E0 = float(self.bess["E_init"])
        else:
            E0 = self.bess["E_nom"] * max(self.bess["soc_min"], 0.5)
        self.E_meas = min(max(E0, self.bess["E_min"]), self.bess["E_max"])
        self._prev_Pb = 0.0

        self.outage_active = False
        self.outage_end_time = None
        self._generate_outage_calendar()
        self._set_mode_from_calendar()

        self._log(
            "reset: start=%s, E=%.3f kWh (E_min=%.3f, E_max=%.3f, Pmax=%s, ramp=%s)"
            % (self.start_dt0, self.E_meas, self.bess["E_min"], self.bess["E_max"],
               ("inf" if math.isinf(self.bess["P_max"]) else f"{self.bess['P_max']:.3f}"),
               str(self.bess["ramp"]))
        )

    def _generate_outage_calendar(self):
        """Pre-generate outage events from EDS.seed for the whole window.

        At most one outage per day (Bernoulli draw), start uniform within the
        day, duration lognormal. Deterministic given seed + window, so all
        controllers see the same events.
        """
        self._outage_calendar = []
        if self.outage_prob_daily <= 0.0:
            return

        rng = random.Random(self.outage_seed)
        m = self.mean_outage_duration_h
        s = self.std_dev_outage_duration_h
        if s > 0.0:
            sigma2 = math.log(1.0 + (s / m) ** 2)
            mu = math.log(m) - 0.5 * sigma2
            sigma = math.sqrt(sigma2)
        else:
            mu, sigma = math.log(m), 0.0

        end_dt = self.start_dt0 + pd.Timedelta(hours=self.n_iters * self.dt_h)
        day = self.start_dt0.normalize()
        prev_end = None
        while day < end_dt:
            if rng.random() < self.outage_prob_daily:
                start = day + pd.Timedelta(hours=24.0 * rng.random())
                duration_h = rng.lognormvariate(mu, sigma) if sigma > 0.0 else m
                duration_h = max(self.dt_h, duration_h)  # at least one timestep
                if prev_end is None or start >= prev_end:
                    end = start + pd.Timedelta(hours=duration_h)
                    self._outage_calendar.append(
                        {"start": start, "end": end, "duration_h": float(duration_h)}
                    )
                    prev_end = end
            day = day + pd.Timedelta(days=1)

        self._log(
            f"Outage calendar generated: {len(self._outage_calendar)} event(s) "
            f"in [{self.start_dt0}, {end_dt}) with seed={self.outage_seed}."
        )

    def save_outage_calendar(self, path) -> None:
        """Persist the outage calendar (and its parameters) as JSON for auditability."""
        payload = {
            "seed": self.outage_seed,
            "outage_prob_daily": self.outage_prob_daily,
            "duration_mean_h": self.mean_outage_duration_h,
            "duration_std_h": self.std_dev_outage_duration_h,
            "duration_distribution": "lognormal",
            "window_start": str(self.start_dt0),
            "window_end": str(self.start_dt0 + pd.Timedelta(hours=self.n_iters * self.dt_h)),
            "events": [
                {
                    "start": str(ev["start"]),
                    "end": str(ev["end"]),
                    "duration_h": ev["duration_h"],
                }
                for ev in self._outage_calendar
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log(f"Outage calendar saved at: {path.as_posix()}")

    def _set_mode_from_calendar(self):
        """Set mode/outage flags for the current timestamp from the pre-generated calendar."""
        active = None
        for ev in self._outage_calendar:
            if ev["start"] <= self.timestamp < ev["end"]:
                active = ev
                break
        if active is not None:
            if not self.outage_active:
                self._log(f"--- OUTAGE ACTIVE at {self.timestamp} (until {active['end']}) ---")
            self.outage_active = True
            self.outage_end_time = active["end"]
            self.mode = "offgrid"
        else:
            if self.outage_active:
                self._log(f"Outage period finished. Returning to on-grid mode at {self.timestamp}.")
            self.outage_active = False
            self.outage_end_time = None
            self.mode = "ongrid"

    def _update_outage_status(self):
        """Update the contingency state at each step tail; sets mode for the next step."""
        self._set_mode_from_calendar()

    def done(self) -> bool:
        return self.iter_k >= self.n_iters

    def step(
        self,
        P_bess_kw: float,
        X_L: Optional[float],
        X_PV: Optional[float],
        obj: Optional[float] = None,
        exec_time_sec: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Apply one command and advance the physical state."""
        load0 = self.load_kw_s.get(self.timestamp, 0.0)
        pv0 = self.pv_kw_s.get(self.timestamp, 0.0)

        XL_cmd = 0.0 if X_L is None else float(X_L)
        XPV_cmd = 0.0 if X_PV is None else float(X_PV)
        XL = min(max(XL_cmd, 0.0), 1.0)
        XPV = min(max(XPV_cmd, 0.0), 1.0)

        clamps: Dict[str, Any] = {}
        if XL != XL_cmd or XPV != XPV_cmd:
            clamps["fractions"] = {"X_L": XL, "X_PV": XPV}

        if self.mode == "offgrid":
            if XL > 0.0:
                clamps["offgrid_force_XL0"] = {"prev_XL": XL, "reason": "meet_load"}
            if XPV > 0.0:
                clamps["offgrid_force_XPV0"] = {"prev_XPV": XPV, "reason": "use_all_pv"}
            XL = 0.0
            XPV = 0.0

            served = load0
            shed = 0.0
            usedpv = pv0
            curt = 0.0

            Pgrid_in = 0.0
            Pgrid_out = 0.0

            Pmax = self.bess["P_max"]
            if math.isinf(Pmax):
                Pmax = 1e12
            if self.bess["ramp"] is not None:
                Pb_min_ramp = self._prev_Pb - self.bess["ramp"]
                Pb_max_ramp = self._prev_Pb + self.bess["ramp"]
            else:
                Pb_min_ramp = -Pmax
                Pb_max_ramp = Pmax
            Pb_min = max(-Pmax, Pb_min_ramp)
            Pb_max = min(Pmax, Pb_max_ramp)

            Pdis_cap_E = self.bess["eta_d"] * max(self.E_meas - self.bess["E_min"], 0.0) / max(self.dt_h, 1e-9)
            Pch_cap_E = max(self.bess["E_max"] - self.E_meas, 0.0) / (self.bess["eta_c"] * max(self.dt_h, 1e-9))

            Pb_des_raw = served - usedpv
            Pb_des_ramp = min(max(Pb_des_raw, Pb_min), Pb_max)
            if Pb_des_ramp >= 0.0:
                Pb_des = min(Pb_des_ramp, Pdis_cap_E)
            else:
                Pb_des = -min(-Pb_des_ramp, Pch_cap_E)

            clamps["offgrid_cmd_from_diff"] = {
                "delta_kw": float(served - usedpv),
                "Pb_des_raw_kw": float(Pb_des_raw),
                "ramp_window_kw": [float(Pb_min), float(Pb_max)],
                "Pb_des_after_ramp_kw": float(Pb_des_ramp),
                "Pdis_cap_E_kw": float(Pdis_cap_E),
                "Pch_cap_E_kw": float(Pch_cap_E),
                "Pb_des_after_caps_kw": float(Pb_des),
            }
        else:
            served = load0 * (1.0 - XL)
            shed = load0 - served
            usedpv = pv0 * (1.0 - XPV)
            curt = pv0 - usedpv
            Pgrid_in = 0.0
            Pgrid_out = 0.0
            Pb_des = float(P_bess_kw)

        # Actuator noise is disabled during islanded emergency control.
        Pb_after_noise = Pb_des
        noise_applied = False
        if (self.mode == "ongrid") and self.noise["enabled"]:
            Pmax_for_noise = self.bess["P_max"]
            fallback = (0.05 * Pmax_for_noise) if (not math.isinf(Pmax_for_noise)) else 1.0
            base = max(abs(Pb_des), fallback)
            sigma = max(0.0, self.noise["std_kw"]) + max(0.0, self.noise["std_frac"]) * base
            if sigma > 0.0:
                eps = self._rng.gauss(0.0, sigma)
                Pb_after_noise = Pb_des + eps
                clamps["bess_noise"] = {"eps_kw": eps, "sigma_kw": sigma, "base_kw": base}
                noise_applied = True

        Pb = Pb_after_noise
        if abs(Pb) > self.bess["P_max"]:
            Pb = max(min(Pb, self.bess["P_max"]), -self.bess["P_max"])
            clamps["bess_pmax"] = Pb
        if self.bess["ramp"] is not None:
            Pb_min_lim = self._prev_Pb - self.bess["ramp"]
            Pb_max_lim = self._prev_Pb + self.bess["ramp"]
            if Pb < Pb_min_lim or Pb > Pb_max_lim:
                Pb = min(max(Pb, Pb_min_lim), Pb_max_lim)
                clamps["bess_ramp"] = {"min": Pb_min_lim, "max": Pb_max_lim, "applied": Pb}

        Pdis = max(Pb, 0.0)
        Pch = max(-Pb, 0.0)

        if Pdis > 0.0:
            Pdis_cap = self.bess["eta_d"] * max(self.E_meas - self.bess["E_min"], 0.0) / max(self.dt_h, 1e-9)
            if Pdis > Pdis_cap:
                Pdis = Pdis_cap
                clamps["bess_energy_min"] = Pdis_cap
        if Pch > 0.0:
            Pch_cap = max(self.bess["E_max"] - self.E_meas, 0.0) / (self.bess["eta_c"] * max(self.dt_h, 1e-9))
            if Pch > Pch_cap:
                Pch = Pch_cap
                clamps["bess_energy_max"] = Pch_cap

        Pb_eff = Pdis - Pch

        supply = usedpv + Pdis + Pgrid_in - Pgrid_out
        ref_line = served + Pch
        residual = ref_line - supply

        # Continuous shedding avoids the surplus created by the former 10% plant increments.
        if self.mode == "offgrid":
            # After battery saturation, shed or curtail exactly the remaining mismatch.
            if residual > self.tol_kw:
                additional_shed = min(float(residual), max(float(served), 0.0))
                if additional_shed > 0.0:
                    served -= additional_shed
                    shed += additional_shed
                    XL = min(1.0, max(0.0, _safe_div(shed, load0)))
                    clamps["offgrid_autofix_shed"] = {
                        "additional_kw": float(additional_shed),
                        "new_XL_pct": float(XL * 100.0),
                    }
                supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply
            elif residual < -self.tol_kw:
                surplus = -residual
                dXPV = min(1.0 - XPV, _safe_div(surplus, pv0))
                if dXPV > 0:
                    XPV += dXPV
                    usedpv = pv0 * (1.0 - XPV)
                    curt   = pv0 - usedpv
                    clamps["offgrid_autofix_curtail"] = {"dXPV": dXPV}
                supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply
        else:
            if residual > self.tol_kw:
                take = min(residual, self.grid_caps["P_import_max"])
                if take > 0:
                    Pgrid_in += take
                    residual -= take
                    clamps["grid_import"] = {"kW": take, "outage": False}
            if residual > self.tol_kw and Pch > 1e-9:
                # Interrupt charging before shedding load.
                dec = min(Pch, residual)
                Pch -= dec
                Pb_eff = Pdis - Pch
                clamps["ongrid_autofix_reduce_charge"] = {"kW": float(dec)}
                supply = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply
            if residual > self.tol_kw:
                # Increase feasible discharge before shedding load.
                pmax = self.bess["P_max"]
                if math.isinf(pmax):
                    pmax = 1e12
                hi = pmax
                if self.bess["ramp"] is not None:
                    hi = min(hi, self._prev_Pb + self.bess["ramp"])
                pdis_cap_e = self.bess["eta_d"] * max(self.E_meas - self.bess["E_min"], 0.0) / max(self.dt_h, 1e-9)
                boost = min(residual, max(0.0, hi - Pb_eff), max(0.0, pdis_cap_e - Pdis))
                if boost > self.tol_kw:
                    Pdis += boost
                    Pb_eff = Pdis - Pch
                    residual -= boost
                    clamps["ongrid_autofix_discharge_boost"] = {"kW": float(boost)}
                    supply = usedpv + Pdis + Pgrid_in - Pgrid_out
                    ref_line = served + Pch
                    residual = ref_line - supply
            if residual > self.tol_kw:
                additional_shed = min(float(residual), max(float(served), 0.0))
                if additional_shed > 0.0:
                    served -= additional_shed
                    shed += additional_shed
                    XL = min(1.0, max(0.0, _safe_div(shed, load0)))
                    clamps["ongrid_autofix_shed"] = {
                        "additional_kw": float(additional_shed),
                        "new_XL_pct": float(XL * 100.0),
                    }
                supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply
            if residual < -self.tol_kw:
                surplus = -residual
                take = min(surplus, self.grid_caps["P_export_max"])
                if take > 0:
                    Pgrid_out += take
                    residual  += take
                    clamps["grid_export"] = {"kW": take, "outage": False}
            if residual < -self.tol_kw:
                surplus = -residual
                dXPV = min(1.0 - XPV, _safe_div(surplus, pv0))
                if dXPV > 0:
                    XPV += dXPV
                    usedpv = pv0 * (1.0 - XPV)
                    curt   = pv0 - usedpv
                    clamps["ongrid_autofix_curtail"] = {"dXPV": dXPV}
                supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply
            if residual < -self.tol_kw and Pdis > self.tol_kw:
                # Only reached when export and PV curtailment cannot absorb
                # the remaining discharge surplus.
                pmax = self.bess["P_max"]
                if math.isinf(pmax):
                    pmax = 1e12
                lo = -pmax
                if self.bess["ramp"] is not None:
                    lo = max(lo, self._prev_Pb - self.bess["ramp"])
                reducible = min(Pdis, max(0.0, Pb_eff - lo))
                reduction = min(-residual, reducible)
                if reduction > self.tol_kw:
                    Pdis -= reduction
                    Pb_eff = Pdis - Pch
                    clamps["ongrid_emergency_reduce_discharge"] = {
                        "kW": float(reduction),
                        "ramp_min_kw": float(lo),
                    }
                    supply = usedpv + Pdis + Pgrid_in - Pgrid_out
                    ref_line = served + Pch
                    residual = ref_line - supply

        if abs(residual) <= self.tol_kw:
            residual = 0.0
        else:
            raise RuntimeError(
                "Physical power balance could not be restored at "
                f"{self.timestamp} in {self.mode} mode: residual={residual:.12g} kW, "
                f"load={load0:.12g} kW, pv={pv0:.12g} kW, "
                f"bess={Pb_eff:.12g} kW."
            )

        if self.mode == "offgrid" and (
            abs(Pgrid_in) > self.tol_kw or abs(Pgrid_out) > self.tol_kw
        ):
            raise RuntimeError(
                "Grid exchange remained active during an outage at "
                f"{self.timestamp}: import={Pgrid_in:.12g} kW, "
                f"export={Pgrid_out:.12g} kW."
            )

        E_next = self.E_meas + self.dt_h * (self.bess["eta_c"] * Pch - (1.0 / self.bess["eta_d"]) * Pdis)
        E_next = min(max(E_next, self.bess["E_min"]), self.bess["E_max"])
        soc_pct = 100.0 * (E_next / self.bess["E_nom"])
        if self.clamp_soc_pct:
            soc_pct = min(max(soc_pct, 0.0), 100.0)

        tou = 0.0
        if self.mode == "ongrid":
            key = _hour_key(pd.Timestamp(self.timestamp))
            tou = float(self.costs["TOU"].get(key, 0.0))

        energy_grid_kwh = Pgrid_in * self.dt_h
        energy_shed_kwh = shed * self.dt_h
        energy_curt_kwh = curt * self.dt_h
        cost_grid = tou * energy_grid_kwh
        cost_shed = self.costs["c_shed"] * energy_shed_kwh
        cost_curt = self.costs["c_curt"] * energy_curt_kwh
        cost_total = cost_grid + cost_shed + cost_curt

        row = {
            "timestamp": pd.Timestamp(self.timestamp),
            "cmd_P_bess_kw": float(Pb_des),
            "cmd_X_L": XL_cmd, "cmd_X_PV": XPV_cmd,
            "cmd_P_bess_kw_after_noise": Pb_after_noise,
            "Load_kw": load0, "PV_kw": pv0,
            "Load_served_kw": served, "Shedding_kw": shed,
            "PV_used_kw": usedpv, "Curtailment_kw": curt,
            "P_bess_kw": Pb_eff, "P_bess_discharge_kw": Pdis, "P_bess_charge_mag_kw": Pch,
            "P_grid_in_kw": Pgrid_in, "P_grid_out_kw": Pgrid_out,
            "Total_supply_kw": usedpv + Pdis + Pgrid_in - Pgrid_out,
            "Target_kw": served + Pch,
            "Residual_kw": residual,
            "E_kwh": E_next, "SoC_pct": soc_pct,
            "TOU_cperkwh": tou,
            "cost_grid": cost_grid, "cost_shed": cost_shed, "cost_curt": cost_curt, "cost_total": cost_total,
            "obj": (None if obj is None else float(obj)),
            "mode": self.mode,
            "clamps": clamps,
            "outage_active": self.outage_active,
            "exec_time_sec": exec_time_sec,
            "noise_applied": bool(noise_applied),
            "cmd_diff_kw": float(served - usedpv) if self.mode == "offgrid" else None,
        }

        self._rows.append(row)
        self.E_meas = E_next
        self._prev_Pb = Pb_eff
        self.timestamp = self.timestamp + pd.Timedelta(minutes=self.dt_h * 60)
        self.iter_k += 1

        self._update_outage_status()

        self._log(
            f"iter={self.iter_k:04d} t0={pd.Timestamp(self.timestamp)} mode={self.mode} "
            f"Pb_eff={Pb_eff:.3f} SoC={soc_pct:.2f}% residual={residual:.4f} cost={cost_total:.3f}"
        )

        return row, self.done()

    def to_dataframe(self) -> pd.DataFrame:
        """Return logged rows as a time-indexed DataFrame (or empty if no rows)."""
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows).set_index("timestamp").sort_index()
