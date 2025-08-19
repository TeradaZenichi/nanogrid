# -*- coding: utf-8 -*-
"""
env/grid_env.py

Environment that enforces JSON constraints independently from the MPC,
with safety actions in OFF-GRID and TOU grid import in ON-GRID.

OFF-GRID per-step safety:
  1) apply noise to commanded P_bess_kw,
  2) check balance with X_L and X_PV,
  3) if mismatch: auto-fix via shedding/curtailment (BESS already capped per step).

ON-GRID:
  - receive P_bess_kw, X_L, X_PV,
  - cover mismatch via grid import subject to EDS-like caps and outages,
  - if still mismatch: shedding/curtail,
  - compute time-of-use grid cost and penalties.

IMPORTANT: start_dt0 MUST be provided in __init__. If it doesn't exactly match
an index timestamp, it will be aligned to the next available timestamp (or the
last one if past the end).
"""

import math
import random
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

from opt.utils import (
    build_dt_vector, build_time_grid,
    load_series_scaled, slice_forecasts,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_nested(d: Dict[str, Any], path_list: List[str], default: Any = None):
    """Try multiple dotted paths (e.g., 'a.b.c') and return the first found value."""
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
    """Build the horizon step vector (minutes) robustly (supports nested params.time.*)."""
    for key in ["dt_vector", "dt_minutes", "dt_min_vector", "dt_min_list"]:
        if key in params and isinstance(params[key], (list, tuple)):
            vec = params[key]
            if all(isinstance(x, (int, float)) for x in vec):
                return list(vec)

    horizon_hours = _get_nested(params, ["time.horizon_hours", "horizon_hours", "mpc.horizon_hours"], 24)
    fine_hours    = _get_nested(params, ["time.fine_hours", "fine_hours", "mpc.fine_hours"], 1)
    t1            = _get_nested(params, ["time.timestep_1_min", "timestep_1_min", "mpc.timestep_1_min", "timestep_min"], 5)
    t2            = _get_nested(params, ["time.timestep_2_min", "timestep_2_min", "mpc.timestep_2_min", "dt_min2"], t1)

    try:
        h  = int(horizon_hours)
        fh = int(fine_hours)
        dt1 = int(t1)
        dt2 = int(t2)
    except Exception:
        h, fh, dt1, dt2 = 24, 1, 5, 5

    return build_dt_vector(h, fh, dt1, dt2)


def _hour_key(ts: pd.Timestamp) -> str:
    """Return 'HH:00' key for costs.EDS mapping."""
    return f"{int(ts.hour):02d}:00"


def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / b if abs(b) > eps else 0.0


# ---------------------------------------------------------------------
# GridEnv
# ---------------------------------------------------------------------

class GridEnv:
    """
    Modes (manual only):
      - "offgrid": no grid; environment enforces balance via BESS + shedding/curtailment.
      - "ongrid" : grid import allowed with EDS-like caps & outages; TOU cost.

    The environment:
      - holds scaled load/PV series,
      - exposes rolling windows (times, forecasts),
      - applies first-step decisions (BESS, shedding, curtailment, grid import),
      - ENFORCES JSON limits (BESS power/ramp/energy; EDS/grid caps, outages),
      - optionally injects actuator noise in BESS net power (BESS.noisy),
      - integrates battery energy and tracks SoC,
      - computes grid/shedding/curtail costs per step,
      - logs commanded vs applied values and what was clamped/auto-fixed.
    """

    # ----------------------------- init -----------------------------

    def __init__(
        self,
        params: Dict[str, Any],
        load_csv: str,
        pv_csv: str,
        start_dt0,  # <-- REQUIRED
        n_iters: int = 288,
        traces_path: str = "outputs/step_traces.json",
        mode: str = "offgrid",
        *,
        debug: bool = False,
        clamp_soc_pct: bool = True,
        tol_kw: float = 1e-6,
    ):
        self.p = params
        self.load_csv = load_csv
        self.pv_csv = pv_csv
        self.n_iters = n_iters
        self.traces_path = traces_path
        self.mode = mode.lower()
        self.debug = debug
        self.clamp_soc_pct = clamp_soc_pct
        self.tol_kw = tol_kw

        # Horizon grid (minutes)
        self.dt_min = _resolve_dt_vector(self.p)

        # Scaled series (kW)
        self.load_kw_s, self.pv_kw_s = load_series_scaled(self.p, self.load_csv, self.pv_csv)

        # Common time index
        self.common_index = self.load_kw_s.index.intersection(self.pv_kw_s.index)
        if len(self.common_index) == 0:
            raise RuntimeError("Load and PV series have no overlapping timestamps.")

        # Parse constraints/costs/noise
        self._parse_constraints_and_costs()

        # State & buffers
        self._rows: List[Dict[str, Any]] = []
        self._last_times: Optional[List[pd.Timestamp]] = None
        self._last_forecasts: Optional[Dict[str, Any]] = None

        # MUST set start now
        self.reset(start_dt0=start_dt0)

    # --------------------- parse constraints ----------------------

    def _parse_constraints_and_costs(self):
        """Read BESS, EDS (used as grid proxy), costs and noise from JSON."""
        # ---- BESS ----
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
            "enabled": noisy_flag or bool(noise_dict),
            "type": noise_dict.get("type", "gauss"),
            "std_frac": float(noise_dict.get("std_frac", 0.05)),
            "std_kw": float(noise_dict.get("std_kw", 0.0)),
            "seed": noise_dict.get("seed", None),
        }
        self._rng = random.Random(self.noise["seed"]) if self.noise["enabled"] and self.noise["seed"] is not None else None

        self.bess = {
            "E_nom": E_nom, "E_min": E_min, "E_max": E_max, "soc_min": soc_min,
            "P_max": P_max, "ramp": ramp, "eta_c": eta_c, "eta_d": eta_d, "E_init": E_init,
        }

        # ---- EDS as "grid proxy" ----
        EDS = self.p.get("EDS", {})
        self.grid_caps = {
            "P_import_max": float(EDS.get("Pmax_kw", EDS.get("Pmax", float("+inf")))),
            "P_export_max": 0.0,  # export disabled by default (can be extended)
            "outage_prob": float(EDS.get("outage_probability_pct", 0.0)) / 100.0,
            "Pmin": float(EDS.get("Pmin", 0.0)),
        }
        self._rng_outage = random.Random(self.noise["seed"] if self.noise["seed"] is not None else None)

        # ---- Costs ----
        C = self.p.get("costs", {})
        self.costs = {
            "c_shed": float(C.get("c_shed_per_kwh", 0.0)),
            "c_curt": float(C.get("c_pv_curt_per_kwh", 0.0)),
            "TOU": C.get("EDS", {})  # dict "HH:00" -> price [$/kWh]
        }

    # --------------------------- logging ---------------------------

    def _log(self, msg: str):
        if self.debug:
            print(f"[GridEnv] {msg}")

    # -------------------------- lifecycle --------------------------

    def _align_start(self, start_dt0: pd.Timestamp) -> pd.Timestamp:
        """Align start to the next available timestamp in the common index (or last if beyond)."""
        ts = pd.Timestamp(start_dt0)
        idx = self.common_index
        if ts in idx:
            return ts
        # next available (ceil)
        pos = idx.searchsorted(ts, side="left")
        if pos >= len(idx):
            aligned = pd.Timestamp(idx[-1])
            self._log(f"start {ts} beyond data — aligned to last available {aligned}")
            return aligned
        aligned = pd.Timestamp(idx[pos])
        self._log(f"start {ts} not in data — aligned to next available {aligned}")
        return aligned

    def reset(self, start_dt0):
        """Reset the environment to a specific start datetime (MANDATORY)."""
        aligned = self._align_start(pd.Timestamp(start_dt0))
        self.start_dt0 = pd.Timestamp(aligned).to_pydatetime()
        self.iter_k = 0

        if self.bess["E_init"] is not None:
            E0 = float(self.bess["E_init"])
        else:
            E0 = self.bess["E_nom"] * max(self.bess["soc_min"], 0.5)
        self.E_meas = min(max(E0, self.bess["E_min"]), self.bess["E_max"])
        self._prev_Pb = 0.0  # for ramp

        self._log(
            "reset: start=%s, E=%.3f kWh (E_min=%.3f, E_max=%.3f, Pmax=%s, ramp=%s, noisy=%s, grid_cap=%.3f)"
            % (self.start_dt0, self.E_meas, self.bess["E_min"], self.bess["E_max"],
               ("inf" if math.isinf(self.bess["P_max"]) else f"{self.bess['P_max']:.3f}"),
               str(self.bess["ramp"]), str(self.noise["enabled"]), self.grid_caps["P_import_max"])
        )

    def set_mode(self, mode: str):
        self.mode = mode.lower()
        self._log(f"mode set to: {self.mode}")

    def done(self) -> bool:
        return self.iter_k >= self.n_iters

    # ---------------------- window -------------------------

    def get_window(self) -> Tuple[Optional[List[pd.Timestamp]], Optional[Dict[str, Any]]]:
        """Return (times, forecasts) for the current rolling window (does NOT advance time)."""
        times = build_time_grid(self.start_dt0, self.dt_min)
        if any(t not in self.load_kw_s.index for t in times) or any(t not in self.pv_kw_s.index for t in times):
            self._log("end of data — get_window() returns None.")
            return None, None
        forecasts = slice_forecasts(times, self.load_kw_s, self.pv_kw_s)
        self._last_times = times
        self._last_forecasts = forecasts
        return times, forecasts

    # ----------------------------- step -----------------------------

    def step(
        self,
        P_bess_kw: float,
        X_L: Optional[float],
        X_PV: Optional[float],
        *,
        obj: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Apply commanded decisions with JSON enforcement and safety actions.
        OFF-GRID: noise -> check balance -> auto-fix (BESS (already step-capped), then shedding/curtail).
        ON-GRID : noise -> grid import within EDS caps/outage -> auto-fix -> costs.

        Inputs:
          - P_bess_kw: commanded net BESS power (+ discharge, − charge).
          - X_L, X_PV: MPC-provided fractions in [0,1] (can be None -> assume 0).
        """
        if self._last_times is None or self._last_forecasts is None:
            raise RuntimeError("step() called without an active window. Call get_window() first.")

        times = self._last_times
        forecasts = self._last_forecasts
        t0 = times[0]
        dt_h = (times[1] - times[0]).total_seconds() / 3600.0

        load0 = float(forecasts["load_kw"][t0])
        pv0   = float(forecasts["pv_kw"][t0])

        # ---- commanded values ----
        XL_cmd = 0.0 if X_L is None else float(X_L)
        XPV_cmd = 0.0 if X_PV is None else float(X_PV)
        XL = min(max(XL_cmd, 0.0), 1.0)
        XPV = min(max(XPV_cmd, 0.0), 1.0)
        clamps: Dict[str, Any] = {}
        if XL != XL_cmd or XPV != XPV_cmd:
            clamps["fractions"] = {"X_L": XL, "X_PV": XPV}

        # ---- BESS noise + clamps (Pmax/ramp) ----
        Pb_des = float(P_bess_kw)
        Pb_after_noise = Pb_des
        if self.noise["enabled"]:
            Pmax = self.bess["P_max"]
            fallback = (0.05 * Pmax) if (not math.isinf(Pmax)) else 1.0
            base = max(abs(Pb_des), fallback)
            sigma = max(0.0, self.noise["std_kw"]) + max(0.0, self.noise["std_frac"]) * base
            if sigma > 0.0:
                rng = self._rng or random
                eps = rng.gauss(0.0, sigma)
                Pb_after_noise = Pb_des + eps
                clamps["bess_noise"] = {"eps_kw": eps, "sigma_kw": sigma, "base_kw": base}

        # Pmax symmetric
        Pb = Pb_after_noise
        if abs(Pb) > self.bess["P_max"]:
            Pb = max(min(Pb, self.bess["P_max"]), -self.bess["P_max"])
            clamps["bess_pmax"] = Pb
        # Ramp
        if self.bess["ramp"] is not None:
            Pb_min = self._prev_Pb - self.bess["ramp"]
            Pb_max = self._prev_Pb + self.bess["ramp"]
            if Pb < Pb_min or Pb > Pb_max:
                Pb = min(max(Pb, Pb_min), Pb_max)
                clamps["bess_ramp"] = {"min": Pb_min, "max": Pb_max, "applied": Pb}

        # Split net power
        Pdis = max(Pb, 0.0)
        Pch  = max(-Pb, 0.0)

        # ---- Energy window caps (DoD) ----
        if Pdis > 0.0:
            Pdis_cap = self.bess["eta_d"] * max(self.E_meas - self.bess["E_min"], 0.0) / max(dt_h, 1e-9)
            if Pdis > Pdis_cap:
                Pdis = Pdis_cap
                clamps["bess_energy_min"] = Pdis_cap
        if Pch > 0.0:
            Pch_cap = max(self.bess["E_max"] - self.E_meas, 0.0) / (self.bess["eta_c"] * max(dt_h, 1e-9))
            if Pch > Pch_cap:
                Pch = Pch_cap
                clamps["bess_energy_max"] = Pch_cap
        Pb_eff = Pdis - Pch

        # Initial balance with MPC X's
        served = load0 * (1.0 - XL)
        shed   = load0 - served
        usedpv = pv0   * (1.0 - XPV)
        curt   = pv0   - usedpv

        Pgrid_in = 0.0
        Pgrid_out = 0.0

        supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
        ref_line = served + Pch
        residual = ref_line - supply

        # ---------------- OFF-GRID SAFETY ACTIONS ----------------
        if self.mode == "offgrid":
            if residual > self.tol_kw:
                # deficit: shed as needed (BESS already step-capped)
                needed = residual
                dXL = min(1.0 - XL, _safe_div(needed, load0))
                if dXL > 0:
                    XL += dXL
                    served = load0 * (1.0 - XL)
                    shed   = load0 - served
                    clamps["offgrid_autofix_shed"] = {"dXL": dXL}
                supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply

            elif residual < -self.tol_kw:
                # surplus: curtail as needed (Pch already step-capped)
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

        # ------------------- ON-GRID ACTIONS ---------------------
        else:  # "ongrid"
            # outage + import cap from EDS
            outage = (random.random() < self.grid_caps["outage_prob"]) if self.grid_caps["outage_prob"] > 0 else False
            cap_imp = 0.0 if outage else max(0.0, self.grid_caps["P_import_max"])

            if residual > self.tol_kw:
                take = min(residual, cap_imp)
                if take > 0:
                    Pgrid_in += take
                    residual -= take
                    clamps["grid_import"] = {"kW": take, "outage": outage}

            if residual > self.tol_kw:
                # still deficit -> shed
                dXL = min(1.0 - XL, _safe_div(residual, load0))
                if dXL > 0:
                    XL += dXL
                    served = load0 * (1.0 - XL)
                    shed   = load0 - served
                    clamps["ongrid_autofix_shed"] = {"dXL": dXL}
                supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
                ref_line = served + Pch
                residual = ref_line - supply

            if residual < -self.tol_kw:
                # surplus -> curtail
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

        if abs(residual) <= self.tol_kw:
            residual = 0.0

        # ---- integrate battery energy (applied values) ----
        E_next = self.E_meas + dt_h * (self.bess["eta_c"] * Pch - (1.0 / self.bess["eta_d"]) * Pdis)
        E_next = min(max(E_next, self.bess["E_min"]), self.bess["E_max"])
        soc_pct = 100.0 * (E_next / self.bess["E_nom"])
        if self.clamp_soc_pct:
            soc_pct = min(max(soc_pct, 0.0), 100.0)

        # ---- costs (TOU grid + penalties) ----
        tou = 0.0
        if self.mode == "ongrid":
            key = _hour_key(pd.Timestamp(t0))
            tou = float(self.costs["TOU"].get(key, 0.0))
        energy_grid_kwh = Pgrid_in * dt_h
        energy_shed_kwh = shed * dt_h
        energy_curt_kwh = curt * dt_h
        cost_grid = tou * energy_grid_kwh
        cost_shed = self.costs["c_shed"] * energy_shed_kwh
        cost_curt = self.costs["c_curt"] * energy_curt_kwh
        cost_total = cost_grid + cost_shed + cost_curt

        # ---- build row ----
        row = {
            "timestamp": pd.Timestamp(t0),

            # commanded (for audit)
            "cmd_P_bess_kw": float(P_bess_kw),
            "cmd_X_L": XL_cmd, "cmd_X_PV": XPV_cmd,
            "cmd_P_bess_kw_after_noise": Pb_after_noise,

            # applied (enforced)
            "Load_kw": load0, "PV_kw": pv0,
            "Load_served_kw": served, "Shedding_kw": shed,
            "PV_used_kw": usedpv, "Curtailment_kw": curt,
            "P_bess_kw": Pb_eff, "P_bess_discharge_kw": Pdis, "P_bess_charge_mag_kw": Pch,
            "P_grid_in_kw": Pgrid_in, "P_grid_out_kw": Pgrid_out,

            # balance
            "Total_supply_kw": usedpv + Pdis + Pgrid_in - Pgrid_out,
            "Target_kw": served + Pch,
            "Residual_kw": residual,

            # battery
            "E_kwh": E_next, "SoC_pct": soc_pct,

            # costs
            "TOU_cperkwh": tou,
            "cost_grid": cost_grid, "cost_shed": cost_shed, "cost_curt": cost_curt, "cost_total": cost_total,

            # misc
            "obj": (None if obj is None else float(obj)),
            "mode": self.mode,
            "clamps": clamps,
        }

        # ---- advance environment ----
        self._rows.append(row)
        self.E_meas = E_next
        self._prev_Pb = Pb_eff
        self.start_dt0 = times[1]
        self.iter_k += 1

        # clear active window
        self._last_times = None
        self._last_forecasts = None

        # debug
        self._log(
            f"iter={self.iter_k:04d} t0={pd.Timestamp(t0)} "
            f"Pb_cmd={P_bess_kw:.3f} -> Pb_eff={Pb_eff:.3f} (ch={Pch:.3f}, dis={Pdis:.3f}) "
            f"XL={XL:.3f} XPV={XPV:.3f} GridIn={Pgrid_in:.3f} "
            f"SoC={soc_pct:.2f}% residual={residual:.4f} cost={cost_total:.3f} clamps={bool(clamps)}"
        )

        return row, self.done()

    # --------------------------- exports ---------------------------

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows).set_index("timestamp").sort_index()
