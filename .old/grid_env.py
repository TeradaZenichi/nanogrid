import math
import random
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from opt.utils import build_dt_vector, load_series_scaled


def _get_nested(d: Dict[str, Any], path_list: List[str], default: Any = None):
    """Accesses a value from a JSON based on a provided path, keeping the original structure."""
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
    """Resolves the timestep vector based on the original (unflattened) JSON."""
    for key in ["dt_vector", "dt_minutes", "dt_min_vector", "dt_min_list"]:
        if key in params and isinstance(params[key], (list, tuple)):
            vec = params[key]
            if all(isinstance(x, (int, float)) for x in vec):
                return list(vec)

    horizon_hours = _get_nested(params, ["time.horizon_hours", "horizon_hours", "mpc.horizon_hours"], 24)
    outage_hours  = _get_nested(params, ["EDS.outage_duration_hours", "outage_duration_hours", "time.outage_duration_hours"], 0)
    t1 = _get_nested(params, ["time.timestep_1_min", "timestep_1_min", "mpc.timestep_1_min", "timestep_min"], 5)
    t2 = _get_nested(params, ["time.timestep_2_min", "timestep_2_min", "mpc.timestep_2_min", "dt_min2"], t1)

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
    """Extracts Load and PV Pmax directly from the JSON (unflattened)."""
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
        outage_prob_daily = self.p.get("EDS", {}).get("outage_probability_pct", 0.0) / 100.0
        timestep_min = self.dt_h * 60
        self.outage_prob_per_step = outage_prob_daily * (timestep_min / (24 * 60))
        self.start_dt0 = pd.Timestamp(start_dt0)

        # --- NEW CONTINGENCY LOGIC ---
        EDS = self.p.get("EDS", {})
        self.mean_outage_duration_h = float(EDS.get("outage_duration_hours", 4.0)) / 2.0
        std_dev_frac = float(EDS.get("outage_duration_std_dev_frac", 0.3))
        self.std_dev_outage_duration_h = self.mean_outage_duration_h * std_dev_frac
        
        self.outage_active = False
        self.outage_end_time = None
        self.outage_triggered_today = False
        self.current_day = None
        # --- END OF NEW LOGIC ---

        scaling = _derive_scaling_params(self.p)
        self.load_kw_s, self.pv_kw_s = load_series_scaled(scaling, self.load_csv, self.pv_csv)

        self.dt_min = _resolve_dt_vector(self.p)
        self._parse_constraints_and_costs()
        self._rows: List[Dict[str, Any]] = []

        self.reset()

    def _parse_constraints_and_costs(self):
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
        self._rng = random.Random(self.noise["seed"]) if self.noise["enabled"] and self.noise["seed"] is not None else random
        self._rng_outage = random.Random(self.noise["seed"]) if self.noise["seed"] is not None else random

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
        self.iter_k = 0
        self.timestamp = self.start_dt0
        if self.bess["E_init"] is not None:
            E0 = float(self.bess["E_init"])
        else:
            E0 = self.bess["E_nom"] * max(self.bess["soc_min"], 0.5)
        self.E_meas = min(max(E0, self.bess["E_min"]), self.bess["E_max"])
        self._prev_Pb = 0.0
        
        # --- RESET CONTINGENCY FLAGS ---
        self.outage_active = False
        self.outage_end_time = None
        self.current_day = self.start_dt0.date()
        self.outage_triggered_today = False
        self.mode = "ongrid" # Always start in on-grid mode
        # --- END OF RESET ---
        
        self._log(
            "reset: start=%s, E=%.3f kWh (E_min=%.3f, E_max=%.3f, Pmax=%s, ramp=%s)"
            % (self.start_dt0, self.E_meas, self.bess["E_min"], self.bess["E_max"],
               ("inf" if math.isinf(self.bess["P_max"]) else f"{self.bess['P_max']:.3f}"),
               str(self.bess["ramp"]))
        )

    def _update_outage_status(self):
        """New method to manage the contingency state at each step."""
        if self.timestamp.date() > self.current_day:
            self._log(f"New day detected ({self.timestamp.date()}). Resetting daily outage flag.")
            self.current_day = self.timestamp.date()
            self.outage_triggered_today = False

        if self.outage_active:
            if self.timestamp >= self.outage_end_time:
                self._log(f"Outage period finished. Returning to on-grid mode at {self.timestamp}.")
                self.outage_active = False
                self.outage_end_time = None
                self.mode = "ongrid"
            else:
                self.mode = "offgrid"
            return

        if not self.outage_active and not self.outage_triggered_today:
            if self._rng_outage.random() < self.outage_prob_per_step:
                self.outage_active = True
                self.outage_triggered_today = True
                self.mode = "offgrid"
                
                duration_h = self._rng_outage.gauss(self.mean_outage_duration_h, self.std_dev_outage_duration_h)
                duration_h = max(self.dt_h, duration_h) # Must last at least one timestep
                
                self.outage_end_time = self.timestamp + pd.Timedelta(hours=duration_h)
                self._log(f"--- OUTAGE TRIGGERED at {self.timestamp} ---")
                self._log(f"Duration: {duration_h:.2f} hours. Expected end time: {self.outage_end_time}.")
            else:
                self.mode = "ongrid"
        else:
            self.mode = "ongrid"
            
    def done(self) -> bool:
        return self.iter_k >= self.n_iters

    def step(
        self,
        P_bess_kw: float,
        X_L: Optional[float],
        X_PV: Optional[float],
        obj: Optional[float] = None,
        exec_time_sec: Optional[float] = None
    ) -> Tuple[Dict[str, Any], bool]:
        
        # A chamada para _update_outage_status foi removida do início.
        # A lógica agora prossegue usando o 'self.mode' que foi determinado no final do passo anterior.

        load0 = self.load_kw_s.get(self.timestamp, 0.0)
        pv0   = self.pv_kw_s.get(self.timestamp, 0.0)
        
        XL_cmd = 0.0 if X_L is None else float(X_L)
        XPV_cmd = 0.0 if X_PV is None else float(X_PV)
        XL = min(max(XL_cmd, 0.0), 1.0)
        XPV = min(max(XPV_cmd, 0.0), 1.0)
        clamps: Dict[str, Any] = {}
        if XL != XL_cmd or XPV != XPV_cmd:
            clamps["fractions"] = {"X_L": XL, "X_PV": XPV}

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

        Pb = Pb_after_noise
        if abs(Pb) > self.bess["P_max"]:
            Pb = max(min(Pb, self.bess["P_max"]), -self.bess["P_max"])
            clamps["bess_pmax"] = Pb
        if self.bess["ramp"] is not None:
            Pb_min = self._prev_Pb - self.bess["ramp"]
            Pb_max = self._prev_Pb + self.bess["ramp"]
            if Pb < Pb_min or Pb > Pb_max:
                Pb = min(max(Pb, Pb_min), Pb_max)
                clamps["bess_ramp"] = {"min": Pb_min, "max": Pb_max, "applied": Pb}

        Pdis = max(Pb, 0.0)
        Pch  = max(-Pb, 0.0)

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

        served = load0 * (1.0 - XL)
        shed   = load0 - served
        usedpv = pv0   * (1.0 - XPV)
        curt   = pv0   - usedpv

        Pgrid_in = 0.0
        Pgrid_out = 0.0
        supply   = usedpv + Pdis + Pgrid_in - Pgrid_out
        ref_line = served + Pch
        residual = ref_line - supply

        if self.mode == "offgrid":
            if residual > self.tol_kw:
                needed_kw = residual
                shed_step_kw = 0.10 * load0
                if shed_step_kw > 1e-6:
                    num_steps_needed = math.ceil(needed_kw / shed_step_kw)
                    new_total_XL = num_steps_needed * 0.10
                    new_total_XL = min(1.0, new_total_XL)
                    if new_total_XL > XL:
                        dXL = new_total_XL - XL
                        XL = new_total_XL
                        served = load0 * (1.0 - XL)
                        shed   = load0 - served
                        clamps["offgrid_autofix_shed"] = {"dXL": dXL, "new_XL_pct": XL * 100}
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
        else: # ongrid
            if residual > self.tol_kw:
                take = min(residual, self.grid_caps["P_import_max"])
                if take > 0:
                    Pgrid_in += take
                    residual -= take
                    clamps["grid_import"] = {"kW": take, "outage": False}
            if residual > self.tol_kw:
                needed_kw = residual
                shed_step_kw = 0.10 * load0
                if shed_step_kw > 1e-6:
                    num_steps_needed = math.ceil(needed_kw / shed_step_kw)
                    new_total_XL = num_steps_needed * 0.10
                    new_total_XL = min(1.0, new_total_XL)
                    if new_total_XL > XL:
                        dXL = new_total_XL - XL
                        XL = new_total_XL
                        served = load0 * (1.0 - XL)
                        shed   = load0 - served
                        clamps["ongrid_autofix_shed"] = {"dXL": dXL, "new_XL_pct": XL * 100}
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

        if abs(residual) <= self.tol_kw:
            residual = 0.0

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
            "cmd_P_bess_kw": float(P_bess_kw),
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
            "exec_time_sec": exec_time_sec
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
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows).set_index("timestamp").sort_index()