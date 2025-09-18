import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import pandas as pd

class BatteryEnergyManagement:
    def __init__(self, horizon, data_path):
        with open(data_path, 'r') as f:
            cfg = json.load(f)
        self.horizon   = horizon
        self.Pmax      = cfg["BESS"]["Pmax"]
        self.Emax      = cfg["BESS"]["Emax"]
        self.eff       = cfg["BESS"]["eff"]
        self.DoD       = cfg["BESS"]["DoD"]
        self.min_soc   = self.Emax * (1 - self.DoD)
        self.dt1       = cfg["timestep_1"]    # minutes
        self.dt2       = cfg["timestep_2"]    # minutes
        self.outage_h  = cfg["EDS"]["outage"] # hours
        self.model     = None

    def _create_model(self, time_index):
        m = pyo.ConcreteModel()
        m.T = pyo.Set(initialize=time_index, ordered=True)

        m.pv          = pyo.Param(m.T, mutable=True, initialize=0)
        m.demand      = pyo.Param(m.T, mutable=True, initialize=0)
        m.grid_status = pyo.Param(m.T, mutable=True, initialize=1)

        m.battery_max_power = self.Pmax
        m.efficiency        = self.eff
        m.initial_soc       = pyo.Param(initialize=self.Emax, mutable=True)

        times  = list(time_index)
        dt_map = {t: (self.dt1/60 if i == 0 else self.dt2/60)
                  for i, t in enumerate(times)}
        m.dt_hour = pyo.Param(m.T, initialize=dt_map)

        m.EBESS   = pyo.Var(m.T, bounds=(self.min_soc, self.Emax))
        m.PBESS_c = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.PBESS_d = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.PEDSp   = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.PEDSn   = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.XLoad   = pyo.Var(m.T, within=pyo.NonNegativeReals)

        def soc_balance(m, t):
            idx      = times.index(t)
            prev_soc = m.initial_soc if idx == 0 else m.EBESS[times[idx - 1]]
            dt       = self.dt1 if idx == 0 else self.dt2
            return m.EBESS[t] == prev_soc + dt * (
                m.PBESS_c[t] * m.efficiency - m.PBESS_d[t] / m.efficiency)
        m.soc_balance = pyo.Constraint(m.T, rule=soc_balance)

        def power_balance(m, t):
            return (
                m.pv[t] + m.PBESS_d[t] + m.PEDSp[t]
                == m.demand[t] + m.PBESS_c[t] + m.PEDSn[t] + m.XLoad[t]
            )
        m.power_balance = pyo.Constraint(m.T, rule=power_balance)

        m.charge_limit    = pyo.Constraint(m.T, rule=lambda m, t: m.PBESS_c[t] <= self.Pmax)
        m.discharge_limit = pyo.Constraint(m.T, rule=lambda m, t: m.PBESS_d[t] <= self.Pmax)

        def outage_limit(m):
            return sum((1 - m.grid_status[t]) * m.dt_hour[t] for t in m.T) <= self.outage_h
        m.outage_limit = pyo.Constraint(rule=outage_limit)

        self.model = m

    def optimize(self, time_index, pv_fc, dem_fc, grid_stat, init_soc):
        self._create_model(time_index)
        m = self.model
        m.initial_soc = init_soc

        for t in m.T:
            m.pv[t]          = pv_fc[t]
            m.demand[t]      = dem_fc[t]
            m.grid_status[t] = grid_stat[t]

        res = pyo.SolverFactory('gurobi').solve(m, tee=False)
        if (res.solver.status == pyo.SolverStatus.ok and
            res.solver.termination_condition == pyo.TerminationCondition.optimal):
            return {
                t: {
                    'EBESS':   pyo.value(m.EBESS[t]),
                    'PBESS_c': pyo.value(m.PBESS_c[t]),
                    'PBESS_d': pyo.value(m.PBESS_d[t]),
                    'PEDSp':   pyo.value(m.PEDSp[t]),
                    'PEDSn':   pyo.value(m.PEDSn[t]),
                    'XLoad':   pyo.value(m.XLoad[t]),
                } for t in m.T
            }
        return None

if __name__ == "__main__":
    start_date    = datetime(2006, 12, 17, 0, 0)
    horizon       = 24
    data_path     = "data/parameters.json"

    pv_df   = pd.read_csv("data/pv_5min.csv", parse_dates=["datetime"], index_col="datetime")
    load_df = pd.read_csv("data/load_5min.csv", parse_dates=["datetime"], index_col="datetime")

    with open(data_path, 'r') as f:
        cfg = json.load(f)
    current_soc = cfg["BESS"]["Emax"]
    grid_mode   = "off_grid"
    status_val  = 1 if grid_mode == "on_grid" else 0

    bem = BatteryEnergyManagement(horizon, data_path)

    end_date_global = pv_df.index[-1]
    while start_date < end_date_global:
        window = pv_df.loc[start_date:].head(horizon).index
        if len(window) < horizon:
            break

        pv_fc     = {t: pv_df.at[t, "value"] for t in window}
        dem_fc    = {t: load_df.at[t, "value"] for t in window}
        grid_stat = {t: status_val for t in window}

        sol = bem.optimize(window, pv_fc, dem_fc, grid_stat, current_soc)
        print(f"\nWindow starting {start_date}:")
        if sol:
            for t, v in sol.items():
                print(f"  {t}: {v}")
            current_soc = sol[window[-1]]['EBESS']
        else:
            print("  No solution found.")
            break

        start_date += timedelta(minutes=bem.dt2)
