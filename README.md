# nanogrid

Planning and operation of a residential **PV + battery (BESS) nanogrid** that
must ride through grid outages. The project couples three layers:

1. an **environment** that simulates the physical nanogrid step by step (the
   "plant"),
2. a **sizing** model that chooses PV and BESS capacities under uncertainty and
   multi-year degradation, and
3. a **rolling-horizon MPC** that operates the nanogrid online, fed by different
   **forecasters** (perfect, LSTM, analog-day prototypes, hybrid) and compared
   against an offline stochastic plan.

All scalar parameters live in [`data/parameters.json`](data/parameters.json).
Optimization models are built with Pyomo and solved with Gurobi when a license
is available, falling back automatically to the open-source HiGHS solver
(see [`opt/utils.py`](opt/utils.py) `solve_model`).

---

## 1. Environment — [`env/grid_env.py`](env/grid_env.py)

`GridEnv` is the closed-loop **plant**: it advances real load/PV series at a
fixed 5-minute step, applies the controller's command, enforces the physical
limits the optimizer may have approximated, and logs everything. It is what the
MPC and the stochastic plan are *evaluated on*.

### How it works

**BESS state of charge.** Each step updates the stored energy with charge/
discharge efficiencies:

$$E_{t+1} = E_t + \Delta t \left( \eta_c\, P^{\mathrm{ch}}_t - \tfrac{1}{\eta_d}\, P^{\mathrm{dis}}_t \right),\qquad (1-\mathrm{DoD})\,E_{\mathrm{nom}} \le E_t \le E_{\mathrm{nom}}$$

**Power balance.** The command is reconciled against the actual load/PV; any
residual is covered by the grid (on-grid) or by shedding/curtailment, in 10%
steps, until balanced.

**Two operating modes**, driven by the outage state:
- *On-grid*: the controller's command is applied; the grid absorbs the residual.
- *Off-grid (islanded)*: no grid exchange — the BESS is forced to follow the
  exact deficit/surplus $P^{\mathrm{des}} = \text{Load} - \text{PV}$, limited by
  ramp, power and energy caps; unmet load is shed.

**Outage calendar.** Outages are *pre-generated* from a single seed
(`EDS.seed`), so every controller and every sweep case sees the **same events**
(common random numbers). One Bernoulli draw per day decides whether an outage
occurs (`outage_probability_pct`); its start is uniform within the day and its
duration is **lognormal** with mean `outage_duration_hours` and coefficient of
variation `outage_duration_std_dev_frac`. The calendar is written to
`outage_calendar.json` for auditability.

**Actuator noise (optional).** With `BESS.noisy = true`, the applied BESS power
is perturbed on-grid only:

$$\tilde P_t = P_t + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\sigma),\qquad \sigma = \sigma_{kW} + \sigma_{\mathrm{frac}}\,|P_t|$$

seeded by `BESS.noise.seed`. This is what makes closed-loop MPC (which re-measures
the SoC each step) differ from an open-loop plan.

**Cost accounting.** Per step, in the run currency:

$$\text{cost}_t = c^{\mathrm{grid}}_t\, E^{\mathrm{grid,in}}_t + c^{\mathrm{shed}}\, E^{\mathrm{shed}}_t + c^{\mathrm{curt}}\, E^{\mathrm{curt}}_t$$

with `c_grid` from the time-of-use map (`costs.EDS`).

### How to use

```python
from env.grid_env import GridEnv
import json, pandas as pd

params = json.load(open("data/parameters.json", encoding="utf-8"))
env = GridEnv(params, "data/load_5min_test.csv", "data/pv_5min_test.csv",
              start_dt0=pd.Timestamp("2009-05-01"), n_iters=2880)
env.save_outage_calendar("outage_calendar.json")

while not env.done():
    row, done = env.step(P_bess_kw=..., X_L=..., X_PV=...)  # controller command

df = env.to_dataframe()   # full operation log
```

Key attributes the controllers read: `env.timestamp`, `env.mode`, `env.E_meas`
(measured SoC), `env.dt_min` (the MPC time mesh), `env.load_kw_s`/`env.pv_kw_s`
(scaled series), `env.pv_scaling`/`env.load_scaling`.

---

## 2. Sizing — [`1-sizing.py`](1-sizing.py), [`sizing/`](sizing/)

A **two-stage stochastic program** that picks the PV peak power
$\hat P^{\mathrm{PV}}$ and BESS energy $\hat E^{\mathrm{BESS}}$ minimizing capital
cost plus the present value of operation over a 25-year horizon, while the
system must survive grid contingencies. Scenarios come from DTW clusters of
historical load/PV days (in [`data/sizing/`](data/sizing)), weighted by their
joint probabilities — the *same* scenarios the operation layer uses.

### Objective

$$\min\; \underbrace{C^{\mathrm{PV}}\hat P^{\mathrm{PV}} + C^{\mathrm{BESS}}\hat E^{\mathrm{BESS}}}_{\text{CAPEX}} \; + \; \underbrace{\sum_{y=1}^{N_Y} \frac{\text{OPEX}^{\mathrm{annual}}}{(1+r)^y}}_{\text{NPV of OPEX}}$$

where the expected daily OPEX sums grid import, load shedding, PV curtailment
and BESS degradation over scenarios $s$ and contingencies $c$.

### What makes it specific

- **Multi-year degradation.** The available BESS energy fades year over year
  (calendar + cyclic), and PV output degrades per its datasheet:

  $$E^{\mathrm{year}}_{y+1} = E^{\mathrm{year}}_y - \alpha_{\mathrm{cal}}\,E^{\mathrm{year}}_y - \alpha_{\mathrm{cyc}}\,N_d\,\Theta_y, \qquad \alpha_{\mathrm{cyc}} = \frac{1-\gamma_{\mathrm{EOL}}}{2\,N_{\mathrm{cycle}}}$$

  with $\Theta_y$ the expected daily throughput. Equations:
  [`equacoes_degradacao_bess.tex`](equacoes_degradacao_bess.tex).
- **Contingencies with non-anticipativity**: before an outage reveals itself,
  the contingent decisions must equal the base case `c0`.
- **BESS Extn-LP (Pozo et al., Formulation 5)**: charge/discharge exclusivity is
  enforced *without binaries*, via an affine coupling plus energy caps — the
  model stays a pure LP. Full model: [`sizing/README.tex`](sizing/README.tex).

### How to run

```bash
python 1-sizing.py
```

Runs two cases (with / without degradation) and writes decision variables,
yearly capacity, OPEX breakdown and operation CSVs to `Results/sizing/<case>/`.
The fast LP method (barrier/IPM without crossover) is selected automatically for
this large model.

---

## 3. Operation: MPC and forecasts — [`opt/`](opt/), [`forecasting/`](forecasting/)

The nanogrid is operated online by a **rolling-horizon MPC**
([`opt/ongrid.py`](opt/ongrid.py)): at every on-grid step it builds a short
optimization over the prediction horizon, solves it, applies only the **first**
action to the environment, then slides forward and re-solves with the freshly
measured SoC.

### Non-uniform time mesh

The horizon uses a **two-resolution mesh** (`opt/utils.py::build_dt_vector`):
a fine step `timestep_1_min` over a near-term window equal to the contingency
duration, then a coarse step `timestep_2_min` for the rest of the horizon. This
keeps resolution where it matters (riding through an outage) while shrinking the
problem — and is the axis swept in experiment 5.

### Model

Power balance per step $t$ and contingency $c$ (shared control before the
outage, via non-anticipativity):

$$P^{\mathrm{PV}}_t(1-X^{\mathrm{PV}}_{t,c}) + P^{\mathrm{dis}}_{t,c} - P^{\mathrm{ch}}_{t,c} + P^{\mathrm{in}}_{t,c} - P^{\mathrm{out}}_{t,c} = P^{\mathrm{L}}_t(1-X^{\mathrm{L}}_{t,c})$$

Objective (expected operating cost over the horizon):

$$\min \sum_{t}\sum_{c} \pi_c\,\Delta t_t \left( c^{\mathrm{shed}} P^{\mathrm{L}}_t X^{\mathrm{L}}_{t,c} + c^{\mathrm{curt}} P^{\mathrm{PV}}_t X^{\mathrm{PV}}_{t,c} + c^{\mathrm{grid}}_t P^{\mathrm{in}}_{t,c} + c^{\mathrm{deg}} |P^{\mathrm{bess}}_{t,c}| \right)$$

The BESS uses the same **Pozo Extn-LP** relaxation as sizing, so with
`relaxation=True` the MPC is a **pure LP** (no binaries). The relaxation is exact
here — simultaneous charge/discharge never appears at the optimum — and the
simulators audit this (`max_simultaneous_ch_dis_kw` in `metrics.json`).

### How forecasts enter

At each step the MPC asks a **forecaster** for the load/PV window aligned to the
time mesh. Every forecaster exposes the same interface:

```python
forecaster.get_forecasts(start_dt0, intervals=None, dt_min=env.dt_min,
                         include_actuals=False)
# -> {"load_kw": {ts: kW, ...}, "pv_kw": {ts: kW, ...}}
```

Interchangeable implementations:

| Forecaster | Module | Idea |
|---|---|---|
| `PerfectForecast` | `forecasting/prototype_forecast.py` | oracle — returns the actual future (upper bound) |
| `ForecastMPC` | `forecasting/get_forecasting.py` | LSTM (VSTF + day-ahead); loads TensorFlow |
| `PrototypeForecast` | `forecasting/prototype_forecast.py` | **analog day**: pick the current day's cluster prototype (`calendar` / `prefix` / `knn`) and use it as the long-horizon forecast |
| `HybridForecast` | `forecasting/prototype_forecast.py` | LSTM on the fine steps, prototype on the coarse steps |

The analog-day forecaster reuses the *same* DTW cluster prototypes as the sizing
scenarios, which makes the prototype-MPC the *certainty-equivalent* of the
stochastic plan — comparing the two measures the value of keeping the full
distribution.

### Driving a simulation

Two reusable drivers in [`opt/operation.py`](opt/operation.py) wrap the loop and
save artifacts (`parameters_used.json`, `outage_calendar.json`,
`operation_final.csv`, `metrics.json`):

```python
from opt import simulate_mpc, simulate_stochastic
from forecasting import PerfectForecast

simulate_mpc(params, PerfectForecast(load_s, pv_s), start, n_iters, out_dir)
simulate_stochastic(params, start, n_iters, out_dir)  # solve once, run open-loop
```

### Experiments

| Script | Purpose |
|---|---|
| [`2-forecast_eval.py`](2-forecast_eval.py) | offline forecast quality (prototypes vs naive vs LSTM); resumable via checkpoint |
| [`3-forecaster_comparison.py`](3-forecaster_comparison.py) | operation: ideal vs stochastic vs LSTM vs prototype vs hybrid |
| [`4-sized_system.py`](4-sized_system.py) | same comparison on the *sized* system (closes the planning → operation loop) |
| [`5-mesh_sweep.py`](5-mesh_sweep.py) | sweep of the time mesh $(h, \Delta t_1, \Delta t_2)$ |
| [`6-robustness.py`](6-robustness.py) | actuator noise, outage probability and seeds |

Each writes a `summary.csv` under `Results/<experiment>/`. Configuration is
plain constants at the top of each script — no CLI flags.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-exp.txt                 # pyomo, gurobipy, highspy, tensorflow, ...
```

Gurobi is used if licensed (set `GRB_LICENSE_FILE`); otherwise the solver falls
back to HiGHS automatically. For running the campaign on a remote/cloud machine
see [`scripts/EC2.md`](scripts/EC2.md) and
[`scripts/TMUX_GUROBI.md`](scripts/TMUX_GUROBI.md).
