# Residential PV--BESS planning and operation

Planning and operation of a residential **PV + battery (BESS) system** that
must ride through grid outages. The project couples three layers:

1. an **environment** that simulates the physical system step by step (the
   "plant"),
2. a **sizing** model that chooses PV and BESS capacities under uncertainty and
   multi-year degradation, and
3. a **rolling-horizon MPC** that operates the residential system online, fed by different
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

**Power balance.** The command is reconciled against the actual load/PV
following a physical merit order: grid import up to its cap, then interrupting
battery charging, then dispatching available battery discharge (bounded by
power, ramp and stored energy) — load shedding, at value-of-lost-load, is the
last resort. Surpluses go to grid export and then PV curtailment.

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
- **Cyclic representative days**: the final-period BESS action is propagated
  back to a common optimized initial energy. This prevents free daily SoC
  resets while allowing the schedule to reserve energy for late outages.
- **Positive-probability scenarios only**: zero-probability load--PV pairs are
  removed before model construction, avoiding degenerate operating variables.
- **BESS Extn-LP (Pozo et al., Formulation 5)**: charge/discharge exclusivity is
  enforced *without binaries*, via an affine coupling plus energy caps — the
  model stays a pure LP. Full model: [`sizing/README.tex`](sizing/README.tex).

### How to run

```bash
python 1-sizing.py
```

Runs the 0%, 50%, and 100% outage-service cases with and without degradation.
It writes decision variables, yearly capacity, cost breakdowns, and
Gulliver-font reports to `outputs/sizing-sweep/<degradation>/<case>/`.
The fast LP method (barrier/IPM without crossover) is selected automatically for
this large model.

---

## 3. Operation: MPC and forecasts — [`opt/`](opt/), [`forecasting/`](forecasting/)

The residential system is operated online by a **rolling-horizon MPC**
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

$$\min \sum_{t}\sum_{c} \pi_c\,\Delta t_t \left( c^{\mathrm{shed}} P^{\mathrm{L}}_t X^{\mathrm{L}}_{t,c} + c^{\mathrm{curt}} P^{\mathrm{PV}}_t X^{\mathrm{PV}}_{t,c} + c^{\mathrm{grid}}_t P^{\mathrm{in}}_{t,c} + c^{\mathrm{deg}} \left(P^{\mathrm{ch}}_{t,c}+P^{\mathrm{dis}}_{t,c}\right) \right)$$

The BESS uses the same **Pozo Extn-LP** relaxation as sizing, so with
`relaxation=True` the MPC is a **pure LP** (no binaries). The degradation term
prices bidirectional throughput rather than absolute net power, preventing an
objective-neutral charge/discharge loop. Every solved interval and contingency
branch is audited; the maximum overlap is stored as
`max_simultaneous_ch_dis_kw` in `metrics.json`, and the corrected pipeline
rejects cases above $10^{-6}$ kW.

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

The analog-day forecaster reuses the *same* DTW cluster prototypes as the sizing
scenarios, which makes the prototype-MPC the *certainty-equivalent* of the
stochastic plan — comparing the two measures the value of keeping the full
distribution.

### Driving a simulation

Two reusable drivers in [`opt/operation.py`](opt/operation.py) wrap the loop and
save artifacts (`parameters_used.json`, `outage_calendar.json`,
`operation_final.parquet`, `metrics.json`):

```python
from opt import load_sized_parameters, simulate_baseline, simulate_mpc, simulate_stochastic
from forecasting import PerfectForecast
from opt import SelfConsumptionController

params = load_sized_parameters()  # uses the versioned reference sizing artifact
simulate_mpc(params, PerfectForecast(load_s, pv_s), start, n_iters, out_dir)
simulate_stochastic(params, start, n_iters, out_dir)  # receding plan, re-solved daily
simulate_baseline(params, SelfConsumptionController(), start, n_iters, out_dir)
```

Four causal rule-based references are implemented as independent modules in
`opt/`: `reserve_only`, `self_consumption`, `load_shifting`, and
`peak_shaving`. They use the same physical environment and islanded emergency
control as the MPC. The first keeps the BESS idle while connected to the grid;
it is therefore a reserve-only policy, not a system without storage.

### Experiments

The repository has two explicit experiment entry points. Operation requires a
traceable sizing artifact, but the sizing optimization does not need to run on
the operational machine.

| Script | Purpose |
|---|---|
| [`1-sizing.py`](1-sizing.py) | 0%, 50%, and 100% outage-service sizing, with and without degradation |
| [`2-operation.py`](2-operation.py) | causal, versioned, and resumable operation of the three degradation-aware sizing cases |

The operational pipeline writes resumable cases under
`outputs/operation-sweep/with-degradation/<sizing-case>/` and continuously
updates `summary.csv` and `pipeline_state.json`. Figures and tables are rebuilt from the summaries by
`scripts/make_figures.py` and `scripts/make_tables.py`.

The corrected campaign uses a fixed physical outage support, hazard-based
scenario weights, complete pre-outage non-anticipativity, a 5-min ramp
reference, terminal-energy closure, and causal hourly LSTM inputs assembled
only from fully completed observation hours. Its temporal-mesh sweep evaluates
ideal, prototype-prefix, and LSTM forecasts. The ideal forecast is retained as
a perfect-information target but does not participate in mesh selection.
Candidate meshes are selected from the Pareto front using prototype and LSTM
relative regret together with solve time. The annual campaign evaluates all 81
mesh--forecaster
configurations for 10 days in each of the 12 months of the test year. Each
month uses a distinct outage seed shared by all configurations, preserving
common-random-number comparisons. The completed May sweep is reused only
within the same model version, leaving 891 additional monthly runs. Superseded
runs remain local under `.old/`; only audited scientific artifacts
are promoted to `paper/`. Distributed campaigns use `outputs/sweeps/` and must
pass the same audits before promotion.

Large trajectories and stochastic plans remain under ignored `outputs/` and
are not committed to Git. Every operational output root stores copies of the
base parameters, sizing artifact, and materialized operational parameters
under `campaign-inputs/`.

Run the complete sizing and operation campaigns with:

```bash
python 1-sizing.py
python 2-operation.py
```

The constants at the beginning of each script expose the intended campaign
configuration without command-line parsers. `1-sizing.py` compares degradation
assumptions for planning, while `2-operation.py` uses only the three
degradation-aware sizing artifacts. The operational stages are
`causal-pilot`, `mesh`, `annual-mesh`, `forecast`, `recourse`, and
`robustness`; each stage continuously updates `summary.csv` and
`pipeline_state.json`, including elapsed time and an ETA.

The peak-shaving threshold is fixed from training net load; no test-period
future value is used by any baseline.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # includes HiGHS and Parquet
# Optional licensed solver: pip install gurobipy
```

Gurobi is used if licensed (set `GRB_LICENSE_FILE`); otherwise the solver falls
back to HiGHS automatically.
