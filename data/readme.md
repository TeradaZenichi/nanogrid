# Parameter assumptions and sources

## 1. Scope and reproducibility

The experiments represent a synthetic residential energy system connected to a low-reliability distribution grid. Economic quantities are expressed in constant 2023 U.S. dollars. The case is not tied to a particular battery manufacturer or electricity utility. The executable values are available in the [`data/parameters.json` file](parameters.json).

Values described as *derived* are calculated from linked source data. Values described as *modeling assumptions* are author-defined choices rather than measurements attributed to an external source.

## 2. Economic parameters

| Definition | Base value | Unit | Source or status |
|---|---:|---|---|
| Residential PV CAPEX | 2,740 | USD/kW | [DOE/SETO 2024Q1 residential PV MSP](https://www.energy.gov/cmei/systems/solar-photovoltaic-system-cost-benchmarks) |
| Residential BESS CAPEX | 1,043 | USD/kWh | Derived from the [DOE/SETO residential PV and PV-plus-ESS benchmarks](https://www.energy.gov/cmei/systems/solar-photovoltaic-system-cost-benchmarks) |
| BESS replacement cost | 1,000 | USD/kWh | Rounded replacement proxy based on the [DOE/SETO incremental installed BESS cost](https://www.energy.gov/cmei/systems/solar-photovoltaic-system-cost-benchmarks) |
| Residential value of lost load | 5.00 | USD/kWh unserved | [DOE National Transmission Planning Study, Chapter 5](https://www.energy.gov/sites/default/files/2024-10/NationalTransmissionPlanningStudy-Chapter5.pdf) |
| PV curtailment penalty | 0.001 | USD/kWh | [Author-defined numerical tie-breaker](parameters.json), not an empirical damage cost |
| Real discount rate | 8% | year-1 | [Author-defined central case](parameters.json); financial-method context from the [NREL ATB](https://atb.nrel.gov/electricity/2024/2023/financial_cases_%26_methods) |
| Planning horizon | 25 | years | [Author-defined long-term horizon](parameters.json) |

### 2.1 PV and BESS capital costs

The central theoretical PV case adopts a capital cost of 1,600 USD/kW and fixed O&M of 12 USD/(kW year), following the microgrid resilience assumptions of [Anderson et al. (2021)](https://doi.org/10.1016/j.adapen.2021.100049). These values describe a literature-based prospective microgrid scenario rather than the observed installed cost of a small residential system.

For comparison, the [DOE/SETO 2024Q1 benchmark](https://www.energy.gov/cmei/systems/solar-photovoltaic-system-cost-benchmarks) reports an MSP of 2.74 USD/Wdc for an 8 kWdc residential PV system and 4.50 USD/Wdc for an 8 kWdc PV system coupled to a 13.5 kWh BESS. The source expresses all costs in 2023 USD. The DOE value is retained as the upper PV-cost sensitivity. Its incremental storage cost is

\[
c_{\mathrm{BESS}}^{\mathrm{cap}}
=\frac{8{,}000(4.50-2.74)}{13.5}
=1{,}042.96\ \mathrm{USD/kWh},
\]

which is rounded to 1,043 USD/kWh. The replacement proxy is rounded to 1,000 USD/kWh and is tested separately because installed-system and future replacement costs need not be identical.

### 2.2 Synthetic time-of-use tariff

The tariff is calibrated to the 2023 U.S. average residential price of 0.1600 USD/kWh reported by the [U.S. Energy Information Administration](https://www.eia.gov/electricity/annual/table.php?t=epa_02_04.html).

| Definition | Hours | Price [USD/kWh] | Source or status |
|---|---|---:|---|
| Off-peak period | 00:00-15:59 and 22:00-23:59 | 0.125 | [Synthetic period calibrated to the EIA average](https://www.eia.gov/electricity/annual/table.php?t=epa_02_04.html) |
| Intermediate period | 16:00-17:59 and 21:00-21:59 | 0.190 | [Synthetic period calibrated to the EIA average](https://www.eia.gov/electricity/annual/table.php?t=epa_02_04.html) |
| Peak period | 18:00-20:59 | 0.340 | [Synthetic period calibrated to the EIA average](https://www.eia.gov/electricity/annual/table.php?t=epa_02_04.html) |

The duration-weighted mean is

\[
\frac{18(0.125)+3(0.190)+3(0.340)}{24}
=0.160\ \mathrm{USD/kWh}.
\]

The hourly levels are author-defined and must not be described as the tariff of a specific utility.

### 2.3 Value of lost load

The base load-shedding coefficient is 5 USD/kWh, matching the residential VOLL used by the [DOE National Transmission Planning Study](https://www.energy.gov/sites/default/files/2024-10/NationalTransmissionPlanningStudy-Chapter5.pdf). Sensitivities of 2 and 10 USD/kWh are [configured author assumptions](parameters.json). A much larger coefficient, such as 1,000 USD/kWh, may be used only as a numerical feasibility penalty and must not be reported as empirical residential VOLL.

### 2.4 Export compensation

Exported energy is compensated at a fixed rate of 0.10 USD/kWh. Imports retain the time-of-use tariff defined above, so the model distinguishes the price paid for grid energy from the compensation received for surplus PV generation. The resilience sizing campaign varies the minimum served-load fraction during outages (0%, 50%, and 100%) while keeping this export compensation fixed.

## 3. Low-reliability grid stress test

| Definition | Base value | Source or status |
|---|---:|---|
| Daily interruption probability | 10% | [Author-defined stress assumption](parameters.json), benchmarked against [EIA reliability data](https://www.eia.gov/todayinenergy/detail.php?id=61303) |
| Probability reference window | 24 h | [Author-defined interpretation of the daily probability](parameters.json) |
| Mean interruption duration | 2 h | [Author-defined stress assumption](parameters.json), benchmarked against [EIA reliability data](https://www.eia.gov/todayinenergy/detail.php?id=61303) |
| Duration standard deviation | 50% of the mean | [Author-defined stochastic assumption](parameters.json) |
| Operational contingency spacing | 2 h | [Author-defined quadrature support](parameters.json), using the same spacing as the representative-day sizing model |

These assumptions imply

\[
N_{\mathrm{out}}=365(0.10)=36.5\ \mathrm{interruptions/year},
\]

\[
H_{\mathrm{out}}=36.5(2)=73\ \mathrm{h/year}.
\]

For context, the [EIA reports](https://www.eia.gov/todayinenergy/detail.php?id=61303) approximately 1.4 interruptions and 5.6 interruption-hours per U.S. customer in 2022 when major events are included. The adopted case is therefore an intentionally severe synthetic stress test rather than an estimate for a particular distribution company.

For operational optimization, the daily probability is converted into a homogeneous first-arrival hazard,

\[
\lambda=-\frac{\ln(1-p_{24})}{24}.
\]

For a candidate-start bin \([a_i,b_i)\), the outage-scenario weight is \(\pi_i=\exp(-\lambda a_i)-\exp(-\lambda b_i)\), and the no-outage weight is \(\pi_0=\exp[-\lambda(T^{\mathrm{hor}}-H^{\mathrm{out}})]\). Candidate starts are placed every 2 h in physical time and only complete 2 h interruptions are represented. Consequently, changing the optimization mesh does not change the physical support or redistribute a fixed probability among a different number of grid points. This is a [modeling convention encoded in the parameter file](parameters.json), not an empirical outage-arrival law.

## 4. Generic residential LFP BESS

The storage system is defined as a scalable generic residential LFP BESS rather than a commercial product. Its reference power, energy, chemistry, round-trip efficiency, use pattern, and technical life follow the [NREL 2024 Annual Technology Baseline for residential battery storage](https://atb.nrel.gov/electricity/2024/residential_battery_storage) and the associated [NREL technology definitions](https://atb.nrel.gov/electricity/2024b/definitions).

| Definition | Base value | Source or status |
|---|---:|---|
| Chemistry | LFP | [NREL residential battery benchmark](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Reference power | 5 kW | [NREL residential battery benchmark](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Reference energy | 12.5 kWh | [NREL residential battery benchmark](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Energy-to-power ratio | 2.5 h | Derived from the [NREL 5 kW/12.5 kWh system](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| C-rate | 0.4C | Derived from the [NREL 5 kW/12.5 kWh system](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Round-trip efficiency | 85% | [NREL residential battery benchmark](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Charge efficiency | 92.195% | Symmetric split of the [NREL round-trip efficiency](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Discharge efficiency | 92.195% | Symmetric split of the [NREL round-trip efficiency](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Design technical life | 15 years | [NREL technology definitions](https://atb.nrel.gov/electricity/2024b/definitions) |
| Reference use | Approximately one full equivalent cycle per day | [NREL residential battery capacity-factor assumption](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Full equivalent cycle life | 5,475 cycles | Derived as 365 cycles/year over the [15-year NREL design life](https://atb.nrel.gov/electricity/2024b/definitions) |
| End-of-life capacity | 80% of initial capacity | [NREL energy-storage performance convention](https://docs.nrel.gov/docs/fy21osti/77621.pdf) |
| Maximum DoD | 90% | [Author-defined operational limit](parameters.json), not used to derive the full-equivalent-cycle coefficient |
| Initial fallback SoC | 50% | [Author-defined initialization](parameters.json); sized experiments overwrite it with the sizing result |
| Net-power ramp per 5-min physical step | 5 kW | [Author-defined actuator limit and reference interval](parameters.json) |
| Terminal stored energy | At least the measured initial energy | [Author-defined rolling-horizon closure](parameters.json) used to suppress end-of-horizon depletion |

The per-direction efficiencies are obtained from

\[
\eta^{\mathrm{ch}}=\eta^{\mathrm{dis}}=\sqrt{0.85}=0.92195,
\]

so that their product matches the [85% NREL round-trip efficiency](https://atb.nrel.gov/electricity/2024/residential_battery_storage).

## 5. Equivalent-throughput degradation model

The linear model follows the energy-throughput representation examined by [Wankmuller et al. (2017)](https://www.osti.gov/pages/biblio/1393934), who model capacity decay as a function of processed energy and analyze an 80% EOL criterion. The coefficient is therefore named the *equivalent-throughput capacity-fade coefficient* rather than a universal electrochemical degradation rate.

### 5.1 Physical capacity-fade coefficient

The NREL use assumption corresponds to

\[
N^{\mathrm{FEC}}=365(15)=5{,}475\ \mathrm{full\ equivalent\ cycles}.
\]

Because a full equivalent cycle is already normalized by nominal energy, the 90% operational DoD is not multiplied into this conversion. With bidirectional throughput, the reference lifetime energy is

\[
Q^{\mathrm{life}}=2N^{\mathrm{FEC}}E^{\mathrm{BESS}}.
\]

Using the [80% NREL EOL convention](https://docs.nrel.gov/docs/fy21osti/77621.pdf), the linear fade coefficient is

\[
\alpha^{\mathrm{ET}}
=\frac{1-0.80}{2(5{,}475)}
=1.82648\times10^{-5}.
\]

The coefficient converts bidirectional energy throughput into loss of available energy capacity in the sizing model. The linear-throughput approximation is supported by [Wankmuller et al.](https://doi.org/10.1016/j.est.2016.12.004); it is not claimed to reproduce all LFP aging mechanisms.

### 5.2 Economic throughput cost

The operational MPC and stochastic benchmark do not propagate long-term state of health. They therefore use a marginal wear proxy derived from the [1,000 USD/kWh replacement assumption](parameters.json):

\[
c^{\mathrm{deg}}
=\frac{c^{\mathrm{rep}}}{2N^{\mathrm{FEC}}}
=\frac{1000}{2(5{,}475)}
=0.091324\ \mathrm{USD/kWh}.
\]

The factor of two is required because the implemented throughput is the sum of charging and discharging energy. This convention is consistent with the bidirectional energy-throughput definition used in the [configured model](parameters.json).

### 5.3 Calendar degradation

Calendar fade is zero in the base case because the [NREL ATB](https://atb.nrel.gov/electricity/2024/residential_battery_storage) provides an aggregate design life rather than a standalone annual calendar-capacity-fade coefficient. Rates of 0.5% and 1.0%/year are [author-defined sensitivity cases](parameters.json) and are not attributed to NREL.

This model intentionally omits nonlinear temperature, SoC, C-rate, cycle-depth, and knee-point effects. The sensitivity analysis should therefore be interpreted as an assessment of model uncertainty. Because a depreciation penalty can be an imperfect proxy for the future value lost through aging, the paper should also report a case with zero marginal wear cost while retaining physical fade; this concern is discussed by [Reniers and Howey (2024)](https://arxiv.org/abs/2403.10617).

## 6. PV degradation

| Definition | Base value | Source |
|---|---:|---|
| First-year PV degradation | 1.0% | [Trina Vertex S+ warranty specification](https://vertexsplus.trinasolar.com/wp-content/uploads/2024/04/Datasheet_Vertex-S_NEG9R.25_EN_2024_PA_web-1.pdf) |
| Subsequent annual PV degradation | 0.4%/year | [Trina Vertex S+ warranty specification](https://vertexsplus.trinasolar.com/wp-content/uploads/2024/04/Datasheet_Vertex-S_NEG9R.25_EN_2024_PA_web-1.pdf) |

The module datasheet is used only to define a representative degradation trajectory; the optimized PV capacity remains continuous and is not restricted to a particular module count.

## 7. Other model limits

| Definition | Base value | Source or status |
|---|---:|---|
| Nominal residential load | 5 kW | [Author-defined profile scaling](parameters.json) |
| Grid import limit | 5 kW | [Author-defined connection limit](parameters.json) |
| Grid export limit | 1 kW | [Author-defined conservative export limit](parameters.json) |
| PV sizing upper bound | 5 kW | [Author-defined residential search bound](parameters.json) |
| BESS sizing upper bound | 12.5 kWh | [NREL reference energy](https://atb.nrel.gov/electricity/2024/residential_battery_storage) |
| Operational prediction horizon | 36 h | [Author-defined MPC horizon](parameters.json) |
| Fine time step | 5 min | [Author-defined control resolution](parameters.json) |
| Coarse time step | 60 min | [Author-defined horizon compression](parameters.json) |

## 8. Required sensitivity cases

| Sensitivity definition | Values | Source or status |
|---|---|---|
| Real discount rate | 5%, 8%, 12% | [Author-defined financial sensitivity](parameters.json) informed by [NREL financial methods](https://atb.nrel.gov/electricity/2024/2023/financial_cases_%26_methods) |
| Residential VOLL | 2, 5, 10 USD/kWh | Central case from the [DOE study](https://www.energy.gov/sites/default/files/2024-10/NationalTransmissionPlanningStudy-Chapter5.pdf); low/high values are [author assumptions](parameters.json) |
| PV CAPEX | 2,200, 2,740, 3,300 USD/kW | Central case from [DOE/SETO](https://www.energy.gov/cmei/systems/solar-photovoltaic-system-cost-benchmarks); bounds are [author assumptions](parameters.json) |
| BESS CAPEX | 750, 1,043, 1,300 USD/kWh | Central case derived from [DOE/SETO](https://www.energy.gov/cmei/systems/solar-photovoltaic-system-cost-benchmarks); bounds are [author assumptions](parameters.json) |
| Calendar fade | 0%, 0.5%, 1.0%/year | [Author-defined model uncertainty](parameters.json) |
| Daily outage probability | 2%, 5%, 10% | [Author-defined stress levels](parameters.json), benchmarked against [EIA reliability data](https://www.eia.gov/todayinenergy/detail.php?id=61303) |
| Marginal BESS wear cost | 0 and 0.091324 USD/kWh | [Accounting sensitivity](parameters.json) motivated by [Reniers and Howey](https://arxiv.org/abs/2403.10617) |

These cases separate technology cost, physical aging, marginal wear valuation, and the additional investment induced by low grid reliability.
