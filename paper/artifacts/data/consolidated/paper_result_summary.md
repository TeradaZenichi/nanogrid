# Paper result summary

## Degradation-aware sizing

| Sizing case | BESS without degradation (kWh) | BESS with degradation (kWh) | BESS change | Lifetime-cost change |
|---|---:|---:|---:|---:|
| critical_50 | 6.250 | 6.525 | +4.41% | +4.48% |
| full_100 | 12.500 | 13.012 | +4.10% | +4.23% |

## Outage-autonomy sensitivity

| Sizing case | Outage duration (h) | BESS with degradation (kWh) | PV with degradation (kW) | Lifetime objective (USD) |
|---|---:|---:|---:|---:|
| critical_50 | 1 | 6.292 | 2.199 | 14525.18 |
| critical_50 | 2 | 6.525 | 2.389 | 15372.36 |
| critical_50 | 4 | 7.966 | 2.635 | 17125.87 |
| critical_50 | 6 | 12.894 | 2.771 | 22567.43 |
| full_100 | 1 | 12.542 | 2.199 | 21043.93 |
| full_100 | 2 | 13.012 | 2.574 | 22504.29 |
| full_100 | 4 | 15.884 | 2.946 | 25925.69 |
| full_100 | 6 | infeasible; minimum 25.628 > cap 25.0 | -- | -- |

The 6 h full-service point with degradation is infeasible under the 25 kWh design cap. An expanded-bound minimum-capacity solve certified 25.6276 kWh, a 0.6276 kWh shortfall.

## Paired controller comparison

Differences are comparator minus prototype MPC for one 10-day window. Negative differences favor the comparator.

| Sizing case | Comparator | Mean difference (USD) | Bootstrap 95% CI | Holm-adjusted p-value |
|---|---|---:|---:|---:|
| critical_50 | ideal | -0.196 | [-0.274, -0.114] | 0.0029 |
| critical_50 | lstm | -0.201 | [-0.263, -0.146] | 0.0029 |
| critical_50 | load_shifting | +0.583 | [+0.478, +0.689] | 0.0029 |
| critical_50 | reserve_only | +0.507 | [+0.395, +0.621] | 0.0029 |
| critical_50 | peak_shaving | +3.508 | [+2.329, +5.367] | 0.0029 |
| critical_50 | self_consumption | +5.866 | [+4.264, +7.913] | 0.0029 |
| full_100 | ideal | -0.184 | [-0.300, -0.048] | 0.0210 |
| full_100 | lstm | -0.291 | [-0.426, -0.176] | 0.0029 |
| full_100 | load_shifting | +0.493 | [+0.361, +0.622] | 0.0029 |
| full_100 | reserve_only | +0.518 | [+0.380, +0.659] | 0.0029 |
| full_100 | peak_shaving | +2.868 | [+2.156, +3.843] | 0.0029 |
| full_100 | self_consumption | +6.396 | [+4.706, +8.525] | 0.0029 |

## Robustness

| Sizing case | Controller | Variant | Mean-cost change from base |
|---|---|---|---:|
| critical_50 | prototype | noise_005 | +1.77% |
| critical_50 | prototype | noise_010 | +3.42% |
| critical_50 | prototype | outage_2pct | -2.47% |
| critical_50 | prototype | outage_5pct | -1.63% |
| critical_50 | stochastic | noise_005 | +0.30% |
| critical_50 | stochastic | noise_010 | +0.62% |
| critical_50 | stochastic | outage_2pct | -1.13% |
| critical_50 | stochastic | outage_5pct | -0.68% |
| full_100 | prototype | noise_005 | +3.06% |
| full_100 | prototype | noise_010 | +6.12% |
| full_100 | prototype | outage_2pct | -2.85% |
| full_100 | prototype | outage_5pct | -2.41% |
| full_100 | stochastic | noise_005 | +0.63% |
| full_100 | stochastic | noise_010 | +1.29% |
| full_100 | stochastic | outage_2pct | -1.09% |
| full_100 | stochastic | outage_5pct | -0.66% |

## Statistical protocol

- 20,000 paired bootstrap resamples with seed 20260908.
- Two-sided Wilcoxon signed-rank tests.
- Holm correction within each sizing case and metric.
- Twelve paired seasonal windows per controller comparison.
