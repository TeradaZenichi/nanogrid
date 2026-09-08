# Temporal-mesh selection decision model

## Purpose

This document records the decision rule used to select the nonuniform temporal
mesh of the operational MPC. It corresponds to
`operational-forecaster-regret-annual-wear-v5` in `2-operation.py`.

The selected mesh is a compromise between economic performance under the two
implementable forecasters and computational effort. Perfect-information
forecasts are retained as an evaluation reference but are not used for mesh
selection.

## Candidate set and eligibility

The candidate grid is the Cartesian product

\[
H \in \{12,24,36\}\ \mathrm{h},\qquad
t_1 \in \{5,10,15\}\ \mathrm{min},\qquad
t_2 \in \{30,60,120\}\ \mathrm{min},
\]

for a total of 27 meshes. A candidate is eligible only when it has an audited,
successful result for every one of the 12 monthly validation windows and both
required forecasters:

- LSTM;
- prototype/analog-day forecast.

Consequently, each eligible mesh is supported by 24 operational runs. Runs
that fail a solver or physical-consistency audit are excluded. A mesh with
incomplete forecaster or monthly coverage is not eligible.

## Forecaster-specific regret

Let \(C_{m,f,j}\) be the realized operating cost for mesh \(m\), forecaster
\(f\), and monthly window \(j\). The annual-validation mean is

\[
\bar C_{m,f}=\frac{1}{12}\sum_{j=1}^{12} C_{m,f,j}.
\]

The reference cost is independently determined for each forecaster:

\[
C_f^\star=\min_m \bar C_{m,f}.
\]

The nonnegative relative regret is

\[
r_{m,f}=\max\left(0,
\frac{\bar C_{m,f}-C_f^\star}{|C_f^\star|}\right).
\]

Using a separate reference for each forecaster prevents differences in their
absolute cost levels from biasing the mesh comparison.

## Decision objectives

For every eligible mesh, three quantities are minimized:

\[
\bar r_m=\frac{r_{m,\mathrm{LSTM}}+r_{m,\mathrm{prototype}}}{2},
\]

\[
r_m^{\max}=\max(r_{m,\mathrm{LSTM}},r_{m,\mathrm{prototype}}),
\]

and the mean solution time per MPC step, \(T_m\), averaged over the two
forecasters and 12 windows.

The first objective rewards good average performance. The second prevents a
mesh from being selected when it performs well with one forecaster but poorly
with the other. The third represents online computational effort.

## Pareto filter and normalized score

A mesh is Pareto dominated when another eligible mesh is no worse in all
three objectives and strictly better in at least one. Selection is first
restricted to the nondominated Pareto set.

Each objective is min--max normalized over the eligible candidates:

\[
z(x_m)=\frac{x_m-\min_k x_k}{\max_k x_k-\min_k x_k}.
\]

If all candidates have the same value, the normalized value is defined as
zero. The final score is

\[
S_m=0.4z(\bar r_m)+0.4z(r_m^{\max})+0.2z(T_m).
\]

Thus, 80% of the score is assigned to economic robustness across the two
implementable forecasters and 20% to computational effort. The Pareto mesh
with the lowest score is selected. Remaining ties are ordered by, in sequence,
lower maximum regret, lower mean regret, and lower solution time.

## Recorded selections

Both resilience cases selected \(H=24\) h, \(t_1=5\) min, and \(t_2=120\)
min.

| Case | LSTM regret | Prototype regret | Mean regret | Maximum regret | Mean solve time | Score |
|---|---:|---:|---:|---:|---:|---:|
| `critical_50` | 0.236% | 0.816% | 0.526% | 0.816% | 0.539 s | 0.12045 |
| `full_100` | 0.451% | 1.218% | 0.834% | 1.218% | 0.338 s | 0.06150 |

Scores are comparable only within each sizing case because normalization is
performed separately for each case. The selected mesh is not claimed to be a
universal optimum; it is the preferred compromise under this declared
candidate set, validation protocol, and weighting policy.

## Source artifacts

- Decision implementation: `2-operation.py::_candidate_table` and
  `2-operation.py::select_meshes`.
- Critical-case candidate table:
  `outputs/operation-sweep/with-degradation/critical_50/02-seasonal-mesh/champion_mesh.csv`.
- Full-service candidate table:
  `outputs/operation-sweep/with-degradation/full_100/02-seasonal-mesh/champion_mesh.csv`.
- Machine-readable selected records: the corresponding `champion_mesh.json`
  files.

