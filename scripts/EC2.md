# Rodando os experimentos num EC2 / SageMaker Jupyter

Setup único (terminal Linux):

```bash
# 1. código
git clone <repo> nanogrid && cd nanogrid

# 2. ambiente
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-exp.txt   # pinado, UTF-8, inclui pyomo+gurobipy

# 3. licença Gurobi (WLS acadêmica — a named-user local não valida fora do campus)
export GRB_WLSACCESSID=... GRB_WLSSECRET=... GRB_LICENSEID=...
python -c "import gurobipy; gurobipy.Model()"   # smoke test da licença
```

Execução (dentro de `tmux` para sobreviver à desconexão): **um main por
experimento, sem argumentos nem env vars** — toda a configuração são
constantes maiúsculas no topo de cada arquivo:

```bash
python 1-sizing.py                  # dimensionamento (com/sem degradação)
python 2-forecast_eval.py           # E0: avaliação de previsão (leve)
python 3-forecaster_comparison.py   # E1: ideal/estocástico/lstm/prototype/hybrid
python 4-sized_system.py            # E2: idem no sistema dimensionado
python 5-mesh_sweep.py              # E3: sweep da malha (h x dt1 x dt2, paralelo)
python 6-robustness.py              # E4: ruído/outage/seeds (após E1/E3)
```

- Paralelismo do sweep: constante `WORKERS` no topo de `5-mesh_sweep.py`
  (1 thread de solver por processo).
- Cada caso salva `parameters_used.json`, `outage_calendar.json`,
  `operation_final.csv` e `metrics.json`; cada experimento salva um
  `summary.csv` no seu `Results/<experimento>/`.
- Mesmos outages entre estratégias de um experimento por construção
  (`EDS.seed` em `data/parameters.json` — não alterar entre runs comparados).
- **Instância burstable (t2/t3)**: CPU sustentada esgota créditos e afoga
  (~22–40 %/vCPU). Use família c5/c6i, ou reduza `WORKERS`.
- Trazer resultados: `tar czf results.tar.gz Results/ logs/` + scp, ou
  `aws s3 sync Results/ s3://<bucket>/nanogrid/Results/`.
