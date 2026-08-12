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

Execução dentro de `tmux` para sobreviver à desconexão:

```bash
python experiments/01_sizing.py                  # dimensionamento (com/sem degradação)
python experiments/01_1_sizing.py                # sensibilidade do dimensionamento
python experiments/02_forecast_eval.py           # E0: avaliação de previsão (leve)
python experiments/12_corrected_pipeline.py --stage causal-pilot --workers 4
python experiments/12_corrected_pipeline.py --stage all --workers 4
```

O experimento 01 é uma dependência obrigatória: previsão e operação carregam
`Results/sizing/alpha_gt_0` e recusam parâmetros de catálogo ou artefatos
anteriores ao fechamento cíclico do SoC.

- Paralelismo: argumento `--workers` do pipeline 12.
- Cada caso salva `parameters_used.json`, `outage_calendar.json`,
  `operation_final.csv` e `metrics.json`; cada experimento salva um
  `summary.csv` no diretório do estágio em `Results`.
- Mesmos outages entre estratégias de um experimento por construção
  (`EDS.seed` em `data/parameters.json` — não alterar entre runs comparados).
- **Instância burstable (t2/t3)**: CPU sustentada esgota créditos e afoga
  (~22–40 %/vCPU). Use família c5/c6i, ou reduza `WORKERS`.
- Trazer resultados: `tar czf results.tar.gz Results/ logs/` + scp, ou
  `aws s3 sync Results/ s3://<bucket>/nanogrid/Results/`.
