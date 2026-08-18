# Rodando os experimentos num EC2 / SageMaker Jupyter

Setup único (terminal Linux):

```bash
# 1. código
git clone <repo> nanogrid && cd nanogrid

# 2. ambiente
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # inclui HiGHS e o runtime Parquet
pip install gurobipy                  # opcional; requer licenca valida

# 3. licença Gurobi (WLS acadêmica — a named-user local não valida fora do campus)
export GRB_WLSACCESSID=... GRB_WLSSECRET=... GRB_LICENSEID=...
python -c "import gurobipy; gurobipy.Model()"   # smoke test da licença
```

Execução dentro de `tmux` para sobreviver à desconexão:

```bash
python experiments/01_sizing.py                  # dimensionamento (com/sem degradação)
python experiments/01_1_sizing.py                # sensibilidade do dimensionamento
python experiments/02_forecast_eval.py           # E0: avaliação de previsão (leve)
pwsh operation-campaigns/critical_50/run_full_sweep.ps1
# Em outra copia/maquina: pwsh operation-campaigns/full_100/run_full_sweep.ps1
```

O repositório inclui em `paper/sizing/` os artefatos auditados necessários para
previsão e operação. Rode o experimento 01 apenas para refazer o sizing e,
depois, promova o resultado com `python scripts/build_paper_results.py`.

- Paralelismo: argumento `--workers` do pipeline 12.
- Cada caso salva `parameters_used.json`, `outage_calendar.json`,
  `operation_final.parquet` e `metrics.json`; cada experimento salva um
  `summary.csv` no diretório do estágio em `outputs/sweeps/<campaign_id>/`.
- Mesmos outages entre estratégias de um experimento por construção
  (`EDS.seed` em `data/parameters.json` — não alterar entre runs comparados).
- **Instância burstable (t2/t3)**: CPU sustentada esgota créditos e afoga
  (~22–40 %/vCPU). Use família c5/c6i, ou reduza `WORKERS`.
- Trazer saídas brutas: `tar czf outputs.tar.gz outputs/ logs/` + scp, ou
  `aws s3 sync outputs/ s3://<bucket>/nanogrid/outputs/`.
