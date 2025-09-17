README — Da Suavização à Normalização (Max=1)

Gerado em: 2025-09-17T18:59:02

Arquivos de entrada (suavizados)
--------------------------------
- load_5min_train_smoothed.csv  | Período: 2006-12-17 00:00:00 a 2009-04-25 23:55:00 | linhas: 247968
- load_5min_test_smoothed.csv   | Período: 2009-04-26 00:00:00 a 2010-04-30 18:30:00 | linhas: 106495

Arquivos de saída (normalizados)
--------------------------------
- load_5min_train_smoothed_normalized.csv | Período: 2006-12-17 00:00:00 a 2009-04-25 23:55:00 | linhas: 247968
- load_5min_test_smoothed_normalized.csv  | Período: 2009-04-26 00:00:00 a 2010-04-30 18:30:00 | linhas: 106495

Colunas relevantes
------------------
- timestamp: carimbo de tempo (5 min).
- p_norm_smooth: série **suavizada** (em escala original).
- p_norm_smooth_norm: série **normalizada** (máximo global = 1).

Metodologia (resumo)
--------------------
1) **Suavização robusta (offline, centrada)** sobre a série original em 5 minutos
   para reduzir variações rápidas de curto prazo:
   - Hampel (janela=9 amostras ≈ 45 min, k=3.0) para aparar picos/artefatos locais;
   - Mediana móvel (janela=5 ≈ 25 min) para remover espículas curtas remanescentes;
   - Média móvel **triangular** (SMA duplo, janela=7 ≈ 35 min efetivos) para dar forma suave;
   - **EMA** (Exponential Moving Average) com span=12 (≈ 1 h) para reduzir lag residual.
   → Resultado gravado em `p_norm_smooth`.
   → Observação: Após a suavização, a **energia diária foi preservada** reescalando
     cada dia por um fator r_d = (Σ original)/(Σ suavizado), mantendo a forma intradiária.

2) **Normalização global (max=1)** aplicada sobre `p_norm_smooth`:
   - Definição: `p_norm_smooth_norm = p_norm_smooth / M`.
   - **M** = máximo global de `p_norm_smooth` no conjunto combinado (train + test).
   - Valor utilizado nesta normalização: **M = 0.682927467**.
   - A normalização **não altera a forma temporal**, apenas a escala (adimensional).
     Para reverter: `p_norm_smooth = p_norm_smooth_norm * M`.

Boas práticas
-------------
- Para manter consistência, normalize qualquer novo conjunto **com o mesmo M**.
- Para uso em **tempo real/MPC**, use versões **causais** dos filtros (sem olhar futuro)
  e considere normalização posterior com o mesmo M.

Licença e reprodutibilidade
---------------------------
- Este README documenta parâmetros e passos principais utilizados na geração dos arquivos.
- Os CSVs permitem auditoria e comparação direta entre série suavizada e normalizada.
