# -*- coding: utf-8 -*-
"""E0 — Avaliacao offline de previsao (prototipos vs naives vs LSTM).

Leve (sem solver). Define qual estrategia do prototipo os demais
experimentos usam. Saidas em Results/forecasting/.
"""

from forecasting.evaluate_prototype import run_evaluation

DAYS = 365          # janela de avaliacao (dias do conjunto de teste)
EVERY_MIN = 60      # espacamento entre origens de previsao
WITH_LSTM = True    # inclui o ForecastMPC (carrega TensorFlow)

if __name__ == "__main__":
    run_evaluation(days=DAYS, every_min=EVERY_MIN, with_lstm=WITH_LSTM)
