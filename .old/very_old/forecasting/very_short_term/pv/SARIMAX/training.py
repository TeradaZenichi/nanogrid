# C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/SARIMAX/optuna_sarimax.py

import optuna
import pandas as pd
import numpy as np
import statsmodels.api as sm
from optuna.exceptions import TrialPruned
import joblib

def load_series(csv_file):
    """
    Carrega a série temporal a partir do CSV, define a frequência para 5 minutos,
    preenche lacunas com forward fill e remove NaNs.
    """
    df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
    series = df['Ppower'].asfreq('5min').fillna(method='ffill').dropna()
    return series

def objective(trial):
    # Sugestão dos parâmetros não sazonais
    p = trial.suggest_int('p', 0, 2)   # testa 0, 1 ou 2
    d = trial.suggest_int('d', 0, 1)   # testa 0 ou 1
    q = trial.suggest_int('q', 0, 2)   # testa 0, 1 ou 2

    # Sugestão dos parâmetros sazonais
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)
    
    s = 288  # Período sazonal fixo para dados de 5min (24*60/5)

    order = (p, d, q)
    seasonal_order = (P, D, Q, s)
    
    # Para acelerar, usamos um subconjunto dos dados (por exemplo, os primeiros 500 pontos)
    csv_file = r'C:/Users/Lucas/Code/nanogrid/data/pv_5min.csv'
    series = load_series(csv_file)
    series_sample = series.iloc[:288*4]
    
    try:
        model = sm.tsa.statespace.SARIMAX(
            series_sample,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)
    except Exception as e:
        # Se ocorrer algum erro (por exemplo, problema de convergência), fazemos o prune do trial
        raise TrialPruned() from e

    aic = result.aic
    # Reporta a métrica para o Optuna
    trial.report(aic, step=0)
    if trial.should_prune():
        raise TrialPruned()

    return aic

def main():
    # Cria o estudo com direção de minimização, usando o TPESampler (que é o padrão)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=10)  # número de trials pode ser ajustado

    print("Melhores hiperparâmetros encontrados:")
    print(study.best_params)
    print("Melhor AIC:", study.best_value)
    
    # Salva o estudo para análise futura ou plotagem
    joblib.dump(study, r"C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/SARIMAX/optuna_study.pkl")

if __name__ == '__main__':
    main()
