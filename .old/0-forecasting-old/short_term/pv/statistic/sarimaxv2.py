import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# 1. CARREGAR OS DADOS
file_path = "data/load_5min.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# 2. REMOVER DIAS COM GERAÇÃO ZERO
full_zero_days = df.resample('D').sum()
full_zero_days = full_zero_days[full_zero_days['Ppower'] == 0].index
df_clean = df[~df.index.normalize().isin(full_zero_days)]

# 3. USAR UMA FRAÇÃO DOS PRIMEIROS DADOS PARA OTIMIZAR TEMPO
df_sampled = df_clean.head(5000)  # Trabalhando com 10.000 pontos

# 4. AUTO-ARIMA PARA DESCOBRIR OS MELHORES PARÂMETROS
print("Executando Auto-ARIMA para encontrar os melhores parâmetros...")

best_model = auto_arima(
    df_sampled['Ppower'], 
    seasonal=True, 
    m=288,  # Sazonalidade diária (5 min * 288 = 24h)
    trace=True,  # Exibir progresso
    suppress_warnings=True, 
    stepwise=True,  # Otimiza a busca
    n_jobs=-1,
    error_action='ignore',  # Ignora modelos que não convergem
)

# 5. RECUPERAR OS MELHORES PARÂMETROS ENCONTRADOS
p, d, q = best_model.order
P, D, Q, s = best_model.seasonal_order
print(f"\nMelhores parâmetros encontrados: (p,d,q) = ({p},{d},{q}), (P,D,Q,s) = ({P},{D},{Q},{s})")
