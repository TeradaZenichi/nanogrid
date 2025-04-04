import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import time
import warnings
warnings.filterwarnings("ignore")

# 1. CARREGAR OS DADOS
file_path = "data/pv_5min.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# 2. REMOVER DIAS COM GERAÇÃO ZERO
full_zero_days = df.resample('D').sum()
full_zero_days = full_zero_days[full_zero_days['Ppower'] == 0].index
df_clean = df[~df.index.normalize().isin(full_zero_days)]

# 3. USAR UMA FRAÇÃO DOS **PRIMEIROS** DADOS PARA OTIMIZAR O TEMPO
df_sampled = df_clean.head(5000)  # Reduzindo para 5.000 pontos

# 4. TESTE DE ESTAÇIONARIEDADE (ADF) PARA DEFINIR `d`
def test_stationarity(series):
    result = adfuller(series.dropna())
    print(f"Teste ADF: Estatística = {result[0]}, p-valor = {result[1]}")
    return result[1] > 0.05  # Retorna True se precisar diferenciar

d = 1 if test_stationarity(df_sampled['Ppower']) else 0
print(f"Definido d = {d}")

# 5. IDENTIFICAR (p, q) COM ACF E PACF
fig, axes = plt.subplots(1,2, figsize=(12,5))
plot_acf(df_sampled['Ppower'].dropna(), ax=axes[0])  # ACF mostra q
plot_pacf(df_sampled['Ppower'].dropna(), ax=axes[1])  # PACF mostra p
plt.show()

# Definição manual dos parâmetros observando os gráficos
p = 1  # Ordem AR
q = 1  # Ordem MA

# 6. DEFINIR PARÂMETROS SAZONAIS (P, D, Q, s)
s = 96  # Reduzido para 8 horas em vez de 24h
df_sampled['seasonal_diff'] = df_sampled['Ppower'].diff(periods=s)

# Testar necessidade de diferenciação sazonal
D = 1 if test_stationarity(df_sampled['seasonal_diff']) else 0
print(f"Definido D = {D}")

# ACF/PACF para sazonalidade
fig, axes = plt.subplots(1,2, figsize=(12,5))
plot_acf(df_sampled['seasonal_diff'].dropna(), lags=10*s, ax=axes[0])  # ACF mostra Q
plot_pacf(df_sampled['seasonal_diff'].dropna(), lags=10*s, ax=axes[1])  # PACF mostra P
plt.show()

P = 0  # Reduzido para 0 para evitar consumo excessivo de RAM
Q = 0  # Reduzido para 0 para simplificar modelo

# 7. TESTAR DIFERENTES COMBINAÇÕES E ESCOLHER MELHOR MODELO
print("Otimizando parâmetros do SARIMAX usando AIC...")

# LIMITAMOS OS PARÂMETROS PARA ACELERAR
p_range = q_range = P_range = Q_range = range(0, 2)  # Testando apenas 0 e 1
param_combinations = list(itertools.product(p_range, [d], q_range, P_range, [D], Q_range, [s]))

best_aic = np.inf
best_params = None

for params in param_combinations:
    try:
        start_time = time.time()
        model = SARIMAX(
            df_sampled['Ppower'], 
            order=params[:3], 
            seasonal_order=params[3:], 
            enforce_stationarity=False, 
            enforce_invertibility=False,
            simple_differencing=True  # Reduz memória usada para diferenciação
        )
        
        # USANDO UM OTIMIZADOR MAIS RÁPIDO
        results = model.fit(disp=False, method='nm', low_memory=True)

        elapsed_time = time.time() - start_time
        
        # INTERROMPER SE O MODELO LEVAR MAIS DE 30s PARA TREINAR
        if elapsed_time > 60:  
            print(f"Modelo {params} demorou demais ({elapsed_time:.2f}s), ignorando...")
            continue
        
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = params
        
        print(f"Testando {params} - AIC: {results.aic} - Tempo: {elapsed_time:.2f}s")

    except Exception as e:
        print(f"Erro com {params}: {e}")
        continue

print(f"\nMelhor modelo encontrado: (p,d,q) = {best_params[:3]}, (P,D,Q,s) = {best_params[3:]} com AIC = {best_aic}")
