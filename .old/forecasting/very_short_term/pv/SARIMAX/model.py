# C:\Users\Lucas\Code\nanogrid\forecasting\very_short_term\pv\SARIMAX\model.py

import statsmodels.api as sm

class SARIMAXModel:
    def __init__(self, order, seasonal_order):
        """
        Inicializa o modelo SARIMAX com os parâmetros não sazonais (order)
        e sazonais (seasonal_order), que devem ser uma tupla, por exemplo:
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def fit(self, series):
        """
        Ajusta o modelo SARIMAX à série temporal.
        """
        self.model = sm.tsa.statespace.SARIMAX(series,
                                               order=self.order,
                                               seasonal_order=self.seasonal_order,
                                               enforce_stationarity=False,
                                               enforce_invertibility=False)
        self.results = self.model.fit(disp=False)
        return self.results

    def forecast(self, steps):
        """
        Gera a previsão para o número de passos especificado.
        Retorna a previsão e os intervalos de confiança.
        """
        forecast_obj = self.results.get_forecast(steps=steps)
        pred_mean = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        return pred_mean, conf_int
