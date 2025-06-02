# predictor.py

from prophet import Prophet
import pandas as pd
import numpy as np

def prepare_category_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prepara df para Prophet:
     - Renombra date → ds y category → y
     - Asegura valores ≥ 0 y aplica log1p
    """
    df_cat = df[['date', category]].copy()
    df_cat = df_cat.rename(columns={'date': 'ds', category: 'y'})
    df_cat['y'] = df_cat['y'].clip(lower=0)
    df_cat['y'] = np.log1p(df_cat['y'])
    return df_cat

def prepare_product_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara df para Prophet para producto:
     - Renombra date → ds y value → y
     - Asegura valores ≥ 0 y aplica log1p
    """
    df_prod = df[['date', 'value']].copy()
    df_prod = df_prod.rename(columns={'date': 'ds', 'value': 'y'})
    df_prod['y'] = df_prod['y'].clip(lower=0)
    df_prod['y'] = np.log1p(df_prod['y'])
    return df_prod

def es_forecast_inestable(history: list[int], forecast: list[int], umbral: float = 1.5) -> bool:
    """
    Retorna True si el promedio del forecast es > umbral × promedio de las últimas 3 semanas de history.
    """
    if len(history) < 3 or len(forecast) == 0:
        return False
    avg_hist = np.mean(history[-3:])
    avg_fore = np.mean(forecast)
    return (avg_hist > 0) and (avg_fore > avg_hist * umbral)

def predict_category_sales(
    df: pd.DataFrame,
    category: str,
    weeks: int = 4,
    umbral_inestabilidad: float = 1.5
) -> pd.DataFrame:
    """
    Predicción robusta para una categoría con variabilidad:

      1) Prepara la serie (log1p).
      2) Si hay < 6 semanas con venta > 0, fallback a promedio móvil + ruido.
      3) Si el forecast de Prophet es muy alto (> umbral × promedio histórico), fallback a promedio móvil + ruido.
      4) En otro caso, usa Prophet con changepoint_prior_scale conservador y agrega ruido final.
      Devuelve DataFrame con ['ds', 'yhat'] en escala original (enteros).
    """
    # 1) Preparamos la serie
    df_cat = prepare_category_df(df, category)

    # 2) Contar cuántos puntos > 0 hay en la serie original
    non_zero_count = (df_cat['y'] > 0).sum()

    # Función para aplicar ruido uniforme ±10 %
    def add_noise(array: np.ndarray, pct: float = 0.1) -> np.ndarray:
        """
        Multiplica cada elemento de array por un factor uniforme en [1-pct, 1+pct].
        Luego redondea a entero y asegura ≥ 0.
        """
        factors = np.random.uniform(1 - pct, 1 + pct, size=array.shape)
        noisy = (array * factors).round().astype(int)
        noisy = np.clip(noisy, 0, None)
        return noisy

    #  Si hay muy pocos (< 6) valores distintos de cero → fallback a promedio móvil + ruido
    if non_zero_count < 6:
        avg_log = df_cat['y'].mean()
        fechas = pd.date_range(
            start=df_cat['ds'].max() + pd.Timedelta(weeks=1),
            periods=weeks,
            freq='W'
        )
        # Predicción constante en escala original
        base_val = int(round(np.expm1(avg_log)))
        base_array = np.array([base_val] * weeks)
        yhat_noisy = add_noise(base_array)
        return pd.DataFrame({'ds': fechas, 'yhat': yhat_noisy})

    # 3) Entrenar Prophet con parámetros conservadores
    model = Prophet(
        weekly_seasonality=True,    # activar estacionalidad semanal
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # tendencia más suave
        seasonality_mode='additive'
    )
    model.fit(df_cat)

    # 4) Hacer forecast
    future = model.make_future_dataframe(periods=weeks, freq='W')
    forecast = model.predict(future)

    # Revertir log1p → expm1 y quedarnos solo con las semanas futuras
    forecast['yhat'] = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    df_fore = forecast[['ds', 'yhat']].tail(weeks).reset_index(drop=True)

    # 5) Verificar inestabilidad del forecast
    #    Convertir histórico a escala original:
    hist_vals = np.expm1(df_cat['y'].values).astype(int)
    fore_vals = df_fore['yhat'].values
    if es_forecast_inestable(hist_vals.tolist(), fore_vals.tolist(), umbral=umbral_inestabilidad):
        # Fallback a promedio móvil + ruido basada en últimas 3 semanas del historial
        avg_last3 = int(round(hist_vals[-3:].mean()))
        fechas = pd.date_range(
            start=df_cat['ds'].max() + pd.Timedelta(weeks=1),
            periods=weeks,
            freq='W'
        )
        base_array = np.array([avg_last3] * weeks)
        yhat_noisy = add_noise(base_array)
        return pd.DataFrame({'ds': fechas, 'yhat': yhat_noisy})

    # 6) Si pasa el filtro de estabilidad, devolvemos el forecast de Prophet + ruido suave
    yhat_array = df_fore['yhat'].values
    yhat_noisy = add_noise(yhat_array)
    return pd.DataFrame({'ds': df_fore['ds'], 'yhat': yhat_noisy})

def predict_product_sales(
    df: pd.DataFrame,
    weeks: int = 4,
    umbral_inestabilidad: float = 1.5
) -> pd.DataFrame:
    """
    Predicción robusta para productos individuales:
      1) Prepara la serie con log1p.
      2) Si hay < 6 semanas con ventas > 0, fallback a promedio móvil + ruido.
      3) Si el forecast es inestable, fallback a promedio móvil + ruido.
      4) Si pasa, Prophet con parámetros conservadores + ruido.
      Devuelve ['ds', 'yhat'] en escala original (enteros).
    """
    # Preparamos la serie (ya debe venir con columnas ds, y y log1p aplicado)
    df_prod = df.copy()

    # 1) Verificar cantidad de semanas con ventas
    non_zero_count = (df_prod['y'] > 0).sum()

    # Función para aplicar ruido uniforme ±10 %
    def add_noise(array: np.ndarray, pct: float = 0.1) -> np.ndarray:
        factors = np.random.uniform(1 - pct, 1 + pct, size=array.shape)
        noisy = (array * factors).round().astype(int)
        noisy = np.clip(noisy, 0, None)
        return noisy

    # Fallback si hay pocos datos
    if non_zero_count < 6:
        avg_log = df_prod['y'].mean()
        base_val = int(round(np.expm1(avg_log)))
        fechas = pd.date_range(
            start=df_prod['ds'].max() + pd.Timedelta(weeks=1),
            periods=weeks,
            freq='W'
        )
        base_array = np.array([base_val] * weeks)
        yhat_noisy = add_noise(base_array)
        return pd.DataFrame({'ds': fechas, 'yhat': yhat_noisy})

    # 2) Entrenar Prophet
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )
    model.fit(df_prod)

    # 3) Forecast
    future = model.make_future_dataframe(periods=weeks, freq='W')
    forecast = model.predict(future)

    # Transformar predicciones a escala original
    forecast['yhat'] = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    df_fore = forecast[['ds', 'yhat']].tail(weeks).reset_index(drop=True)

    # 4) Verificar inestabilidad
    hist_vals = np.expm1(df_prod['y'].values).astype(int)
    fore_vals = df_fore['yhat'].values
    if es_forecast_inestable(hist_vals.tolist(), fore_vals.tolist(), umbral=umbral_inestabilidad):
        avg_last3 = int(round(hist_vals[-3:].mean()))
        fechas = pd.date_range(
            start=df_prod['ds'].max() + pd.Timedelta(weeks=1),
            periods=weeks,
            freq='W'
        )
        base_array = np.array([avg_last3] * weeks)
        yhat_noisy = add_noise(base_array)
        return pd.DataFrame({'ds': fechas, 'yhat': yhat_noisy})

    # 5) Forecast aceptado + ruido
    yhat_noisy = add_noise(df_fore['yhat'].values)
    return pd.DataFrame({'ds': df_fore['ds'], 'yhat': yhat_noisy})

