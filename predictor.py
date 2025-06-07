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


def es_forecast_inestable(history: list[int], forecast: list[int], umbral: float = 2.5) -> bool:
    """
    Retorna True si el promedio del forecast es > umbral × promedio de las últimas 3 semanas de history.
    umbral elevado (default 2.5) para reducir falsas alarmas.
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
    umbral_inestabilidad: float = 2.5
) -> pd.DataFrame:
    """
    Predicción robusta para una categoría con variabilidad:

      1) Prepara la serie (log1p).
      2) Si hay < 6 semanas con venta > 0, fallback a promedio móvil + ruido.
      3) Usa Prophet con parámetros más flexibles.
      4) Revisa estabilidad y, si falla, fallback a promedio original.
      5) Devuelve DataFrame con ['ds', 'yhat'] en escala original (enteros).
    """
    df_cat = prepare_category_df(df, category)

    non_zero_count = (df_cat['y'] > 0).sum()

    def add_noise(array: np.ndarray, pct: float = 0.1) -> np.ndarray:
        factors = np.random.uniform(1 - pct, 1 + pct, size=array.shape)
        noisy = (array * factors).round().astype(int)
        return np.clip(noisy, 0, None)

    # Fallback si pocos datos
    if non_zero_count < 6:
        avg_val = df[category].mean()  # usar media en escala original
        base_val = int(round(avg_val))
        fechas = pd.date_range(
            start=df_cat['ds'].max() + pd.Timedelta(weeks=1),
            periods=weeks,
            freq='W-MON'  # alinear en lunes
        )
        yhat_noisy = add_noise(np.array([base_val] * weeks))
        return pd.DataFrame({'ds': fechas, 'yhat': yhat_noisy})

    # Entrenar Prophet
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,  # tendencia más flexible
        seasonality_mode='additive'
    )
    # Agregar estacionalidad mensual
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(df_cat)

    future = model.make_future_dataframe(periods=weeks, freq='W-MON')
    forecast = model.predict(future)

    forecast['yhat'] = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    df_fore = forecast[['ds', 'yhat']].tail(weeks).reset_index(drop=True)

    hist_vals = df[category].tolist()
    fore_vals = df_fore['yhat'].tolist()

    # Desactivar temporalmente fallback de inestabilidad (solo Prophet)
    # if es_forecast_inestable(hist_vals, fore_vals, umbral=umbral_inestabilidad):
    #     avg_last3 = int(round(np.mean(hist_vals[-3:])))
    #     fechas = pd.date_range(
    #         start=df_cat['ds'].max() + pd.Timedelta(weeks=1),
    #         periods=weeks,
    #         freq='W-MON'
    #     )
    #     return pd.DataFrame({'ds': fechas, 'yhat': add_noise(np.array([avg_last3] * weeks))})

    # Aplicar ruido suave al pronóstico
    yhat_noisy = add_noise(df_fore['yhat'].values)
    return pd.DataFrame({'ds': df_fore['ds'], 'yhat': yhat_noisy})


def predict_product_sales(
    df: pd.DataFrame,
    weeks: int = 4,
    umbral_inestabilidad: float = 2.5
) -> pd.DataFrame:
    """
    Predicción robusta para productos individuales:
      - Similar a categorías pero sin fallback de baja frecuencia.
    """
    df_prod = df.copy()
    non_zero_count = (df_prod['y'] > 0).sum()

    def add_noise(array: np.ndarray, pct: float = 0.1) -> np.ndarray:
        factors = np.random.uniform(1 - pct, 1 + pct, size=array.shape)
        noisy = (array * factors).round().astype(int)
        return np.clip(noisy, 0, None)

    if non_zero_count < 6:
        avg_val = np.expm1(df_prod['y']).mean()  # media original
        base_val = int(round(avg_val))
        fechas = pd.date_range(
            start=df_prod['ds'].max() + pd.Timedelta(weeks=1),
            periods=weeks,
            freq='W-MON'
        )
        return pd.DataFrame({'ds': fechas, 'yhat': add_noise(np.array([base_val] * weeks))})

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        seasonality_mode='additive'
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(df_prod)

    future = model.make_future_dataframe(periods=weeks, freq='W-MON')
    forecast = model.predict(future)

    forecast['yhat'] = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    df_fore = forecast[['ds', 'yhat']].tail(weeks).reset_index(drop=True)

    hist_vals = np.expm1(df_prod['y']).astype(int).tolist()
    fore_vals = df_fore['yhat'].tolist()

    # Desactivar chequeo inestabilidad si se requiere ver solo Prophet
    # if es_forecast_inestable(hist_vals, fore_vals, umbral=umbral_inestabilidad):
    #     avg_last3 = int(round(np.mean(hist_vals[-3:])))
    #     fechas = pd.date_range(
    #         start=df_prod['ds'].max() + pd.Timedelta(weeks=1),
    #         periods=weeks,
    #         freq='W-MON'
    #     )
    #     return pd.DataFrame({'ds': fechas, 'yhat': add_noise(np.array([avg_last3] * weeks))})

    yhat_noisy = add_noise(df_fore['yhat'].values)
    return pd.DataFrame({'ds': df_fore['ds'], 'yhat': yhat_noisy})
