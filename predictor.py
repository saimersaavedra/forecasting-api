from prophet import Prophet
import pandas as pd
import numpy as np

def prepare_category_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prepara df para Prophet:
     - renombra date→ds y category→y
     - asegura valores ≥0 y aplica log1p
    """
    df_cat = df[['date', category]].copy()
    df_cat = df_cat.rename(columns={'date': 'ds', category: 'y'})
    df_cat['y'] = df_cat['y'].clip(lower=0)
    df_cat['y'] = np.log1p(df_cat['y'])
    return df_cat

def prepare_product_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara df para Prophet para producto:
     - renombra date→ds y value→y
     - asegura valores ≥0 y aplica log1p
    """
    df_prod = df[['date', 'value']].copy()
    df_prod = df_prod.rename(columns={'date': 'ds', 'value': 'y'})
    df_prod['y'] = df_prod['y'].clip(lower=0)
    df_prod['y'] = np.log1p(df_prod['y'])
    return df_prod

def predict_next_weeks(df: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """
    Entrena Prophet sin estacionalidades y con tendencia suavizada,
    genera forecast y revierte log1p→expm1, clip y round.
    """
    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1,  # control de flexibilidad de la tendencia
        seasonality_mode='additive'
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=weeks, freq='W')
    forecast = model.predict(future)

    # Revertir transformación y limpiar resultado
    yhat = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = yhat.clip(lower=0).round().astype(int)

    return forecast[['ds', 'yhat']].tail(weeks)
