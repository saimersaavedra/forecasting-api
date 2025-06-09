# Forecast Generator (Ventas por Categoría y Producto)

Este módulo en Python permite generar pronósticos de ventas por **categoría** y por **producto** a partir de datos históricos semanales. Utiliza [Prophet](https://facebook.github.io/prophet/) para generar predicciones robustas, incluso ante datos ruidosos o escasos.

---

## 🧠 Funcionalidad

- Pronóstico de ventas por categoría (`generate_category_forecasts`)
- Pronóstico de ventas por producto (`generate_product_forecasts`)
- Detección de predicciones inestables
- Fallback inteligente cuando los datos son insuficientes

---

## 📁 Estructura

```
.
├── scripts/
│   └── generate_forecasts.py     # Script principal
├── predictor.py                  # Funciones de predicción con Prophet
├── data_utils.py                 # Utilidades para obtener y limpiar datos (no incluido aquí)
├── cache/
│   ├── categories_forecast.json # Salida: pronóstico por categorías
│   └── products_forecast.json   # Salida: pronóstico por productos
```

---

## 🚀 Ejecución

```bash
python scripts/generate_forecasts.py
```

Esto generará dos archivos en `cache/`:

- `categories_forecast.json`
- `products_forecast.json`

Cada uno incluye:

```json
[
  {
    "category": "nombre_categoria",
    "history": [ { "date": "YYYY-MM-DD", "value": N }, ... ],
    "forecasting": [ { "date": "YYYY-MM-DD", "value": M }, ... ],
    "weeks": 4
  }
]
```

---

## 📦 Requisitos

- Python ≥ 3.9
- `prophet`
- `pandas`
- `numpy`

Instalación:

```bash
pip install prophet pandas numpy
```

---

## 🧩 Detalles Técnicos

### `predictor.py`

Contiene funciones:

- `prepare_category_df` y `prepare_product_df`: Preprocesamiento con `log1p` para estabilizar varianza.
- `predict_category_sales`: Entrena `Prophet`, agrega estacionalidad mensual y aplica ruido leve.
- `predict_product_sales`: Similar al anterior, ajustado a productos.
- `es_forecast_inestable`: Detecta pronósticos sospechosos, activa fallback si es necesario.

### FallBack (cuando hay pocos datos):

- Si hay menos de 6 semanas con ventas > 0:
  - Usa la media como base.
  - Agrega ruido aleatorio leve (±10%).

---

## 📤 Salida

- Predicciones para las próximas 4 semanas (`WEEKS = 4`)
- Escala original de ventas (enteros, revertido de `log1p`)
- Pronósticos más suaves y confiables en entornos ruidosos

---

## ✅ Ejemplo de uso

```python
from scripts.generate_forecasts import generate_category_forecasts, generate_product_forecasts

generate_category_forecasts()
generate_product_forecasts()
```

---

## 📌 Notas

- Se asume que `data_utils.py` proporciona funciones:
  - `get_and_clean_category_data()`
  - `get_and_clean_product_data(product_id)`
  - `get_all_products()`
- El formato de fecha de salida es `YYYY-MM-DD`.
- Las predicciones comienzan a partir del próximo lunes (`W-MON`).

---
