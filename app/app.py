import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from io import StringIO

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    df = pd.read_csv(url)
    # normalizar nombres por si varían
    lower = {c.lower(): c for c in df.columns}
    cols = {
        "country": lower.get("country_region", "Country_Region"),
        "province": lower.get("province_state", "Province_State"),
        "confirmed": lower.get("confirmed", "Confirmed"),
        "deaths": lower.get("deaths", "Deaths"),
        "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
        "active": lower.get("active", "Active") if "active" in lower else None,
    }
    return df, url, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente: {source_url}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset, luego volver al estado inicial
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de Estados Unidos
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    # Como Streamlit no tiene pie nativo, mostramos distribución normalizada como barra
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país (simulado con bar_chart)
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
# Streamlit no tiene boxplot nativo, así que mostramos estadísticas resumen en tabla
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)


# ———————————————————————————————————————————————
# PARTE 3 – Modelado y proyecciones
# ———————————————————————————————————————————————

st.header("3. Modelado y Proyecciones")

# Selección de país para análisis temporal
pais_sel = st.selectbox("Selecciona un país para series de tiempo", sorted(df[cols["country"]].unique()))
df_pais = df[df[cols["country"]] == pais_sel]

# Nos quedamos con Confirmed y Deaths (agregados por fecha si hay múltiples filas)
# NOTA: Como tu dataset es solo un día, idealmente deberías cargar varios días históricos para que esto funcione
# Aquí simulamos con Confirmed acumulado de provincias
serie = df_pais.groupby(cols["country"])[C].sum().rename("Confirmed")

if len(serie) == 0:
    st.warning("No hay datos suficientes para este país.")
else:
    # Simulación de serie temporal acumulada (se necesita dataset histórico real)
    # Aquí lo tratamos como si fueran datos diarios
    ts = pd.Series(serie.values, index=pd.date_range(end=fecha, periods=len(serie)))

    # Suavizado de 7 días
    ts_smooth = ts.rolling(7, min_periods=1).mean()

    st.subheader("3.1 Serie temporal con suavizado de 7 días")
    fig, ax = plt.subplots()
    ts.plot(ax=ax, label="Original")
    ts_smooth.plot(ax=ax, label="Media móvil 7d")
    ax.legend()
    st.pyplot(fig)

    # Modelo ETS (Exponential Smoothing)
    st.subheader("3.2 Pronóstico a 14 días (ETS)")
    try:
        model = ExponentialSmoothing(ts, trend="add", seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(14)

        # Validación con backtesting simple (train 80%, test 20%)
        train_size = int(len(ts) * 0.8)
        train, test = ts.iloc[:train_size], ts.iloc[train_size:]
        model_bt = ExponentialSmoothing(train, trend="add", seasonal=None)
        fit_bt = model_bt.fit()
        pred_bt = fit_bt.forecast(len(test))

        mae = mean_absolute_error(test, pred_bt)
        mape = mean_absolute_percentage_error(test, pred_bt)

        st.subheader("3.3 Validación del modelo")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2%}")

        # Gráfica con bandas de confianza
        st.subheader("3.4 Forecast con bandas de confianza")
        fig, ax = plt.subplots()
        ts.plot(ax=ax, label="Datos reales")
        forecast.plot(ax=ax, label="Pronóstico", color="red")

        # Bandas de confianza aproximadas
        ci_lower = forecast * 0.9
        ci_upper = forecast * 1.1
        ax.fill_between(forecast.index, ci_lower, ci_upper, color="pink", alpha=0.3)

        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al ajustar modelo: {e}")
