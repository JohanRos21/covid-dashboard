import streamlit as st
import pandas as pd
import numpy as np
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
# h) Estadística descriptiva y avanzada
# ———————————————————————————————————————————————
st.header("h) Estadística descriptiva y avanzada")

from scipy import stats

# --- 2.1 Calcular metricas clave por pais
df_stats = df.groupby(country_col).agg(
    Confirmed=(C, "sum"),
    Deaths=(D, "sum")
).reset_index()
df_stats["CFR"] = df_stats["Deaths"] / df_stats["Confirmed"].replace(0, np.nan)
df_stats["Deaths_per_100k"] = df_stats["Deaths"] / 100_000
df_stats["Confirmed_per_100k"] = df_stats["Confirmed"] / 100_000

st.subheader("2.1 Métricas clave por país")
st.dataframe(df_stats.head(25))

# --- 2.2 Intervalos de confianza para CFR (95%)
st.subheader("2.2 Intervalos de confianza para CFR")
alpha = 0.05
ci_rows = []
for _, row in df_stats.dropna(subset=["CFR"]).iterrows():
    deaths, confirmed = row["Deaths"], row["Confirmed"]
    if confirmed > 0 and deaths > 0:
        p = deaths / confirmed
        se = np.sqrt(p * (1 - p) / confirmed)
        z = stats.norm.ppf(1 - alpha/2)
        ci_low, ci_high = p - z*se, p + z*se
        ci_rows.append([row[country_col], p, ci_low, ci_high])
ci_df = pd.DataFrame(ci_rows, columns=["Country", "CFR", "CI_lower", "CI_upper"])
st.dataframe(ci_df.head(25))

# --- 2.3 Test de hipotesis de proporciones (comparar CFR entre 2 países)
st.subheader("2.3 Comparación de CFR entre dos países")
from statsmodels.stats.proportion import proportions_ztest

paises_disp = df_stats[country_col].tolist()
pais1 = st.selectbox("País 1", paises_disp, index=0)
pais2 = st.selectbox("País 2", paises_disp, index=1)

row1 = df_stats[df_stats[country_col] == pais1].iloc[0]
row2 = df_stats[df_stats[country_col] == pais2].iloc[0]
count = np.array([row1["Deaths"], row2["Deaths"]])
nobs = np.array([row1["Confirmed"], row2["Confirmed"]])
stat, pval = proportions_ztest(count, nobs)

st.write(f"H0: CFR({pais1}) = CFR({pais2})")
st.write(f"Estadístico Z = {stat:.3f}, p-value = {pval:.4f}")
if pval < 0.05:
    st.success("Se rechaza H0: hay diferencia significativa.")
else:
    st.info("No se rechaza H0: no hay diferencia significativa.")

# --- 2.4 Detección de outliers (Z-score e IQR en CFR)
st.subheader("2.4 Deteccion de outliers (CFR)")
cfr_vals = df_stats["CFR"].dropna()
z_scores = np.abs(stats.zscore(cfr_vals))
outliers_z = df_stats.loc[z_scores > 3, [country_col, "CFR"]]

Q1, Q3 = np.percentile(cfr_vals, [25, 75])
IQR = Q3 - Q1
outliers_iqr = df_stats.loc[(df_stats["CFR"] < Q1 - 1.5*IQR) | (df_stats["CFR"] > Q3 + 1.5*IQR), [country_col, "CFR"]]

st.write("Outliers por Z-score (>3):")
st.dataframe(outliers_z)
st.write("Outliers por IQR:")
st.dataframe(outliers_iqr)

# --- 2.5 Gráfico de control (3σ) de muertes diarias
st.subheader("2.5 Gráfico de control (3σ) – Muertes diarias (ejemplo global)")
# Como el dataset es de un día, simulamos acumulado con varios días
# -> Para demo: tomamos una serie ficticia de muertes acumuladas y calculamos diferencias
# Si tuvieras múltiples fechas, aquí se haría el cálculo real
muertes_sim = np.random.poisson(lam=1000, size=30)  # simula 30 días
series_diarias = pd.Series(muertes_sim)

media = series_diarias.mean()
sigma = series_diarias.std()
ucl = media + 3*sigma
lcl = max(0, media - 3*sigma)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(series_diarias, marker="o")
ax.axhline(media, color="green", linestyle="--", label="Media")
ax.axhline(ucl, color="red", linestyle="--", label="UCL (+3σ)")
ax.axhline(lcl, color="red", linestyle="--", label="LCL (-3σ)")
ax.set_title("Gráfico de control (muertes diarias simuladas)")
ax.legend()
st.pyplot(fig)
# ———————————————————————————————————————————————
# i) Modelado y proyecciones
# ———————————————————————————————————————————————
st.header("i) Modelado y proyecciones")

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------- Utilidades ----------
@st.cache_data(show_spinner=True)
def load_daily_report_safe(date_str: str):
    # Reusa el loader existente y devuelve (df, cols)
    df_i, _, cols_i = load_daily_report(date_str)
    return df_i, cols_i

@st.cache_data(show_spinner=True)
def load_range_daily_reports(start_date: str, end_date: str):
    """
    Carga múltiples reportes diarios JHU entre start_date y end_date (YYYY-MM-DD)
    y devuelve un DataFrame consolidado con columnas: Date, Country, Confirmed, Deaths.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    frames = []
    for day in pd.date_range(start, end, freq="D"):
        dstr = day.strftime("%Y-%m-%d")
        try:
            df_i, cols_i = load_daily_report_safe(dstr)
            # Normaliza columnas clave
            C_i, D_i = cols_i["confirmed"], cols_i["deaths"]
            country_i = cols_i["country"]
            tmp = df_i[[country_i, C_i, D_i]].copy()
            tmp = tmp.rename(columns={country_i: "Country", C_i: "Confirmed", D_i: "Deaths"})
            tmp["Date"] = pd.to_datetime(dstr)
            # A veces existen NaNs/negativos -> los limpiamos
            tmp[["Confirmed", "Deaths"]] = tmp[["Confirmed", "Deaths"]].clip(lower=0).fillna(0)
            frames.append(tmp)
        except Exception:
            # Algunos días no existen o cambió el esquema: los ignoramos
            continue
    if not frames:
        return pd.DataFrame(columns=["Date", "Country", "Confirmed", "Deaths"])
    df_all = pd.concat(frames, ignore_index=True)
    # Agregamos por país y fecha por seguridad
    df_all = df_all.groupby(["Date", "Country"], as_index=False).sum(numeric_only=True)
    return df_all

def prepare_country_series(df_all: pd.DataFrame, country: str, metric: str):
    """
    Devuelve una serie diaria (index DatetimeIndex) de 'metric' para un país.
    Aplica orden, relleno de huecos (freq='D') y suavizado móvil 7 días.
    """
    s = (
        df_all[df_all["Country"] == country]
        .sort_values("Date")
        .set_index("Date")[metric]
        .asfreq("D")  # fuerza frecuencia diaria (rellena huecos con NaN)
        .fillna(method="ffill")  # forward fill típico en series acumuladas
        .fillna(0)
    )
    # Si es acumulado (Confirmed/Deaths en JHU lo son), derivamos DIARIOS para modelar
    s_daily = s.diff().clip(lower=0).fillna(0)  # evita negativos espurios
    s_7d = s_daily.rolling(window=7, min_periods=1).mean()
    return s_daily, s_7d

def fit_sarimax_forecast(series_daily: pd.Series, horizon: int = 14, seasonal_period: int = 7):
    """
    Ajusta SARIMAX sobre la serie diaria (no acumulada).
    (1,1,1) + (1,1,1,7) por defecto. Devuelve forecast y conf_int.
    """
    if series_daily.sum() == 0 or len(series_daily.dropna()) < (seasonal_period * 3):
        # Demasiado poca variación/datos -> devolvemos NaNs
        idx_f = pd.date_range(series_daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(index=idx_f, dtype=float), pd.DataFrame(index=idx_f, columns=["lower y", "upper y"])

    # Modelo SARIMAX básico con estacionalidad semanal
    model = SARIMAX(
        series_daily,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="D",
    )
    res = model.fit(disp=False)
    forecast_res = res.get_forecast(steps=horizon)
    fc_mean = forecast_res.predicted_mean.clip(lower=0)  # no negativos
    fc_ci = forecast_res.conf_int(alpha=0.05)            # 95%
    # Asegura nombres de columnas consistentes
    if fc_ci.shape[1] == 2:
        fc_ci.columns = ["lower y", "upper y"]
    return fc_mean, fc_ci

def backtest_holdout(series_daily: pd.Series, horizon: int = 14, seasonal_period: int = 7):
    """
    Backtesting simple de holdout: últimos 'horizon' días como test.
    Retorna MAE y MAPE sobre el test.
    """
    y = series_daily.copy()
    if len(y) <= horizon + seasonal_period * 3:
        return np.nan, np.nan

    y_train = y.iloc[:-horizon]
    y_test = y.iloc[-horizon:]

    # Ajuste en train
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="D",
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
    pred.index = y_test.index

    mae = np.mean(np.abs(y_test - pred))
    mape = np.mean(np.where(y_test != 0, np.abs((y_test - pred) / y_test), 0)) * 100.0
    return mae, mape

# ---------- UI ----------
st.subheader("3.1 Series de tiempo con suavizado 7 días")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    start_default = (pd.to_datetime(fecha) - pd.Timedelta(days=180)).date()
    start_date = st.date_input("Fecha inicio", value=start_default)
with col2:
    end_date = st.date_input("Fecha fin", value=pd.to_datetime(fecha).date())
with col3:
    metric_choice = st.selectbox("Métrica", ["Confirmed", "Deaths"], index=1)

# Cargar rango
df_time = load_range_daily_reports(
    pd.to_datetime(start_date).strftime("%Y-%m-%d"),
    pd.to_datetime(end_date).strftime("%Y-%m-%d"),
)

if df_time.empty:
    st.warning("No se pudieron cargar datos de rango. Prueba con otro intervalo de fechas.")
else:
    # País a modelar
    paises_model = sorted(df_time["Country"].unique().tolist())
    country_sel = st.selectbox("País para modelar", paises_model, index=paises_model.index("Peru") if "Peru" in paises_model else 0)

    # Prepara series
    daily_series, smooth7 = prepare_country_series(df_time, country_sel, metric_choice)

    # Gráfico: serie diaria y suavizado 7 días
    fig1, ax1 = plt.subplots()
    ax1.plot(daily_series.index, daily_series.values, label=f"{metric_choice} diarios")
    ax1.plot(smooth7.index, smooth7.values, label=f"{metric_choice} suavizado 7d")
    ax1.set_title(f"{country_sel} – {metric_choice} diarios y suavizado (7d)")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Casos por día")
    ax1.legend()
    st.pyplot(fig1)

    # ---------- 3.2 Pronóstico 14 días (SARIMAX) ----------
    st.subheader("3.2 Pronóstico a 14 días (SARIMAX con estacionalidad semanal)")
    horizon = st.slider("Horizonte de forecast (días)", 7, 28, 14, step=7)

    fc_mean, fc_ci = fit_sarimax_forecast(daily_series, horizon=horizon, seasonal_period=7)

    if fc_mean.isna().all():
        st.info("No hay suficiente información/variación para ajustar el modelo en este período.")
    else:
        # ---------- 3.3 Backtesting (MAE/MAPE) ----------
        st.subheader("3.3 Backtesting (holdout)")
        mae, mape = backtest_holdout(daily_series, horizon=min(horizon, 14), seasonal_period=7)
        st.write(f"**MAE**: {mae:.2f}" if not np.isnan(mae) else "**MAE**: n/d")
        st.write(f"**MAPE**: {mape:.2f}%" if not np.isnan(mape) else "**MAPE**: n/d")

        # ---------- 3.4 Gráfica con bandas de confianza ----------
        st.subheader("3.4 Forecast con bandas de confianza (95%)")

        # Construimos una serie concatenada para visualizar histórico + forecast
        last_date = daily_series.index[-1]
        idx_fc = fc_mean.index

        fig2, ax2 = plt.subplots()

        # histórico (suavizado 7d para suavizar ruido visual)
        ax2.plot(smooth7.index, smooth7.values, label="Histórico (7d MA)")

        # forecast y bandas
        ax2.plot(idx_fc, fc_mean.values, label="Pronóstico (media)")
        if isinstance(fc_ci, pd.DataFrame) and "lower y" in fc_ci.columns and "upper y" in fc_ci.columns:
            ax2.fill_between(idx_fc, fc_ci["lower y"].values.clip(min=0), fc_ci["upper y"].values.clip(min=0), alpha=0.2, label="IC 95%")

        ax2.set_title(f"{country_sel} – Forecast {metric_choice} diarios (+{horizon}d)")
        ax2.set_xlabel("Fecha")
        ax2.set_ylabel("Casos por día")
        ax2.legend()
        st.pyplot(fig2)

        # Tabla de valores de forecast
        df_fc = pd.DataFrame({
            "Fecha": idx_fc,
            f"Forecast_{metric_choice}_daily": fc_mean.values,
            "IC95_lower": fc_ci["lower y"].values if "lower y" in fc_ci else np.nan,
            "IC95_upper": fc_ci["upper y"].values if "upper y" in fc_ci else np.nan,
        })
        st.dataframe(df_fc, use_container_width=True)
# ———————————————————————————————————————————————
# j) Segmentación y reducción de dimensionalidad
# ———————————————————————————————————————————————
st.header("j) Segmentación y reducción de dimensionalidad")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- 4.1 Preparar dataset con tasas, CFR y crecimiento 7d ---
st.subheader("4.1 Construcción de dataset de clustering")

# Usamos df_time (cargado en sección 3) -> si no existe, pedimos rango amplio
if "df_time" not in locals() or df_time.empty:
    df_time = load_range_daily_reports("2022-06-01", fecha_str)

if df_time.empty:
    st.warning("No hay datos suficientes para segmentación.")
else:
    # Variables base: Confirmed y Deaths acumulados al final del rango
    last_day = df_time["Date"].max()
    df_last = df_time[df_time["Date"] == last_day]

    # CFR
    df_last["CFR"] = df_last["Deaths"] / df_last["Confirmed"].replace(0, np.nan)

    # Crecimiento 7d: delta confirmados en última semana / confirmados previos
    one_week = last_day - pd.Timedelta(days=7)
    df_prev = df_time[df_time["Date"] == one_week]
    merged = pd.merge(df_last, df_prev, on="Country", suffixes=("", "_prev"), how="left").fillna(0)
    merged["Growth7d"] = (merged["Confirmed"] - merged["Confirmed_prev"]) / merged["Confirmed_prev"].replace(0, np.nan)

    features = merged[["Country", "CFR", "Growth7d", "Confirmed", "Deaths"]].copy()
    # Tasas por 100k como proxy (sin población real, usamos Confirmed/Deaths normalizados)
    features["Confirmed_rate"] = features["Confirmed"] / 100_000
    features["Deaths_rate"] = features["Deaths"] / 100_000

    X = features[["CFR", "Growth7d", "Confirmed_rate", "Deaths_rate"]].fillna(0)

    # Estandarizamos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    k = st.slider("Número de clusters (k)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    features["Cluster"] = labels

    st.write("Dataset con asignación de clusters:")
    st.dataframe(features.head(25))

    # --- 4.2 PCA para reducción a 2 componentes ---
    st.subheader("4.2 PCA y visualización")
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    features["PC1"] = pcs[:, 0]
    features["PC2"] = pcs[:, 1]

    fig, ax = plt.subplots()
    scatter = ax.scatter(features["PC1"], features["PC2"], c=features["Cluster"], cmap="tab10", alpha=0.7)
    for i, row in features.iterrows():
        if row["Confirmed"] > 1_000_000:  # anotamos solo países grandes
            ax.text(row["PC1"], row["PC2"], row["Country"], fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA de métricas COVID-19 por país")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    st.pyplot(fig)

    # --- 4.3 Interpretación básica de clusters ---
    st.subheader("4.3 Interpretación de perfiles de clusters")

    cluster_profiles = features.groupby("Cluster")[["CFR", "Growth7d", "Confirmed_rate", "Deaths_rate"]].mean()
    st.write("Promedios por cluster:")
    st.dataframe(cluster_profiles)

    for clust, row in cluster_profiles.iterrows():
        desc = []
        if row["CFR"] > cluster_profiles["CFR"].mean():
            desc.append("alta mortalidad (CFR elevado)")
        else:
            desc.append("baja mortalidad")

        if row["Growth7d"] > cluster_profiles["Growth7d"].mean():
            desc.append("crecimiento rápido 7d")
        else:
            desc.append("crecimiento lento/estable")

        if row["Confirmed_rate"] > cluster_profiles["Confirmed_rate"].mean():
            desc.append("alta incidencia")
        else:
            desc.append("baja incidencia")

        st.markdown(f"**Cluster {clust}:** países con " + ", ".join(desc) + ".")
# ———————————————————————————————————————————————
# k) Dashboard en Streamlit (Sección 5)
# ———————————————————————————————————————————————
st.header("k) Dashboard integral")

import io
import matplotlib.pyplot as plt
import pydeck as pdk
from statsmodels.stats.proportion import proportions_ztest

# ======== Sidebar (5.1) ========
st.sidebar.subheader("Filtros del Dashboard")

# Rango de fechas (usa el mismo default que en i))
start_default = (pd.to_datetime(fecha) - pd.Timedelta(days=180)).date()
date_from = st.sidebar.date_input("Fecha inicio", value=start_default, key="date_from_sidebar")
date_to = st.sidebar.date_input("Fecha fin", value=pd.to_datetime(fecha).date(), key="date_to_sidebar")

# Países / Provincias
df_range = load_range_daily_reports(
    pd.to_datetime(date_from).strftime("%Y-%m-%d"),
    pd.to_datetime(date_to).strftime("%Y-%m-%d"),
)
countries_all = sorted(df_range["Country"].unique().tolist()) if not df_range.empty else []
selected_countries = st.sidebar.multiselect(
    "Países", 
    countries_all, 
    default=("Peru" if "Peru" in countries_all else countries_all[:5]),
    key="countries_sidebar"
)

# Umbral de confirmados para filtrar en tablas/mapa
umbral_confirmados = st.sidebar.number_input(
    "Umbral mínimo de Confirmados (acumulado)", 
    min_value=0, 
    value=1000, 
    step=100, 
    key="umbral_sidebar"
)

# Carga de población (CSV con columnas: Country, Population)
st.sidebar.markdown("**Carga opcional de población (CSV)**")
pop_file = st.sidebar.file_uploader(
    "Sube un CSV con columnas Country,Population", 
    type=["csv"], 
    key="pop_file_sidebar"
)
df_pop = None
if pop_file is not None:
    try:
        df_pop = pd.read_csv(pop_file)
        if {"Country","Population"}.issubset(df_pop.columns):
            st.sidebar.success("Población cargada")
        else:
            st.sidebar.error("El CSV debe tener columnas Country y Population.")
            df_pop = None
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el CSV: {e}")
        df_pop = None


# ======== KPIs (5.2) ========
st.subheader("5.2 KPIs principales")
kpi_cols = st.columns(4)
kpi_confirmed = int(df_last["Confirmed"].sum())
kpi_deaths = int(df_last["Deaths"].sum())
kpi_cfr = (df_last["Deaths"].sum() / df_last["Confirmed"].sum()) if df_last["Confirmed"].sum() > 0 else np.nan
kpi_conf_rate = df_last["Confirmed_per_100k"].mean() if "Confirmed_per_100k" in df_last.columns else np.nan
kpi_death_rate = df_last["Deaths_per_100k"].mean() if "Deaths_per_100k" in df_last.columns else np.nan

kpi_cols[0].metric("Confirmados (acum.)", f"{kpi_confirmed:,}")
kpi_cols[1].metric("Fallecidos (acum.)", f"{kpi_deaths:,}")
kpi_cols[2].metric("CFR", f"{kpi_cfr:.2%}" if pd.notna(kpi_cfr) else "n/d")
kpi_cols[3].metric("Tasa 100k (Conf/Death)", f"C:{kpi_conf_rate:,.1f} / D:{kpi_death_rate:,.2f}" if pd.notna(kpi_conf_rate) else "n/d")

# ======== Tabs (5.3) ========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visión general",
    "Estadística avanzada",
    "Modelado temporal",
    "Clustering y PCA",
    "Calidad de datos"
])

# ---------- Tab 1: Visión general ----------
with tab1:
    st.markdown("### Barras Top-N & Mapa interactivo")

    # Top-N barras por Confirmados y Fallecidos
    topn = st.slider("Top-N países", 5, 30, 10, key="topn_general")
    top_conf = df_last.groupby("Country")["Confirmed"].sum().sort_values(ascending=False).head(topn)
    top_death = df_last.groupby("Country")["Deaths"].sum().sort_values(ascending=False).head(topn)

    colA, colB = st.columns(2)
    with colA:
        st.bar_chart(top_conf)
    with colB:
        st.bar_chart(top_death)

    # Mapa (usa el daily_report del día last_day para lat/long)
    df_day, _, cols_map = load_daily_report(last_day.strftime("%Y-%m-%d"))
    lat_col = None
    long_col = None
    # Busca columnas de lat/long comunes
    for cand in ["Lat", "Latitude"]:
        if cand in df_day.columns: lat_col = cand; break
    for cand in ["Long_", "Long", "Longitude"]:
        if cand in df_day.columns: long_col = cand; break

    if lat_col and long_col:
        df_map = df_day.rename(columns={cols_map["country"]:"Country"})
        if selected_countries:
            df_map = df_map[df_map["Country"].isin(selected_countries)]
        df_map = df_map[[lat_col, long_col, cols_map["confirmed"], cols_map["deaths"], "Country"]].copy()
        df_map = df_map.rename(columns={lat_col:"lat", long_col:"lon", cols_map["confirmed"]:"Confirmed", cols_map["deaths"]:"Deaths"})
        # Pydeck
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(latitude=float(df_map["lat"].mean()), longitude=float(df_map["lon"].mean()), zoom=1.2),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position='[lon, lat]',
                    get_radius="Confirmed / 5",  # escala simple
                    pickable=True,
                    opacity=0.5,
                )
            ],
            tooltip={"text": "Country: {Country}\nConfirmed: {Confirmed}\nDeaths: {Deaths}"}
        ))
    else:
        st.info("No se encontraron columnas de lat/long para el mapa en esta fecha.")

# ---------- Tab 2: Estadística avanzada ----------
with tab2:
    st.markdown("### Boxplots, Test de hipótesis e Intervalos de confianza")

    # Boxplot real con matplotlib (Confirmed, Deaths, CFR)
    df_box = df_last[["Confirmed","Deaths","CFR"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not df_box.empty:
        fig, ax = plt.subplots()
        ax.boxplot([df_box["Confirmed"], df_box["Deaths"], df_box["CFR"]], labels=["Confirmed","Deaths","CFR"])
        ax.set_title("Boxplot de métricas")
        st.pyplot(fig)
    else:
        st.info("Sin datos suficientes para boxplot.")

    # Test de hipótesis de proporciones (CFR)
    st.markdown("#### Comparación de CFR entre dos países")
    paises_disp = sorted(df_last["Country"].unique().tolist())
    if len(paises_disp) >= 2:
        c1 = st.selectbox("País 1", paises_disp, key="ht_p1")
        c2 = st.selectbox("País 2", paises_disp, index=1, key="ht_p2")
        r1 = df_last[df_last["Country"]==c1].agg({"Deaths":"sum","Confirmed":"sum"})
        r2 = df_last[df_last["Country"]==c2].agg({"Deaths":"sum","Confirmed":"sum"})
        counts = np.array([r1["Deaths"], r2["Deaths"]])
        nobs = np.array([r1["Confirmed"], r2["Confirmed"]])
        if nobs.min() > 0:
            stat, pval = proportions_ztest(counts, nobs)
            st.write(f"Z = {stat:.3f} | p-value = {pval:.4f}")
            st.write("**Conclusión:** " + ("Diferencia significativa" if pval < 0.05 else "No significativa"))
        else:
            st.warning("Algún país tiene Confirmed=0; no se puede aplicar la prueba.")
    else:
        st.info("Selecciona al menos 2 países.")

    # Intervalos de confianza para CFR (95%)
    st.markdown("#### Intervalos de confianza del CFR (95%)")
    alpha = 0.05
    ci_rows = []
    for _, row in df_last.replace([np.inf,-np.inf], np.nan).dropna(subset=["CFR"]).iterrows():
        deaths, confirmed = row["Deaths"], row["Confirmed"]
        if confirmed > 0 and deaths >= 0:
            p = deaths / confirmed
            se = np.sqrt(p * (1 - p) / confirmed)
            z = stats.norm.ppf(1 - alpha/2)
            ci_rows.append([row["Country"], p, max(0, p - z*se), min(1, p + z*se)])
    ci_df = pd.DataFrame(ci_rows, columns=["Country","CFR","CI_lower","CI_upper"]).sort_values("CFR", ascending=False)
    st.dataframe(ci_df)

# ---------- Tab 3: Modelado temporal ----------
with tab3:
    st.markdown("### Series, forecast y validación")

    metric_choice = st.selectbox("Métrica a modelar", ["Confirmed","Deaths"], index=1, key="model_metric_tab")
    pais_model = st.selectbox("País", sorted(df_range["Country"].unique().tolist()), key="model_country_tab")

    # Serie diaria + suavizado 7d
    daily_series, smooth7 = prepare_country_series(df_range, pais_model, metric_choice)
    fig1, ax1 = plt.subplots()
    ax1.plot(daily_series.index, daily_series.values, label="Diario")
    ax1.plot(smooth7.index, smooth7.values, label="Suavizado 7d")
    ax1.set_title(f"{pais_model} – {metric_choice} diarios (7d MA)")
    ax1.legend()
    st.pyplot(fig1)

    # Forecast + validación
    horizon = st.slider("Horizonte", 7, 28, 14, step=7, key="model_h")
    fc_mean, fc_ci = fit_sarimax_forecast(daily_series, horizon=horizon, seasonal_period=7)

    # backtesting
    mae, mape = backtest_holdout(daily_series, horizon=min(horizon,14), seasonal_period=7)
    colm = st.columns(2)
    colm[0].metric("MAE", f"{mae:.2f}" if not np.isnan(mae) else "n/d")
    colm[1].metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "n/d")

    if not fc_mean.isna().all():
        fig2, ax2 = plt.subplots()
        ax2.plot(smooth7.index, smooth7.values, label="Histórico (7d MA)")
        ax2.plot(fc_mean.index, fc_mean.values, label="Forecast")
        if isinstance(fc_ci, pd.DataFrame) and set(["lower y","upper y"]).issubset(fc_ci.columns):
            ax2.fill_between(fc_mean.index, fc_ci["lower y"].clip(lower=0), fc_ci["upper y"].clip(lower=0), alpha=0.2, label="IC 95%")
        ax2.set_title(f"Forecast {metric_choice} diarios – {pais_model} (+{horizon}d)")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("No se pudo ajustar un modelo con la ventana seleccionada.")

# ---------- Tab 4: Clustering y PCA ----------
with tab4:
    st.markdown("### Scatter de clusters e interpretación")

    # Reusa cálculos de la sección j) si los tienes; si no, los regeneramos rápido:
    # --- construir features CFR, Growth7d, tasas ---
    prev_day = last_day - pd.Timedelta(days=7)
    df_prev = df_range[df_range["Date"] == prev_day]
    merged = pd.merge(
        df_range[df_range["Date"] == last_day],
        df_prev, on="Country", suffixes=("", "_prev"), how="left"
    ).fillna(0)

    merged["CFR"] = merged["Deaths"] / merged["Confirmed"].replace(0, np.nan)
    merged["Growth7d"] = (merged["Confirmed"] - merged["Confirmed_prev"]) / merged["Confirmed_prev"].replace(0, np.nan)

    if df_pop is not None:
        merged = merged.merge(df_pop, on="Country", how="left")
        merged["Confirmed_rate"] = (merged["Confirmed"] / merged["Population"]) * 100_000
        merged["Deaths_rate"] = (merged["Deaths"] / merged["Population"]) * 100_000
    else:
        merged["Confirmed_rate"] = merged["Confirmed"] / 100_000
        merged["Deaths_rate"] = merged["Deaths"] / 100_000

    # Preparar X
    X = merged[["CFR","Growth7d","Confirmed_rate","Deaths_rate"]].replace([np.inf,-np.inf], np.nan).fillna(0)

    # Escalar + KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = st.slider("K (clusters)", 2, 6, 3, key="kmeans_k_tab")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    merged["Cluster"] = labels

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    merged["PC1"] = pcs[:,0]
    merged["PC2"] = pcs[:,1]

    # Scatter
    figc, axc = plt.subplots()
    sc = axc.scatter(merged["PC1"], merged["PC2"], c=merged["Cluster"], cmap="tab10", alpha=0.8)
    axc.set_xlabel("PC1")
    axc.set_ylabel("PC2")
    axc.set_title("PCA (PC1 vs PC2) por clúster")
    axc.legend(*sc.legend_elements(), title="Cluster")
    st.pyplot(figc)

    # Interpretación por cluster
    prof = merged.groupby("Cluster")[["CFR","Growth7d","Confirmed_rate","Deaths_rate"]].mean()
    st.write("Perfiles promedio por clúster:")
    st.dataframe(prof)
    for cidx, row in prof.iterrows():
        desc = []
        desc.append("alta mortalidad" if row["CFR"] > prof["CFR"].mean() else "baja mortalidad")
        desc.append("crecimiento 7d alto" if row["Growth7d"] > prof["Growth7d"].mean() else "crecimiento 7d bajo/estable")
        desc.append("alta incidencia" if row["Confirmed_rate"] > prof["Confirmed_rate"].mean() else "baja incidencia")
        st.markdown(f"**Cluster {cidx}:** países con " + ", ".join(desc) + ".")

# ---------- Tab 5: Calidad de datos ----------
with tab5:
    st.markdown("### Nulos, inconsistencias y gráfico de control")

    # Nulos
    st.markdown("**Valores nulos (último día)**")
    st.dataframe(df_last.isna().sum())

    # Inconsistencias: acumulados decrecientes -> diffs negativos en la serie diaria
    pais_qc = st.selectbox("País para control de calidad", sorted(df_range["Country"].unique().tolist()), key="qc_country")
    daily_conf, _ = prepare_country_series(df_range, pais_qc, "Confirmed")
    daily_deaths, _ = prepare_country_series(df_range, pais_qc, "Deaths")

    neg_conf = (daily_conf < 0).sum()
    neg_death = (daily_deaths < 0).sum()
    st.write(f"Días con decrementos sospechosos (Confirmed): **{int(neg_conf)}**")
    st.write(f"Días con decrementos sospechosos (Deaths): **{int(neg_death)}**")

    # Gráfico de control (3σ) para muertes diarias (real con la serie)
    series = daily_deaths
    if len(series) >= 14:
        media = series.mean()
        sigma = series.std()
        ucl, lcl = media + 3*sigma, max(0, media - 3*sigma)
        figqc, axqc = plt.subplots()
        axqc.plot(series.index, series.values, marker="o", linestyle="-", label="Muertes diarias")
        axqc.axhline(media, linestyle="--", label="Media")
        axqc.axhline(ucl, linestyle="--", label="UCL (+3σ)")
        axqc.axhline(lcl, linestyle="--", label="LCL (-3σ)")
        axqc.set_title(f"Gráfico de control (muertes diarias) – {pais_qc}")
        axqc.legend()
        st.pyplot(figqc)
    else:
        st.info("Serie insuficiente para gráfico de control.")

# ======== 5.4 Exportación de datos y gráficos ========
st.subheader("5.4 Exportación")
# Exportar snapshot de df_last
csv_bytes = df_last.to_csv(index=False).encode("utf-8")
st.download_button("Descargar datos (CSV)", data=csv_bytes, file_name=f"dashboard_snapshot_{last_day.date()}.csv", mime="text/csv")

# Exportar gráficos (ejemplos: último fig de cada pestaña si se generó)
def download_fig(fig, base_name="grafico"):
    buff_png = io.BytesIO()
    buff_svg = io.BytesIO()
    fig.savefig(buff_png, format="png", bbox_inches="tight")
    fig.savefig(buff_svg, format="svg", bbox_inches="tight")
    st.download_button(f"Descargar {base_name} (PNG)", data=buff_png.getvalue(), file_name=f"{base_name}.png", mime="image/png")
    st.download_button(f"Descargar {base_name} (SVG)", data=buff_svg.getvalue(), file_name=f"{base_name}.svg", mime="image/svg+xml")

# Si existen figuras de pestañas previas, ofrécelas (usa 'in' locals() para evitar errores)
if "fig1" in locals(): download_fig(fig1, "serie_diaria")
if "fig2" in locals(): download_fig(fig2, "forecast_ic")
if "figc" in locals(): download_fig(figc, "pca_clusters")
if "figqc" in locals(): download_fig(figqc, "control_3sigma")

# ======== 5.5 Narrativa automática ========
st.subheader("5.5 Narrativa automática (insights)")
try:
    # Insight básico: país con mayor CFR y con mayor crecimiento 7d
    # (reutiliza 'merged' del tab4 si existe; si no, lo vuelve a crear rápido)
    if "merged" not in locals():
        prev_day = last_day - pd.Timedelta(days=7)
        df_prev = df_range[df_range["Date"] == prev_day]
        merged = pd.merge(
            df_range[df_range["Date"] == last_day],
            df_prev, on="Country", suffixes=("", "_prev"), how="left"
        ).fillna(0)
        merged["CFR"] = merged["Deaths"] / merged["Confirmed"].replace(0, np.nan)
        merged["Growth7d"] = (merged["Confirmed"] - merged["Confirmed_prev"]) / merged["Confirmed_prev"].replace(0, np.nan)

    max_cfr_row = merged.replace([np.inf,-np.inf], np.nan).dropna(subset=["CFR"]).sort_values("CFR", ascending=False).head(1)
    max_growth_row = merged.replace([np.inf,-np.inf], np.nan).dropna(subset=["Growth7d"]).sort_values("Growth7d", ascending=False).head(1)

    texto = []
    if not max_cfr_row.empty:
        texto.append(f"El CFR más alto en el rango corresponde a **{max_cfr_row.iloc[0]['Country']}** con **{max_cfr_row.iloc[0]['CFR']:.2%}**.")
    if not max_growth_row.empty and np.isfinite(max_growth_row.iloc[0]["Growth7d"]):
        texto.append(f"El mayor crecimiento 7d en confirmados lo muestra **{max_growth_row.iloc[0]['Country']}** con **{max_growth_row.iloc[0]['Growth7d']:.1%}**.")
    texto.append(f"En conjunto, los países seleccionados acumulan **{kpi_confirmed:,}** confirmados y **{kpi_deaths:,}** fallecidos al **{last_day.date()}**, con un CFR promedio de **{(kpi_cfr if pd.notna(kpi_cfr) else 0):.2%}**.")
    st.write(" ".join(texto))
except Exception:
    st.info("No se pudo generar narrativa automática para el conjunto actual.")

