import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap, Fullscreen
from math import radians, sin, cos, asin, sqrt
from streamlit_folium import folium_static

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Transportasi TransJakarta",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚌",
)

# ======================================================
# TEMA – simpan di session_state (default: Gelap)
# ======================================================
if "tema" not in st.session_state:
    st.session_state["tema"] = "Gelap"


# ======================================================
# CSS BUILDER
# ======================================================
def build_css(dark: bool) -> str:
    if dark:
        return """
<style>
/* ===== HEADER, TOOLBAR, DAN PANAH SIDEBAR ===== */
header[data-testid="stHeader"] {
    background-color: #020617;
    box-shadow: none;
}
header [data-testid="stToolbar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    position: fixed;
    top: 0.55rem;
    left: 0.6rem;
    z-index: 2000;
}
[data-testid="collapsedControl"] > button {
    border-radius: 999px !important;
    padding: 0.15rem 0.4rem !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    background-color: #020617;
}
[data-testid="collapsedControl"] span {
    display: none;
}

/* ===== LAYOUT UMUM ===== */
.block-container {
    padding: 1.5rem 2.5rem 1.5rem 2.5rem;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: #020617;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid rgba(148, 163, 184, 0.2);
}

/* Accent */
:root { --accent: #f97316; }

/* TOP NAV */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: radial-gradient(circle at top left, rgba(15,23,42,0.95), rgba(15,23,42,0.92));
    border-radius: 0;
    padding: 0.8rem 1.3rem;
    margin-top: 0.4rem;
    margin-bottom: 1.3rem;
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 18px 40px rgba(15,23,42,0.75);
}
.top-left { display:flex; align-items:center; gap:0.75rem; }
.logo-badge {
    width: 34px; height: 34px; border-radius: 999px;
    border: 2px solid rgba(148,163,184,0.5);
    display:flex; align-items:center; justify-content:center;
    background: radial-gradient(circle at 30% 20%, #f97316, #991b1b);
    color:white; font-weight:700; font-size: 0.9rem;
}
.app-title { display:flex; flex-direction:column; }
.app-title span:first-child {
    font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: #9ca3af;
}
.app-title span:last-child { font-size: 1.2rem; font-weight: 600; }

.user-pill { display:flex; align-items:center; gap: 0.75rem; }
.user-meta { display:flex; flex-direction:column; }
.user-meta span:first-child { font-size: 0.9rem; font-weight: 600; }
.user-meta span:last-child { font-size: 0.75rem; color: #9ca3af; }
.user-avatar {
    width: 34px; height: 34px; border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #0f172a, #020617);
    border: 1px solid rgba(148,163,184,0.4);
    display:flex; align-items:center; justify-content:center;
    font-size: 0.9rem;
}

/* CARD / CONTAINER UTAMA */
[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"] ),
[data-testid="stVerticalBlock"] > div:has(> div.stPlotlyChart),
[data-testid="stVerticalBlock"] > div:has(> div.stFolium),
[data-testid="stVerticalBlock"] > div:has(> div > div.stPlotlyChart) {
    background: radial-gradient(circle at top left, rgba(15,23,42,0.98), rgba(15,23,42,0.96));
    border-radius: 0;
    border: 1px solid rgba(148,163,184,0.30);
    box-shadow: 0 18px 40px rgba(15,23,42,0.75);
    padding: 18px;
}

/* Plotly */
.js-plotly-plot, .plotly .main-svg {
    background-color: transparent !important;
}

/* Metric cards bawaan (sidebar) */
[data-testid="stMetric"] {
    background: radial-gradient(circle at top left, rgba(15,23,42,0.9), rgba(15,23,42,0.95));
    border-radius: 0;
    padding: 0.9rem 0.9rem 0.8rem 0.9rem;
    border: 1px solid rgba(148,163,184,0.30);
    box-shadow: 0 12px 30px rgba(15,23,42,0.9);
}
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #9ca3af; }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; color: #f9fafb; }

/* Tabs */
[data-testid="stTabs"] > div > div {
    background-color: transparent;
    border-bottom: 1px solid rgba(55,65,81,0.7);
}
[data-testid="stTabs"] button {
    background-color: transparent;
    border-radius: 0;
    color: #9ca3af;
    padding: 0.6rem 1.1rem;
    border: none;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid var(--accent);
    color: #f9fafb;
}

/* Headings */
h1, h2, h3, h4, h5, h6 { color: #f9fafb; font-weight: 600; }

/* Sidebar section title */
.sidebar-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}

/* ===== KPI GRID (4 kartu seragam) ===== */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.4rem;
}
.kpi-card {
    background: radial-gradient(circle at top left, rgba(15,23,42,0.9), rgba(15,23,42,0.98));
    border-radius: 0;
    border: 1px solid rgba(148,163,184,0.30);
    box-shadow: 0 18px 40px rgba(15,23,42,0.85);
    padding: 1.1rem 1.2rem 0.95rem 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 120px;
}
.kpi-label { font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.4rem; }
.kpi-main { font-size: 1.25rem; font-weight: 600; color: #f9fafb; margin-bottom: 0.6rem; }
.kpi-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    padding: 0.3rem 0.55rem;
    border-radius: 999px;
    font-weight: 500;
}
.kpi-pill-icon { font-size: 0.9rem; }
.kpi-pill-up {
    background: rgba(34,197,94,0.16);
    color: #bbf7d0;
    border: 1px solid #22c55e;
}
.kpi-pill-down {
    background: rgba(248,113,113,0.16);
    color: #fecaca;
    border: 1px solid #f97373;
}
.kpi-pill-neutral {
    background: rgba(148,163,184,0.18);
    color: #e5e7eb;
    border: 1px solid rgba(148,163,184,0.7);
}

/* ===== Ringkasan Insight Utama ===== */
.insight-header-row {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:0.35rem;
}
.insight-title { font-size:1.05rem; font-weight:600; }
.insight-caption { font-size:0.8rem; color:#9ca3af; margin-bottom:0.25rem; }
.insight-pill-active {
    font-size:0.7rem; padding:3px 8px; border-radius:999px;
    background:rgba(34,197,94,0.12); border:1px solid #22c55e; color:#bbf7d0;
}
.insight-pill-off {
    font-size:0.7rem; padding:3px 8px; border-radius:999px;
    background:rgba(148,163,184,0.15); border:1px solid rgba(148,163,184,0.6); color:#e5e7eb;
}
.insight-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.insight-table th, .insight-table td {
    border:1px solid rgba(55,65,81,0.9);
    padding:8px 12px;
}
.insight-table thead { background:rgba(15,23,42,0.9); }
.insight-table th { text-align:left; font-weight:600; }
.insight-table td.label-cell { width:26%; font-weight:500; }
.insight-table td.value-cell { width:37%; }

/* === Chips ringkasan filter di dalam TABLE === */
.filter-chips-row {
    margin-top: 0.25rem;
    display:flex;
    flex-wrap:wrap;
    gap:0.45rem;
    align-items:center;
}
.filter-chip {
    font-size:0.8rem;
    border-radius:999px;
    padding:4px 10px;
    border:1px solid rgba(148,163,184,0.7);
    background:rgba(15,23,42,0.9);
    color:#e5e7eb;
}
.filter-chip-label {
    background:rgba(34,197,94,0.14);
    border-color:#22c55e;
    color:#bbf7d0;
    font-weight:500;
}

/* Footer kelompok */
.footer-kelompok {
    text-align:center; font-size:0.8rem; color:#9ca3af;
    margin-top:1.5rem; padding-top:0.5rem;
    border-top:1px solid rgba(55,65,81,0.8);
}
</style>
"""
    else:
        return """
<style>
/* ===== HEADER, TOOLBAR, PANAH SIDEBAR (TERANG) ===== */
header[data-testid="stHeader"] {
    background-color: #f3f4f6;
    box-shadow: none;
}
header [data-testid="stToolbar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    position: fixed;
    top: 0.55rem;
    left: 0.6rem;
    z-index: 2000;
}
[data-testid="collapsedControl"] > button {
    border-radius: 999px !important;
    padding: 0.15rem 0.4rem !important;
    box-shadow: 0 4px 12px rgba(148,163,184,0.55);
    background-color: #ffffff;
}
[data-testid="collapsedControl"] span {
    display: none;
}

/* LAYOUT & BACKGROUND */
.block-container {
    padding: 1.5rem 2.5rem 1.5rem 2.5rem;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f3f4f6;
    color: #111827;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Accent */
:root { --accent: #2563eb; }

/* TOP NAV */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #ffffff, #f9fafb);
    border-radius: 0;
    padding: 0.8rem 1.3rem;
    margin-top: 0.4rem;
    margin-bottom: 1.3rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 30px rgba(148,163,184,0.25);
}
.top-left { display:flex; align-items:center; gap:0.75rem; }
.logo-badge {
    width: 34px; height: 34px; border-radius: 999px;
    border: 2px solid #e5e7eb;
    display:flex; align-items:center; justify-content:center;
    background: radial-gradient(circle at 30% 20%, #2563eb, #1e40af);
    color:white; font-weight:700; font-size: 0.9rem;
}
.app-title { display:flex; flex-direction:column; }
.app-title span:first-child {
    font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: #6b7280;
}
.app-title span:last-child { font-size: 1.2rem; font-weight: 600; }

.user-pill { display:flex; align-items:center; gap: 0.75rem; }
.user-meta { display:flex; flex-direction:column; }
.user-meta span:first-child { font-size: 0.9rem; font-weight: 600; }
.user-meta span:last-child { font-size: 0.75rem; color: #6b7280; }
.user-avatar {
    width: 34px; height: 34px; border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #e5e7eb, #ffffff);
    border: 1px solid #d1d5db;
    display:flex; align-items:center; justify-content:center;
    font-size: 0.9rem;
}

/* CARD / CONTAINER UTAMA */
[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]),
[data-testid="stVerticalBlock"] > div:has(> div.stPlotlyChart),
[data-testid="stVerticalBlock"] > div:has(> div.stFolium),
[data-testid="stVerticalBlock"] > div:has(> div > div.stPlotlyChart) {
    background: #ffffff;
    border-radius: 0;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 30px rgba(148,163,184,0.25);
    padding: 18px;
}
.js-plotly-plot, .plotly .main-svg {
    background-color: transparent !important;
}

/* Sidebar metric */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 0;
    padding: 0.9rem 0.9rem 0.8rem 0.9rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 12px 24px rgba(148,163,184,0.35);
}
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #6b7280; }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; color: #111827; }

/* Tabs */
[data-testid="stTabs"] > div > div {
    background-color: transparent;
    border-bottom: 1px solid #e5e7eb;
}
[data-testid="stTabs"] button {
    background-color: transparent;
    border-radius: 0;
    color: #6b7280;
    padding: 0.6rem 1.1rem;
    border: none;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid var(--accent);
    color: #111827;
}

/* Headings */
h1, h2, h3, h4, h5, h6 { color: #111827; font-weight: 600; }

/* Sidebar section title */
.sidebar-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #6b7280;
    margin-bottom: 0.3rem;
}

/* KPI GRID */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.4rem;
}
.kpi-card {
    background: #ffffff;
    border-radius: 0;
    border: 1px solid #e5e7eb;
    box-shadow: 0 16px 34px rgba(148,163,184,0.35);
    padding: 1.1rem 1.2rem 0.95rem 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 120px;
}
.kpi-label { font-size: 0.85rem; color: #6b7280; margin-bottom: 0.4rem; }
.kpi-main { font-size: 1.25rem; font-weight: 600; color: #111827; margin-bottom: 0.6rem; }
.kpi-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.82rem; padding: 0.3rem 0.55rem;
    border-radius: 999px; font-weight: 500;
}
.kpi-pill-icon { font-size: 0.9rem; }
.kpi-pill-up {
    background: rgba(22,163,74,0.1);
    color: #166534;
    border: 1px solid #16a34a;
}
.kpi-pill-down {
    background: rgba(220,38,38,0.08);
    color: #b91c1c;
    border: 1px solid #ef4444;
}
.kpi-pill-neutral {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

/* Ringkasan Insight Utama */
.insight-header-row {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:0.35rem;
}
.insight-title { font-size:1.05rem; font-weight:600; }
.insight-caption { font-size:0.8rem; color:#6b7280; margin-bottom:0.25rem; }
.insight-pill-active {
    font-size:0.7rem; padding:3px 8px; border-radius:999px;
    background:rgba(34,197,94,0.08); border:1px solid #22c55e; color:#166534;
}
.insight-pill-off {
    font-size:0.7rem; padding:3px 8px; border-radius:999px;
    background:#e5e7eb; border:1px solid #d1d5db; color:#374151;
}
.insight-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.insight-table th, .insight-table td {
    border:1px solid #e5e7eb;
    padding:8px 12px;
}
.insight-table thead { background:#f9fafb; }
.insight-table th { text-align:left; font-weight:600; }
.insight-table td.label-cell { width:26%; font-weight:500; }
.insight-table td.value-cell { width:37%; }

/* === Chips filter di dalam TABLE === */
.filter-chips-row {
    margin-top: 0.25rem;
    display:flex;
    flex-wrap:wrap;
    gap:0.45rem;
    align-items:center;
}
.filter-chip {
    font-size:0.8rem;
    border-radius:999px;
    padding:4px 10px;
    border:1px solid #d1d5db;
    background:#f9fafb;
    color:#111827;
}
.filter-chip-label {
    background:#dcfce7;
    border-color:#22c55e;
    color:#166534;
    font-weight:500;
}

/* Footer kelompok */
.footer-kelompok {
    text-align:center; font-size:0.8rem; color:#6b7280;
    margin-top:1.5rem; padding-top:0.5rem;
    border-top:1px solid #e5e7eb;
}
</style>
"""


# ======================================================
# DATA UTILS
# ======================================================
@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File data '{file_path}' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return pd.DataFrame()

    df["jumlah_penumpang"] = pd.to_numeric(df["jumlah_penumpang"], errors="coerce").fillna(0).astype(int)

    if "date" not in df.columns:
        if {"year", "month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int),
                    month=pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int),
                    day=1,
                ),
                errors="coerce",
            )

    cols_geo = ["latitude_awal", "longitude_awal", "latitude_tujuan", "longitude_tujuan"]
    for c in cols_geo:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "distance_km" not in df.columns and set(cols_geo).issubset(df.columns):
        df["distance_km"] = df.apply(
            lambda r: haversine_km(
                r.get("latitude_awal"),
                r.get("longitude_awal"),
                r.get("latitude_tujuan"),
                r.get("longitude_tujuan"),
            ),
            axis=1,
        )

    if "date" in df.columns:
        df = df.dropna(subset=["date"])
        df["date"] = pd.to_datetime(df["date"])
    else:
        st.error("Kolom 'date' / ('year','month') tidak ditemukan di dataset.")
        return pd.DataFrame()

    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def fmt_id(n):
    try:
        if isinstance(n, float):
            n = int(n)
        return f"{n:,}".replace(",", ".")
    except Exception:
        return str(n)


# ======================================================
# VISUAL FUNCTIONS
# ======================================================
def plot_distribusi_penumpang(df: pd.DataFrame, dark: bool, key_suffix: str = ""):
    if df.empty:
        st.warning("Data kosong setelah filter. Atur ulang filter di sidebar.")
        return

    desc = df["jumlah_penumpang"].describe()
    median_val = desc["50%"]
    q3_val = desc["75%"]

    template = "plotly_dark" if dark else "plotly_white"
    bg_color = "rgba(15,23,42,1)" if dark else "white"

    # === BOX PLOT ===
    fig = px.box(
        df,
        x="jumlah_penumpang",
        template=template,
        title="Distribusi Jumlah Penumpang per Trayek (Boxplot)",
        color_discrete_sequence=["#f97316"],
        points="suspectedoutliers"
    )

    fig.update_layout(
        xaxis_title="Jumlah Penumpang",
        yaxis_title="",
        margin=dict(t=40, b=25, l=30, r=20),
        plot_bgcolor=bg_color,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_traces(
        hovertemplate="Penumpang: %{x:,}<extra></extra>",
        marker=dict(opacity=0.7)
    )

    # === GARIS MEDIAN & Q3 (MANUAL) ===
    # Median
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Median: {fmt_id(int(median_val))}",
        annotation_position="top left",
        line_color="#22c55e"
    )

    # Q3
    fig.add_vline(
        x=q3_val,
        line_dash="dot",
        line_width=2,
        annotation_text=f"Q3: {fmt_id(int(q3_val))}",
        annotation_position="top right",
        line_color="#f97316"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Unduh CSV
    csv = df[["jumlah_penumpang"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Unduh data grafik ini (CSV)",
        csv,
        file_name=f"distribusi_penumpang{key_suffix}.csv",
        mime="text/csv",
        key=f"dl_distribusi{key_suffix}",
    )

    # === INSIGHT ===
    try:
        st.markdown(
            f"**Insight:** Median jumlah penumpang berada di sekitar **{fmt_id(int(median_val))}**. "
            f"Sedangkan nilai Q3 menunjukkan bahwa 25% trayek teratas memiliki lebih dari **{fmt_id(int(q3_val))}** penumpang. "
            f"Hal ini menandakan sebagian rute memiliki permintaan jauh lebih tinggi dibanding mayoritas."
        )
    except:
        st.markdown("**Insight:** Ringkasan distribusi tidak dapat dihitung untuk periode ini.")


def plot_tren_penumpang(df: pd.DataFrame, dark: bool, key_suffix: str = ""):
    if df.empty:
        st.warning("Data kosong setelah filter. Atur ulang filter di sidebar.")
        return

    template = "plotly_dark" if dark else "plotly_white"
    bg_color = "rgba(15,23,42,1)" if dark else "white"

    monthly = df.groupby("date", as_index=False)["jumlah_penumpang"].sum().sort_values("date")
    monthly["rolling_3m"] = monthly["jumlah_penumpang"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["date"],
            y=monthly["jumlah_penumpang"],
            mode="lines+markers",
            name="Penumpang Bulanan",
            line=dict(width=3, color="#f97316"),
            hovertemplate="Bulan: %{x|%b %Y}<br>Penumpang: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["date"],
            y=monthly["rolling_3m"],
            mode="lines",
            name="Rata-rata 3 Bulan",
            line=dict(width=2, dash="solid", color="#22c55e"),
            hovertemplate="Rata-rata 3 bln: %{y:,}<extra></extra>",
        )
    )

    try:
        fig.add_vrect(
            x0=pd.to_datetime("2021-07-01"),
            x1=pd.to_datetime("2021-08-31"),
            fillcolor="rgba(148,163,184,0.25)",
            line_width=0,
            annotation_text="PPKM",
            annotation_position="top left",
        )
    except Exception:
        pass

    fig.update_layout(
        title="Tren Jumlah Penumpang Bulanan",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Penumpang",
        template=template,
        hovermode="x unified",
        margin=dict(t=40, b=25, l=30, r=20),
        plot_bgcolor=bg_color,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    csv = monthly[["date", "jumlah_penumpang", "rolling_3m"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Unduh data grafik ini (CSV)",
        csv,
        file_name=f"tren_penumpang_bulanan{key_suffix}.csv",
        mime="text/csv",
        key=f"dl_tren{key_suffix}",
    )

    # === INSIGHT SIMPLE ===
    try:
        if len(monthly) >= 2:
            growth = (monthly["jumlah_penumpang"].iloc[-1] - monthly["jumlah_penumpang"].iloc[0]) / max(1, monthly["jumlah_penumpang"].iloc[0]) * 100
            trend_text = f"Naik {growth:.1f}%" if growth > 0 else f"Turun {abs(growth):.1f}%"
        else:
            trend_text = "Data tidak cukup untuk menghitung perubahan."
    except Exception:
        trend_text = "Pola tren tidak dapat dihitung."

    st.markdown(
        f"**Insight:** Jumlah penumpang pada periode terpilih menunjukkan tren **{trend_text}** dari awal hingga akhir periode."
    )


def plot_top_routes_dan_halte(df: pd.DataFrame, dark: bool, key_suffix: str = ""):
    if df.empty:
        st.warning("Data kosong setelah filter. Atur ulang filter di sidebar.")
        return

    template = "plotly_dark" if dark else "plotly_white"
    bg_color = "rgba(15,23,42,1)" if dark else "white"

    total_all = df["jumlah_penumpang"].sum()

    top_routes = (
        df.groupby("trayek", as_index=False)["jumlah_penumpang"]
        .sum()
        .sort_values("jumlah_penumpang", ascending=False)
        .head(10)
    )
    top_routes["share"] = top_routes["jumlah_penumpang"] / total_all * 100
    top_routes["label"] = top_routes["jumlah_penumpang"].apply(fmt_id) + top_routes["share"].map(
        lambda x: f" ({x:.1f}%)"
    )

    vc_awal = df["halte_awal"].value_counts().head(10).reset_index()
    vc_awal.columns = ["Halte", "Frekuensi"]
    vc_awal["Tipe"] = "Awal"

    vc_tuju = df["halte_tujuan"].value_counts().head(10).reset_index()
    vc_tuju.columns = ["Halte", "Frekuensi"]
    vc_tuju["Tipe"] = "Tujuan"

    combined_vc = (
        pd.concat([vc_awal, vc_tuju])
        .sort_values("Frekuensi", ascending=False)
        .drop_duplicates(subset="Halte")
        .head(10)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rute Terpadat (Top 10)")
        fig = px.bar(
            top_routes,
            x="jumlah_penumpang",
            y="trayek",
            orientation="h",
            text="label",
            template=template,
            color="jumlah_penumpang",
            color_continuous_scale=px.colors.sequential.OrRd,
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending", "title": ""},
            xaxis_title="Total Penumpang",
            height=420,
            coloraxis_colorbar=dict(title="Penumpang"),
            margin=dict(t=20, b=25, l=30, r=20),
            plot_bgcolor=bg_color,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Rute: %{y}<br>Penumpang: %{x:,}<br>Pangsa: %{customdata:.1f}%<extra></extra>",
            customdata=np.round(top_routes["share"], 1),
        )
        st.plotly_chart(fig, use_container_width=True)

        csv_routes = top_routes[["trayek", "jumlah_penumpang", "share"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Unduh data Top 10 rute (CSV)",
            csv_routes,
            file_name=f"top10_rute{key_suffix}.csv",
            mime="text/csv",
            key=f"dl_toprute{key_suffix}",
        )

    with col2:
        st.markdown("#### Halte Awal/Tujuan Terpopuler")
        fig_halte = px.bar(
            combined_vc,
            x="Frekuensi",
            y="Halte",
            color="Tipe",
            orientation="h",
            template=template,
            color_discrete_map={"Awal": "#f97316", "Tujuan": "#22c55e"},
            text=combined_vc["Frekuensi"].apply(fmt_id),
        )
        fig_halte.update_layout(
            yaxis={"categoryorder": "total ascending", "title": ""},
            xaxis_title="Frekuensi Penggunaan",
            height=420,
            legend_title_text="Tipe Halte",
            margin=dict(t=20, b=25, l=30, r=20),
            plot_bgcolor=bg_color,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_halte.update_traces(textposition="outside")
        st.plotly_chart(fig_halte, use_container_width=True)

        csv_halte = combined_vc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Unduh data halte terpopuler (CSV)",
            csv_halte,
            file_name=f"halte_populer{key_suffix}.csv",
            mime="text/csv",
            key=f"dl_haltes{key_suffix}",
        )

    st.markdown("")

    # === INSIGHT SIMPLE ===
    try:
        top_route_name = top_routes.iloc[0]['trayek']
        top_route_share = top_routes.iloc[0]['share']
        top_halte_name = combined_vc.iloc[0]['Halte']
        st.markdown(
            f"**Insight:** Rute dengan permintaan tertinggi adalah **{top_route_name}** "
            f"yang menyumbang sekitar **{top_route_share:.1f}%** dari total penumpang. "
            f"Halte tersibuk dalam periode ini adalah **{top_halte_name}**."
        )
    except Exception:
        st.markdown("**Insight:** Tidak cukup data untuk menentukan top rute/halte.")
def plot_korelasi_jarak_penumpang(df: pd.DataFrame, dark: bool, key_suffix: str = ""):
    if df.empty or "distance_km" not in df.columns:
        st.warning("Data jarak tidak tersedia atau kosong.")
        return

    template = "plotly_dark" if dark else "plotly_white"
    bg_color = "rgba(15,23,42,1)" if dark else "white"

    cols = ["distance_km", "jumlah_penumpang"]
    clean = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    N = len(clean)
    if N < 2:
        st.warning("Tidak cukup data valid untuk menghitung korelasi.")
        return

    r, _ = pearsonr(clean["distance_km"], clean["jumlah_penumpang"])
    rho, _ = spearmanr(clean["distance_km"], clean["jumlah_penumpang"], nan_policy="omit")
    r2 = r**2

    # === SCATTER PLOT SAJA (tanpa marginal histogram) ===
    fig = px.scatter(
        clean,
        x="distance_km",
        y="jumlah_penumpang",
        trendline="ols",
        opacity=0.75,
        template=template,
        title="Korelasi Jarak Rute vs. Jumlah Penumpang",
        labels={"distance_km": "Jarak Rute (km)", "jumlah_penumpang": "Jumlah Penumpang"},
    )

    # warna marker
    fig.data[0].marker.color = "#e5e7eb" if dark else "#111827"

    # warna trendline
    for trace in fig.data:
        if isinstance(trace, go.Scatter) and "lines" in trace.mode:
            trace.line.color = "#f97316"
            trace.line.width = 3
            break

    fig.update_traces(hovertemplate="Jarak: %{x:.2f} km<br>Penumpang: %{y:,}<extra></extra>")

    # anotasi statistik
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.02,
        xanchor="right",
        yanchor="bottom",
        align="left",
        text=f"Pearson r = <b>{r:.3f}</b><br>R² = <b>{r2:.3f}</b><br>Spearman ρ = <b>{rho:.3f}</b>",
        showarrow=False,
        bgcolor="rgba(15,23,42,0.9)" if dark else "rgba(243,244,246,0.95)",
        bordercolor="rgba(148,163,184,0.6)",
        borderwidth=1,
        borderpad=5,
    )

    fig.update_layout(
        margin=dict(t=40, b=25, l=30, r=20),
        plot_bgcolor=bg_color,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    csv = clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Unduh data grafik ini (CSV)",
        csv,
        file_name=f"korelasi_jarak_penumpang{key_suffix}.csv",
        mime="text/csv",
        key=f"dl_korelasi{key_suffix}",
    )

    # === INSIGHT DINAMIS ===
    if abs(r) < 0.2:
        interpret = "hubungan sangat lemah"
    elif abs(r) < 0.4:
        interpret = "hubungan lemah"
    elif abs(r) < 0.6:
        interpret = "hubungan sedang"
    else:
        interpret = "hubungan kuat"

    st.markdown(
        f"**Insight:** Korelasi antara jarak rute dan jumlah penumpang tergolong **{interpret}** "
        f"(Pearson r = {r:.2f}). Ini menunjukkan jarak bukan faktor dominan terhadap banyaknya penumpang."
    )


def plot_peta_interaktif(df: pd.DataFrame, dark: bool):
    """Peta hotspot halte. Tile mengikuti tema: gelap / terang."""
    if df.empty:
        st.warning("Data kosong setelah filter. Atur ulang filter di sidebar.")
        return

    LAT_MIN, LAT_MAX = -7.8, -5.5
    LON_MIN, LON_MAX = 106.3, 107.3

    mask_awal = df["latitude_awal"].between(LAT_MIN, LAT_MAX) & df["longitude_awal"].between(LON_MIN, LON_MAX)
    mask_tuju = df["latitude_tujuan"].between(LAT_MIN, LAT_MAX) & df["longitude_tujuan"].between(LON_MIN, LON_MAX)

    geo_ok = df[mask_awal & mask_tuju].dropna(
        subset=["latitude_awal", "longitude_awal", "latitude_tujuan", "longitude_tujuan"]
    )

    if geo_ok.empty:
        st.warning("Data koordinat tidak tersedia atau di luar area Jabodetabek yang valid.")
        return

    # filter jarak
    if "distance_km" in geo_ok.columns:
        geo_ok = geo_ok[geo_ok["distance_km"] <= 60]

    # center map
    center_lat = np.median(pd.concat([geo_ok["latitude_awal"], geo_ok["latitude_tujuan"]]))
    center_lon = np.median(pd.concat([geo_ok["longitude_awal"], geo_ok["longitude_tujuan"]]))

    # tema
    tile_style = "CartoDB DarkMatter" if dark else "CartoDB Positron"

    # buat map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tile_style,
        control_scale=True,
    )

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    n_sample = min(1200, len(geo_ok))
    geo_smpl = geo_ok.sample(n_sample, random_state=42)

    mc_awal = MarkerCluster(name="Halte Awal (Sampel)").add_to(m)
    mc_tuju = MarkerCluster(name="Halte Tujuan (Sampel)").add_to(m)

    for _, r in geo_smpl.iterrows():
        folium.CircleMarker(
            [r["latitude_awal"], r["longitude_awal"]],
            radius=3,
            color="#f97316",
            fill=True,
            fill_opacity=0.85,
            tooltip=f"Awal: {r.get('halte_awal','-')} | Trayek: {r.get('trayek','-')}",
        ).add_to(mc_awal)

        folium.CircleMarker(
            [r["latitude_tujuan"], r["longitude_tujuan"]],
            radius=3,
            color="#22c55e",
            fill=True,
            fill_opacity=0.85,
            tooltip=f"Tujuan: {r.get('halte_tujuan','-')} | Trayek: {r.get('trayek','-')}",
        ).add_to(mc_tuju)

    # heatmap halte awal
    HeatMap(
        geo_ok[["latitude_awal", "longitude_awal"]].values.tolist(),
        name="Kepadatan Halte Awal",
        radius=16,
        blur=22,
        min_opacity=0.35,
    ).add_to(m)

    # heatmap halte tujuan
    HeatMap(
        geo_ok[["latitude_tujuan", "longitude_tujuan"]].values.tolist(),
        name="Kepadatan Halte Tujuan",
        radius=16,
        blur=22,
        min_opacity=0.35,
    ).add_to(m)

    # garis lintasan sampel
    for _, r in geo_smpl.head(min(400, len(geo_smpl))).iterrows():
        folium.PolyLine(
            [(r["latitude_awal"], r["longitude_awal"]), (r["latitude_tujuan"], r["longitude_tujuan"])],
            weight=0.7,
            opacity=0.25,
            color="#64748b",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    st.info("Peta menunjukkan konsentrasi Halte Awal (oranye) dan Halte Tujuan (hijau) serta contoh lintasan rute.")
    folium_static(m, width=1000, height=540)

    # === INSIGHT DINAMIS ===
    # jumlah titik
    n_awal = geo_ok[["halte_awal", "latitude_awal", "longitude_awal"]].dropna().drop_duplicates().shape[0]
    n_tuju = geo_ok[["halte_tujuan", "latitude_tujuan", "longitude_tujuan"]].dropna().drop_duplicates().shape[0]


    # cari daerah paling padat (kelompokkan per 0.01 derajat)
    geo_ok["grid_lat"] = (geo_ok["latitude_awal"] * 100).astype(int)
    geo_ok["grid_lon"] = (geo_ok["longitude_awal"] * 100).astype(int)
    hotspot = (
        geo_ok.groupby(["grid_lat", "grid_lon"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top_density = int(hotspot.iloc[0]["count"]) if len(hotspot) else 0

    # kategori kepadatan
    if top_density > 800:
        dens_text = "sangat tinggi"
    elif top_density > 400:
        dens_text = "tinggi"
    elif top_density > 150:
        dens_text = "moderat"
    else:
        dens_text = "rendah"

    # generate insight akhir
    insight_text = (
        f"**Insight Dinamis:** Terdapat {fmt_id(n_awal)} titik halte awal dan "
        f"{fmt_id(n_tuju)} titik halte tujuan pada periode ini. "
        f"Kepadatan hotspot berada pada kategori **{dens_text}**, "
        f"menandakan adanya konsentrasi aktivitas perjalanan pada beberapa simpul utama. "
        f"Area dengan cluster terbesar mencerminkan lokasi yang paling sering "
        f"menjadi titik keberangkatan maupun tujuan, sehingga berpotensi menjadi prioritas "
        f"peningkatan layanan atau integrasi moda."
    )

    st.markdown(insight_text)



# ======================================================
# MAIN APP
# ======================================================
def main():
    df = load_data("df_final.csv")
    if df.empty:
        return

    # --- Sidebar filter & tema ---
    st.sidebar.markdown('<div class="sidebar-title">Saring Data</div>', unsafe_allow_html=True)

    date_min, date_max = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input(
        "Rentang Tanggal",
        value=[date_min, date_max],
        min_value=date_min,
        max_value=date_max,
        format="DD/MM/YYYY",
    )
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = date_min, date_max

    trayek_all = sorted(df["trayek"].dropna().unique().tolist())
    selected_trayek = st.sidebar.multiselect("Filter Trayek", trayek_all, default=[])

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">Tema</div>', unsafe_allow_html=True)
    tema_pilihan = st.sidebar.radio(
        "Mode tampilan",
        ["Gelap", "Terang"],
        index=0 if st.session_state["tema"] == "Gelap" else 1,
        label_visibility="collapsed",
        key="tema_radio",
    )
    st.session_state["tema"] = tema_pilihan
    dark_mode = st.session_state["tema"] == "Gelap"

    # Inject CSS setelah tahu tema
    st.markdown(build_css(dark_mode), unsafe_allow_html=True)

    # TOP NAV
    st.markdown(
        """
<div class="top-nav">
    <div class="top-left">
        <div class="logo-badge">TJ</div>
        <div class="app-title">
            <span>Analisis Transportasi</span>
            <span>Dashboard TransJakarta 2021</span>
        </div>
    </div>
    <div class="user-pill">
        <div class="user-meta">
            <span>Laboratorium Transportasi Jakarta</span>
            <span>Data &amp; Strategi</span>
        </div>
        <div class="user-avatar">🚌</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

   # Aplikasikan filter
    df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    if selected_trayek:
        df_filtered = df_filtered[df_filtered["trayek"].isin(selected_trayek)]

    has_filter = not (
        start_date == date_min
        and end_date == date_max
        and (not selected_trayek)
    )

    # GLOBAL baseline (tanpa filter)
    global_top_route = (
        df.groupby("trayek")["jumlah_penumpang"].sum().sort_values(ascending=False).idxmax()
        if "trayek" in df.columns and not df.empty
        else "-"
    )
    global_top_halte_awal = (
        df["halte_awal"].mode()[0]
        if "halte_awal" in df.columns and not df["halte_awal"].dropna().empty
        else "-"
    )

    # Sidebar ringkasan + unduh data
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">Ringkasan</div>', unsafe_allow_html=True)

    total_penumpang = int(df_filtered["jumlah_penumpang"].sum()) if not df_filtered.empty else 0
    n_trayek = int(df_filtered["trayek"].nunique()) if not df_filtered.empty else 0
    halte_unik = int(
        (df_filtered["halte_awal"].nunique() if "halte_awal" in df_filtered.columns else 0)
        + (df_filtered["halte_tujuan"].nunique() if "halte_tujuan" in df_filtered.columns else 0)
    ) if not df_filtered.empty else 0

    st.sidebar.metric("Total Penumpang (Filter)", fmt_id(total_penumpang))
    st.sidebar.metric("Jumlah Trayek Aktif", fmt_id(n_trayek))
    st.sidebar.metric("Total Halte Terpakai", fmt_id(halte_unik))

    csv_all = df_filtered.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇ Unduh data terfilter (CSV)",
        csv_all,
        file_name="transjakarta_filter.csv",
        mime="text/csv",
        key="dl_all_filtered",
    )

    # KPI GRID
    monthly = (
        df_filtered.groupby("date", as_index=False)["jumlah_penumpang"].sum().sort_values("date")
        if not df_filtered.empty
        else pd.DataFrame(columns=["date", "jumlah_penumpang"])
    )

    if monthly.empty:
        mx_date = mn_date = None
        mx_val = mn_val = 0
        recovery_text = "—"
        median_passenger = 0
    else:
        mx_idx = monthly["jumlah_penumpang"].idxmax()
        mn_idx = monthly["jumlah_penumpang"].idxmin()
        mx_date, mn_date = monthly.loc[mx_idx, "date"], monthly.loc[mn_idx, "date"]
        mx_val, mn_val = monthly.loc[mx_idx, "jumlah_penumpang"], monthly.loc[mn_idx, "jumlah_penumpang"]
        recovery = (mx_val - mn_val) / mn_val * 100.0 if mn_val > 0 else np.nan
        recovery_text = f"{recovery:.1f}%" if pd.notna(recovery) else "—"
        median_passenger = df_filtered["jumlah_penumpang"].median()

    peak_month_text = mx_date.strftime("%B %Y") if mx_date is not None else "N/A"
    low_month_text = mn_date.strftime("%B %Y") if mn_date is not None else "N/A"
    peak_delta_text = fmt_id(mx_val)
    low_delta_text = fmt_id(mn_val)
    median_text = fmt_id(median_passenger)

    kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Bulan Puncak Penumpang</div>
    <div class="kpi-main">{peak_month_text}</div>
    <div class="kpi-pill kpi-pill-up">
      <span class="kpi-pill-icon">▲</span>
      <span>{peak_delta_text}</span>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Bulan Terendah</div>
    <div class="kpi-main">{low_month_text}</div>
    <div class="kpi-pill kpi-pill-down">
      <span class="kpi-pill-icon">▼</span>
      <span>{low_delta_text}</span>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Pemulihan Penumpang</div>
    <div class="kpi-main">{recovery_text}</div>
    <div class="kpi-pill kpi-pill-up">
      <span class="kpi-pill-icon">↗</span>
      <span>Lembah → Puncak</span>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Median Penumpang / Trayek</div>
    <div class="kpi-main">{median_text}</div>
    <div class="kpi-pill kpi-pill-neutral">
      <span class="kpi-pill-icon">◎</span>
      <span>Nilai tengah</span>
    </div>
  </div>
</div>
"""
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["KPI Utama", "Tren & Distribusi", "Rute & Spasial", "Rekomendasi"])

    # TAB 1 – Ringkasan Insight Utama
    with tab1:
        with st.container(border=True):
            # nilai terfilter
            if not df_filtered.empty:
                filtered_top_route = (
                    df_filtered.groupby("trayek")["jumlah_penumpang"]
                    .sum()
                    .sort_values(ascending=False)
                    .idxmax()
                    if "trayek" in df_filtered.columns
                    else "-"
                )
                filtered_top_halte_awal = (
                    df_filtered["halte_awal"].mode()[0]
                    if "halte_awal" in df_filtered.columns
                    and not df_filtered["halte_awal"].dropna().empty
                    else "-"
                )
            else:
                filtered_top_route = "-"
                filtered_top_halte_awal = "-"

            pill_class = "insight-pill-active" if has_filter else "insight-pill-off"
            pill_text = "Filter aktif" if has_filter else "Tanpa filter"

            # --- Kolom Rute Terpadat ---
            route_cell = f"""
<div>
  <div style="font-size:0.8rem; color:#6b7280;">2021 (tanpa filter)</div>
  <div style="font-weight:600;">{global_top_route}</div>
"""

            if has_filter:
                route_cell += """
  <div style="margin:6px 0; border-bottom:1px solid rgba(148,163,184,0.6);"></div>
  <div style="font-size:0.8rem; color:#16a34a;">Filter saat ini</div>
"""
                if selected_trayek:
                    route_cell += '<ol style="padding-left:1.1rem; margin:2px 0 0 0; font-size:0.82rem;">'
                    for t in selected_trayek:
                        route_cell += f"<li>{t}</li>"
                    route_cell += "</ol>"
                else:
                    route_cell += f'<div style="font-weight:600;">{filtered_top_route}</div>'
            route_cell += "</div>"

            # --- Kolom Halte Awal Tersibuk ---
            halte_cell = f"""
<div>
  <div style="font-size:0.8rem; color:#6b7280;">2021 (tanpa filter)</div>
  <div style="font-weight:600;">{global_top_halte_awal}</div>
"""

            if has_filter and filtered_top_halte_awal != "-":
                halte_cell += f"""
  <div style="margin:6px 0; border-bottom:1px solid rgba(148,163,184,0.6);"></div>
  <div style="font-size:0.8rem; color:#16a34a;">Filter saat ini</div>
  <div style="font-weight:600;">{filtered_top_halte_awal}</div>
"""
            halte_cell += "</div>"

            # ====== render tabel utama ======
            st.markdown(
                f"""
<div class="insight-header-row">
  <div>
    <div class="insight-title">Ringkasan Insight Utama</div>
    <div class="insight-caption">📌 Perbandingan data 2021 vs data terfilter</div>
  </div>
  <div class="{pill_class}">{pill_text}</div>
</div>
<table class="insight-table">
  <thead>
    <tr>
      <th></th>
      <th>Rute Terpadat</th>
      <th>Halte Awal Tersibuk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="label-cell">Ringkasan</td>
      <td class="value-cell">{route_cell}</td>
      <td class="value-cell">{halte_cell}</td>
    </tr>
""",
                unsafe_allow_html=True,
            )

            # ====== baris chip trayek (opsional, di bawah kotak) ======
            if selected_trayek:
                chips_row = '<tr><td colspan="3"><div class="filter-chips-row">'
                chips_row += '<span class="filter-chip filter-chip-label">Trayek difilter</span>'
                for t in selected_trayek:
                    chips_row += f'<span class="filter-chip">{t}</span>'
                chips_row += '</div></td></tr>'
                st.markdown(chips_row, unsafe_allow_html=True)

            st.markdown(
                """
  </tbody>
</table>
""",
                unsafe_allow_html=True,
            )

    # TAB 2 – Tren & Distribusi
    with tab2:
        with st.container(border=True):
            st.subheader("Tren & Distribusi Penumpang")
            st.markdown("---")

            st.markdown("#### Tren Jumlah Penumpang Bulanan")
            plot_tren_penumpang(df_filtered, dark_mode, key_suffix="_tren")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### Distribusi Jumlah Penumpang per Trayek")
            plot_distribusi_penumpang(df_filtered, dark_mode, key_suffix="_distribusi")

    # TAB 3 – Rute & Spasial
    with tab3:
        with st.container(border=True):
            st.subheader("Rute, Halte, dan Dimensi Spasial")
            st.markdown("---")
            plot_top_routes_dan_halte(df_filtered, dark_mode, key_suffix="_rute_halte")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("Hubungan Jarak dan Permintaan")
            st.markdown("---")
            plot_korelasi_jarak_penumpang(df_filtered, dark_mode, key_suffix="_korelasi")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("Peta Hotspot Halte (Interaktif)")
            st.markdown("---")
            plot_peta_interaktif(df_filtered, dark_mode)

    # TAB 4 – Rekomendasi
    with tab4:
        with st.container(border=True):
            st.subheader("Kesimpulan & Rekomendasi Strategi")
            st.markdown("---")

            col_temuan, col_rekom = st.columns(2)
            periode_text = "periode terpilih" if has_filter else "tahun 2021"

            with col_temuan:
                st.markdown("#### Temuan Utama")
                st.markdown(
                    f"""
* *Efek Pareto:* Sebagian kecil rute menyerap porsi sangat besar permintaan penumpang pada {periode_text}.
* *Ketahanan Jaringan:* Pemulihan volume penumpang yang kuat setelah fase pembatasan.
* *Node Kritis:* Beberapa halte bekerja sebagai super hub yang sangat menentukan kualitas perjalanan.
* *Permintaan Non-Linear:* Jarak trayek tidak selalu berbanding lurus dengan banyaknya penumpang.
"""
                )

            with col_rekom:
                st.markdown("#### Rekomendasi Prioritas Aksi")
                st.markdown(
                    f"""
1. *Perkuat Top Rute* – tambah armada & kurangi headway pada rute dengan kontribusi penumpang terbesar di {periode_text}.
2. *Tata Ulang Hub Utama* – tingkatkan kapasitas antrian, informasi real-time, dan integrasi moda di halte tersibuk.
3. *Right-Sizing Rute Panjang* – tinjau kembali rute jauh dengan load factor rendah (hasil korelasi jarak).
4. *Pemantauan Berkala* – gunakan dashboard ini sebagai early warning jika ada pergeseran pola permintaan per rute/halte.
"""
                )

        st.markdown("---")
        st.caption("Dashboard dibangun dengan Python, Streamlit, Plotly, dan Folium. Data: TransJakarta 2021.")

    # FOOTER KELOMPOK
    st.markdown('<div class="footer-kelompok">KELOMPOK HAHAHA</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
