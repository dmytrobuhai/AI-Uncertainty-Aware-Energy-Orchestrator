# main.py
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd

import requests_cache
import openmeteo_requests
from retry_requests import retry
from fp.fp import FreeProxy

from orchestrator import EnergyOrchestrator
from llm_agent import EnergyGraphAgent

@st.cache_resource
def load_llm_agent():
    return EnergyGraphAgent()
# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("data")
MODEL_DIR = Path("serialized")
UKR_SHP = DATA_DIR / "shapefiles/ne_10m_admin_0_countries_ukr.shp"

T_CONTEXT = 72          # 3 дні контексту (Encoder)
HORIZON = 6             # 6 годин прогнозу (Decoder)
GNN_DAYS = 7            
FUTURE_FEAT_PER_STEP = 5 

# Open-Meteo cache
CACHE_FILE = DATA_DIR / ".cache_openmeteo"

st.set_page_config(
    layout="wide",
    page_title="AI Uncertainty-Aware Energy Orchestrator",
    page_icon="⚡"
)

agent = load_llm_agent()

# =========================================================
# PROXY + OPENMETEO CLIENT
# =========================================================
def proxy_generator():
    while True:
        try:
            yield FreeProxy(rand=True, timeout=1).get()
        except Exception:
            time.sleep(1)

proxy_pool = proxy_generator()

def get_openmeteo_client(proxy=None):
    cache_session = requests_cache.CachedSession(str(CACHE_FILE), expire_after=3600)
    if proxy:
        cache_session.proxies = {"http": proxy, "https": proxy}
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    return openmeteo_requests.Client(session=retry_session)

@st.cache_data(ttl=1800)
def fetch_future_weather_6h(lat, lon, horizon):
    client = get_openmeteo_client(next(proxy_pool))
    res = client.weather_api(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "wind_speed_10m", "cloud_cover"],
            "forecast_hours": horizon,
            "timezone": "UTC"
        }
    )[0].Hourly()

    return {
        "temperature": res.Variables(0).ValuesAsNumpy(),
        "wind_speed":  res.Variables(1).ValuesAsNumpy(),
        "cloud_cover": res.Variables(2).ValuesAsNumpy(),
    }


# =========================================================
# LOAD SYSTEM & DATA
# =========================================================
@st.cache_resource
def load_system():
    return EnergyOrchestrator(model_dir=str(MODEL_DIR))

@st.cache_data
def load_data():
    df_load = pd.read_csv(DATA_DIR / "ready2use_region_electricity.csv", parse_dates=["timestamp_utc"])
    df_gnn = pd.read_csv(DATA_DIR / "prepared_energy_dataset.csv")
    if "date" in df_gnn.columns:
        df_gnn["date"] = pd.to_datetime(df_gnn["date"])
    latest_ts = df_load["timestamp_utc"].max()
    return df_load, df_gnn, latest_ts

orchestrator = load_system()
df_load_raw, df_gnn_raw, current_time = load_data()
regions = orchestrator.regions
region_to_idx = {r: i for i, r in enumerate(regions)}


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("Параметри системи")
    st.info(f"Останні дані: {current_time:%Y-%m-%d %H:%M} UTC")

    selected_region = st.selectbox(
        "Регіон для аналізу",
        regions,
        index=regions.index("Київ") if "Київ" in regions else 0
    )

    st.divider()
    st.subheader("Симуляція майбутнього (Decoder)")

    sim_alert = st.toggle("🚨 Повітряна тривога в прогнозі")
    sim_damage = st.toggle("💥 Ураження об'єктів у прогнозі", disabled=not sim_alert)

    st.divider()
    st.subheader("Налаштування балансування")
    
    # Поріг, після якого ми вважаємо, що регіону потрібна допомога
    threshold_deficit = st.slider("Поріг дефіциту (МВт)", 0, 100, 5)
    
    # Поріг, при якому ми готові забирати енергію з регіону-донора
    max_donor_risk = st.slider("Макс. ризик донора (%)", 0, 100, 80) / 100
    
    # Мінімальний профіцит для передачі
    min_surplus = st.slider("Мін. профіцит для передачі (МВт)", 1, 50, 10)


# =========================================================
# INPUT PREPARATION
# =========================================================
def build_x_past(df_load):
    """Створює тензор історії з реальних даних"""
    times = sorted(df_load["timestamp_utc"].unique())[-T_CONTEXT:]
    df_w = df_load[df_load["timestamp_utc"].isin(times)]
    x_past = np.zeros((1, T_CONTEXT, 27, 15), dtype=np.float32)

    for t_i, ts in enumerate(times):
        slice_t = df_w[df_w["timestamp_utc"] == ts]
        for _, r in slice_t.iterrows():
            if r["ua_region"] not in region_to_idx: continue
            rid = region_to_idx[r["ua_region"]]
            demand, gen = float(r["demand_ua_adj"]), float(r["generation_ua_adj"])
            h = ts.hour
            x_past[0, t_i, rid, :] = [
                demand, gen, demand - gen,
                float(r.get("alert_active", 0)), float(r.get("isDamaged", 0)),
                float(r.get("temperature", 0)), float(r.get("humidity", 0)),
                float(r.get("precipitation", 0)), float(r.get("snowfall", 0)),
                float(r.get("wind_speed", 0)), float(r.get("wind_gusts", 0)),
                float(r.get("cloud_cover", 0)), float(r.get("surface_pressure", 0)),
                np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            ]
    return torch.tensor(x_past), times

def build_gnn_tensor(df_gnn, alert_active=False, damage_active=False, selected_reg=None):
    days = sorted(df_gnn["date"].unique())[-GNN_DAYS:]
    gnn_tensor = np.zeros((GNN_DAYS, len(orchestrator.node_names), 12), dtype=np.float32)

    # Генератор випадкового шуму для реалістичності
    rng = np.random.default_rng()

    for d_i, day in enumerate(days):
        df_d = df_gnn[df_gnn["date"] == day]
        for _, r in df_d.iterrows():
            node = f"REGION::{r['region']}"
            if node in orchestrator.node_names:
                nid = orchestrator.node_names.index(node)
                
                # Початкові дані з CSV
                alert_val = float(r.get("alert_intensity", 0))
                is_damaged = float(r.get("isDamaged", 0))

                # СИМУЛЯЦІЯ: Змінюємо останні 3 дні (індекси 4, 5, 6)
                if d_i >= (GNN_DAYS - 3) and r['region'] == selected_reg:
                    if alert_active:
                        # Додаємо рандомну інтенсивність тривоги від 0.7 до 1.3
                        alert_val = max(alert_val, 1.0 * rng.uniform(0.7, 1.3))
                    if damage_active:
                        # Пошкодження з ймовірністю (чим ближче до сьогодні, тим вища ймовірність)
                        prob = (d_i - (GNN_DAYS - 4)) / 3.0
                        if rng.random() < prob:
                            is_damaged = 1.0

                gnn_tensor[d_i, nid, :] = [
                    alert_val,
                    is_damaged,
                    r.get("temperature_mean", 0),
                    r.get("precipitation", 0),
                    r.get("snowfall", 0),
                    r.get("wind_speed_max", 0),
                    r.get("wind_gusts_max", 0),
                    r.get("cloud_cover_mean", 0),
                    r.get("surface_pressure_mean", 0),
                    r.get("log_blackout_consumers", 0),
                    r.get("log_blackout_settlements", 0),
                    0.0 # kv_norm placeholder
                ]
    return torch.tensor(gnn_tensor, dtype=torch.float32)

def build_x_future_sim(weather, alert, damage):
    """Формує вхід для декодера на основі кнопок симуляції"""
    x_future = np.zeros((1, HORIZON * FUTURE_FEAT_PER_STEP), dtype=np.float32)
    for h in range(HORIZON):
        base = h * FUTURE_FEAT_PER_STEP
        x_future[0, base + 0] = float(alert)
        x_future[0, base + 1] = float(damage)
        x_future[0, base + 2] = weather["temperature"][h]
        x_future[0, base + 3] = weather["wind_speed"][h]
        x_future[0, base + 4] = weather["cloud_cover"][h]
    return torch.tensor(x_future, dtype=torch.float32)


# =========================================================
# EXECUTION
# =========================================================
x_past, hist_times = build_x_past(df_load_raw)

coords = orchestrator.region_coords.get(selected_region, (50.45, 30.52))
future_weather = fetch_future_weather_6h(coords[0], coords[1], HORIZON)

# Побудова майбутнього з урахуванням симуляції
x_future = build_x_future_sim(future_weather, sim_alert, sim_damage)
gnn_in = build_gnn_tensor(
    df_gnn_raw, 
    alert_active=sim_alert, 
    damage_active=sim_damage, 
    selected_reg=selected_region
)
# RUN INFERENCE
report, risks, deficits = orchestrator.run_inference(x_past, x_future, gnn_in)

# -------------------------
# BALANCING
# -------------------------
st.divider()
st.subheader("Міжрегіональне балансування")
recs = orchestrator.get_balancing_recommendations(report, risks, deficits, threshold=threshold_deficit,risk_limit=max_donor_risk,min_gen=min_surplus)
if recs:
    for r in recs: st.warning(f"🔄 **{r['from']}** → **{r['to']}** | {r['reason']}")
else:
    st.info("Баланс у нормі — перекидання не потрібне.")


# =========================================================
# UI RENDERING
# =========================================================
st.title("AI Uncertainty-Aware Energy Orchestrator")
st.markdown(f"**Стан системи на {current_time:%Y-%m-%d %H:%M} UTC**")

col_map, col_info = st.columns([2, 1])

# -------------------------
# MAP
# -------------------------
with col_map:
    st.subheader("Карта стратегічних ризиків (GNN)")
    fig, ax = plt.subplots(figsize=(10, 8))
    if UKR_SHP.exists():
        gpd.read_file(UKR_SHP).plot(ax=ax, color="#f8f9fa", edgecolor="#adb5bd", linewidth=0.6)

    ax.set_xlim(21.8, 40.5); ax.set_ylim(44.0, 52.5)

    for r in regions:
        if r not in orchestrator.region_coords: continue
        node_idx = orchestrator.region_to_node.get(r)
        if node_idx is not None:
            risk = float(risks[node_idx])
            lat, lon = orchestrator.region_coords[r]
            color = plt.cm.RdYlGn_r(risk)
            ax.scatter(lon, lat, s=220, c=[color], edgecolors="black", zorder=5)
            ax.annotate(f"{risk:.2f}", (lon, lat), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, weight="bold")

    ax.set_axis_off()
    st.pyplot(fig)

# -------------------------
# INFO PANEL
# -------------------------
with col_info:
    st.subheader(f"Регіон: {selected_region}")
    reg = report[selected_region]
    risk_val = float(reg["strategic_risk_score"])

    # Визначаємо колір: якщо ризик низький — зелений (normal), якщо високий — червоний (inverse)
    is_critical = risk_val > 0.6
    st.metric(
        "Strategic Risk Score", 
        f"{risk_val:.2%}", 
        delta="КРИТИЧНО" if is_critical else "СТАБІЛЬНО", 
        delta_color="inverse" if is_critical else "normal"
    )

    st.write("**План дій (6 годин):**")
    sel_idx = region_to_idx[selected_region]
    for h, step in enumerate(reg["schedule"]):
        with st.expander(f"H+{step['hour']} — {step['mode']}", expanded=(h == 0)):
            d = step['deficit']
            if d > 0: st.error(f"📉 Дефіцит: {d:.1f} МВт")
            else: st.success(f"📈 Профіцит: {abs(d):.1f} МВт")
            st.caption(step["recommendation"])

    st.divider()
    st.write("**Майбутня погода та Симуляція:**")
    wf = pd.DataFrame({
        "h": [f"+{i+1}h" for i in range(HORIZON)],
        "Temp": future_weather["temperature"],
        "Wind": future_weather["wind_speed"],
        "Alert": ["ТАК" if sim_alert else "НІ"] * HORIZON,
        "Damage": ["ТАК" if sim_damage else "НІ"] * HORIZON,
    })
    st.dataframe(wf, use_container_width=True)

    st.divider()
    st.subheader("🤖 AI Диспетчер (LangGraph + RAG)")

    if st.button("Отримати аналіз ситуації"):
        with st.spinner("Агент вивчає регламенти та проводить аналіз..."):
            try:
                # Виклик графа
                explanation = agent.run(
                    selected_region, 
                    reg, 
                    future_weather, 
                    recs
                )
                st.markdown(f"**Вердикт:** {explanation}")
            except Exception as e:
                st.error(f"Помилка AI агента: {str(e)}")

# -------------------------
# DEFICIT CHART
# -------------------------
st.divider()
st.subheader("Прогноз дефіциту потужності")
t0 = pd.to_datetime(hist_times[-1])
time_fut = [t0 + pd.Timedelta(hours=i+1) for i in range(HORIZON)]
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(time_fut, deficits[:, sel_idx], marker="o", linewidth=2, color='#1f77b4')
ax2.axhline(0, linewidth=1, color='black', linestyle='--')
ax2.set_ylabel("Energy deficit (MW)")
ax2.grid(alpha=0.3)
st.pyplot(fig2)




# -------------------------
# REGIONAL BALANCE OVERVIEW
# -------------------------
st.divider()
st.subheader("📊 Загальносистемний моніторинг (H+1)")

balance_data = []
for i, r in enumerate(regions):
    d = deficits[0, i]
    n_idx = orchestrator.region_to_node.get(r, 0)
    r_risk = float(risks[n_idx])
    
    balance_data.append({
        "Регіон": r,
        "Баланс (МВт)": d,
        "Ризик (%)": r_risk,
        "Стан": "🔴 ДЕФІЦИТ" if d > 5 else ("🟢 ПРОФІЦИТ" if d < -5 else "⚪ ОПТИМАЛЬНО")
    })

df_balance = pd.DataFrame(balance_data).sort_values("Баланс (МВт)", ascending=False)

st.dataframe(
    df_balance,
    column_config={
        "Баланс (МВт)": st.column_config.NumberColumn(format="%.1f"),
        "Ризик": st.column_config.ProgressColumn(
            "Ризик (%)",
            help="Ймовірність аварії вузла",
            format="%.1f", # Це відобразить 0.65 як 0.6 (текстом на барі)
            min_value=0,
            max_value=1
        ),
    },
    use_container_width=True,
    hide_index=True
)

# -------------------------
# DEBUG
# -------------------------
with st.expander("Debug System Data"):
    st.json(report[selected_region])
    d0 = pd.DataFrame({"region": regions, "deficit_h1_MW": deficits[0, :]})
    st.dataframe(d0.sort_values("deficit_h1_MW", ascending=False), use_container_width=True)