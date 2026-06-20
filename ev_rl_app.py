import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

MAX_CARS = 10
DT_HOURS = 0.25  # 15 minutes

st.set_page_config(
    page_title="EV Charging Priority",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #ffffff 45%, #f7f2ff 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f4d 0%, #102c6b 55%, #21125e 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0px;
    }

    .gradient-text {
        background: linear-gradient(90deg, #a855f7, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-size: 20px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.08);
        border: 1px solid #e5e7eb;
        height: 130px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    .card {
        background: white;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.08);
        border: 1px solid #e5e7eb;
        margin-top: 20px;
    }

    .info-box {
        background: linear-gradient(90deg, #dbeafe, #eff6ff);
        padding: 16px;
        border-radius: 12px;
        color: #075985;
        font-weight: 500;
    }

    .log-box {
        background: linear-gradient(90deg, #ede9fe, #f5f3ff);
        padding: 16px;
        border-radius: 12px;
        color: #6d28d9;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .recommend-card {
        background: linear-gradient(90deg, #ecfdf5, #f0fdf4);
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 28px;
        margin-top: 25px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.08);
    }

    .recommend-title {
        color: #047857;
        font-size: 18px;
        font-weight: 700;
    }

    .recommend-value {
        color: #166534;
        font-size: 30px;
        font-weight: 800;
    }

    .car-card {
        background: #f8fafc;
        border-left: 5px solid #2563eb;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .simulate-btn button {
        background: linear-gradient(90deg, #a855f7, #2563eb) !important;
        color: white !important;
        border-radius: 12px !important;
        height: 48px;
        font-weight: 700;
        border: none;
    }

    .add-btn button {
        background: linear-gradient(90deg, #34d399, #06b6d4) !important;
        color: white !important;
        border-radius: 12px !important;
        height: 45px;
        font-weight: 700;
        border: none;
    }

    .reset-btn button {
        background: linear-gradient(90deg, #8b5cf6, #ec4899) !important;
        color: white !important;
        border-radius: 12px !important;
        height: 45px;
        font-weight: 700;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return PPO.load("ev_rl_priority_model")


model = load_model()


# ---------------- FUNCTIONS ----------------
def build_state(cars):
    state = []
    cars = sorted(cars, key=lambda x: x["time_left"])[:MAX_CARS]

    for car in cars:
        state.extend([
            car["remaining"],
            car["time_left"],
            car["wait"],
            0.0,
            car["rate"],
        ])

    while len(state) < MAX_CARS * 5:
        state.extend([0, 0, 0, 0, 0])

    return np.array(state, dtype=np.float32)


def get_recommendation(cars):
    if len(cars) == 0:
        return None

    state = build_state(cars)
    action, _ = model.predict(state, deterministic=True)

    cars_sorted = sorted(cars, key=lambda x: x["time_left"])[:MAX_CARS]
    action = int(action)

    if action >= len(cars_sorted):
        return cars_sorted[0]

    return cars_sorted[action]


def simulate_15_minutes():
    if len(st.session_state.cars) == 0:
        return

    selected = get_recommendation(st.session_state.cars)

    if selected is None:
        return

    energy_added = selected["rate"] * DT_HOURS

    for car in st.session_state.cars:
        car["time_left"] = max(car["time_left"] - DT_HOURS, 0)

        if car["name"] == selected["name"]:
            car["remaining"] = max(car["remaining"] - energy_added, 0)
        else:
            car["wait"] += DT_HOURS

    st.session_state.history.append(
        f"Charged {selected['name']} for 15 minutes | Energy added: {energy_added:.2f} kWh"
    )

    st.session_state.cars = [
        car for car in st.session_state.cars
        if car["remaining"] > 0 and car["time_left"] > 0
    ]


def calculate_metrics():
    cars = st.session_state.cars

    total_cars = len(cars)
    total_demand = sum(car["remaining"] for car in cars)
    avg_time = np.mean([car["time_left"] for car in cars]) if cars else 0
    total_rate = sum(car["rate"] for car in cars)

    return total_cars, total_demand, avg_time, total_rate


# ---------------- SESSION STATE ----------------
if "cars" not in st.session_state:
    st.session_state.cars = []

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## ⚡ EV PRIORITY")
st.sidebar.markdown("### RL Charging Optimizer")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚗 Add EV")

name = st.sidebar.text_input("Car Name", placeholder="e.g., Car_1")

remaining = st.sidebar.number_input(
    "Remaining Energy Needed (kWh)",
    min_value=1.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)

time_left = st.sidebar.number_input(
    "Time Left Before Departure (hr)",
    min_value=0.25,
    max_value=24.0,
    value=3.0,
    step=0.25
)

rate = st.sidebar.number_input(
    "Charging Rate (kW)",
    min_value=1.0,
    max_value=50.0,
    value=6.0,
    step=1.0
)

wait = st.sidebar.number_input(
    "Waiting Time (hr)",
    min_value=0.0,
    max_value=24.0,
    value=0.0,
    step=0.25
)

st.sidebar.markdown('<div class="add-btn">', unsafe_allow_html=True)
if st.sidebar.button("➕ Add Car", use_container_width=True):
    st.session_state.cars.append({
        "name": name or f"Car {len(st.session_state.cars) + 1}",
        "remaining": remaining,
        "time_left": time_left,
        "rate": rate,
        "wait": wait,
    })
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="reset-btn">', unsafe_allow_html=True)
if st.sidebar.button("🗑 Reset", use_container_width=True):
    st.session_state.cars = []
    st.session_state.history = []
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)


# ---------------- MAIN HEADER ----------------
st.markdown("""
<div>
    <div class="main-title">⚡ EV Charging Priority</div>
    <div class="main-title gradient-text">using Reinforcement Learning</div>
    <div class="subtitle">Smart • Efficient • Adaptive</div>
</div>
""", unsafe_allow_html=True)


# ---------------- METRIC CARDS ----------------
total_cars, total_demand, avg_time, total_rate = calculate_metrics()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🚗 Total Cars</div>
        <div class="metric-value">{total_cars}</div>
        <div class="metric-title">Active in Queue</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🔋 Total Demand</div>
        <div class="metric-value">{total_demand:.2f} kWh</div>
        <div class="metric-title">Energy Needed</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⏰ Avg. Time Left</div>
        <div class="metric-value">{avg_time:.2f} hr</div>
        <div class="metric-title">Before Departure</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⚡ Total Charging Rate</div>
        <div class="metric-value">{total_rate:.2f} kW</div>
        <div class="metric-title">Available Rate</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAIN CONTENT ----------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🚗 Current Cars")

    if len(st.session_state.cars) == 0:
        st.markdown(
            '<div class="info-box">ℹ️ Add EVs using the sidebar to get started.</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("No cars in the queue yet.")
    else:
        df = pd.DataFrame(st.session_state.cars)
        st.dataframe(df, use_container_width=True)

        for car in st.session_state.cars:
            urgency = car["remaining"] / max(car["time_left"], 0.1)

            st.markdown(f"""
            <div class="car-card">
                <b>{car['name']}</b><br>
                Remaining: <b>{car['remaining']:.2f} kWh</b> |
                Time Left: <b>{car['time_left']:.2f} hr</b> |
                Rate: <b>{car['rate']:.2f} kW</b> |
                Wait: <b>{car['wait']:.2f} hr</b> |
                Urgency: <b>{urgency:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📋 Simulation Log")

    if len(st.session_state.history) == 0:
        st.markdown(
            '<div class="log-box">ℹ️ No simulation steps yet.</div>',
            unsafe_allow_html=True
        )
    else:
        for item in reversed(st.session_state.history[-8:]):
            st.markdown(f'<div class="log-box">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="simulate-btn">', unsafe_allow_html=True)
    if st.button("▶ Simulate Next 15 Minutes", use_container_width=True):
        simulate_15_minutes()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- RECOMMENDATION ----------------
selected = get_recommendation(st.session_state.cars)

st.markdown('<div class="recommend-card">', unsafe_allow_html=True)

if selected is None:
    st.markdown("""
    <div class="recommend-title">🏆 RL Recommendation</div>
    <div class="recommend-value">No Recommendation Yet</div>
    <p>Add cars and run the simulation to get RL recommendations.</p>
    """, unsafe_allow_html=True)
else:
    urgency = selected["remaining"] / max(selected["time_left"], 0.1)
    completion_time = selected["remaining"] / max(selected["rate"], 0.1)

    st.markdown(f"""
    <div class="recommend-title">🏆 RL Recommendation</div>
    <div class="recommend-value">Charge {selected['name']} Next</div>
    <p>
        Remaining Energy: <b>{selected['remaining']:.2f} kWh</b> |
        Time Left: <b>{selected['time_left']:.2f} hr</b> |
        Charging Rate: <b>{selected['rate']:.2f} kW</b> |
        Waiting Time: <b>{selected['wait']:.2f} hr</b>
    </p>
    <p>
        Urgency Score: <b>{urgency:.2f}</b> |
        Estimated Completion Time: <b>{completion_time:.2f} hr</b>
    </p>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)