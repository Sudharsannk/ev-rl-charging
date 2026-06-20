import streamlit as st
import numpy as np
from stable_baselines3 import PPO

MAX_CARS = 10
DT_HOURS = 0.25   # 15 minutes

st.set_page_config(
    page_title="EV RL Charging Priority",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Charging Priority using Reinforcement Learning")


@st.cache_resource
def load_model():
    return PPO.load("ev_rl_priority_model")


model = load_model()


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

    energy_added = selected["rate"] * DT_HOURS

    for car in st.session_state.cars:
        car["time_left"] = max(car["time_left"] - DT_HOURS, 0)

        if car["name"] == selected["name"]:
            car["remaining"] = max(car["remaining"] - energy_added, 0)
        else:
            car["wait"] += DT_HOURS

    st.session_state.history.append(
        f"Charged {selected['name']} for 15 minutes. Energy added = {energy_added:.2f} kWh"
    )

    st.session_state.cars = [
        car for car in st.session_state.cars
        if car["remaining"] > 0 and car["time_left"] > 0
    ]


if "cars" not in st.session_state:
    st.session_state.cars = []

if "history" not in st.session_state:
    st.session_state.history = []


st.sidebar.header("Add EV")

name = st.sidebar.text_input("Car Name")
remaining = st.sidebar.number_input(
    "Remaining Energy Needed (kWh)",
    min_value=1.0,
    max_value=100.0,
    value=20.0
)
time_left = st.sidebar.number_input(
    "Time Left Before Departure (hr)",
    min_value=0.25,
    max_value=24.0,
    value=3.0
)
rate = st.sidebar.number_input(
    "Charging Rate (kW)",
    min_value=1.0,
    max_value=50.0,
    value=6.0
)
wait = st.sidebar.number_input(
    "Waiting Time (hr)",
    min_value=0.0,
    max_value=24.0,
    value=0.0
)

if st.sidebar.button("Add Car"):
    st.session_state.cars.append({
        "name": name or f"Car {len(st.session_state.cars) + 1}",
        "remaining": remaining,
        "time_left": time_left,
        "rate": rate,
        "wait": wait,
    })

if st.sidebar.button("Reset"):
    st.session_state.cars = []
    st.session_state.history = []


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Cars")

    if len(st.session_state.cars) == 0:
        st.info("Add cars from the sidebar.")
    else:
        st.dataframe(st.session_state.cars, use_container_width=True)

        selected = get_recommendation(st.session_state.cars)

        urgency = selected["remaining"] / max(selected["time_left"], 0.1)

        st.success(f"RL Recommendation: Charge **{selected['name']}** next")

        st.write("Remaining energy:", round(selected["remaining"], 2), "kWh")
        st.write("Time left:", round(selected["time_left"], 2), "hr")
        st.write("Charging rate:", round(selected["rate"], 2), "kW")
        st.write("Waiting time:", round(selected["wait"], 2), "hr")
        st.write("Urgency score:", round(urgency, 3))

        if st.button("▶ Simulate Next 15 Minutes"):
            simulate_15_minutes()
            st.rerun()


with col2:
    st.subheader("Simulation Log")

    if len(st.session_state.history) == 0:
        st.info("No simulation steps yet.")
    else:
        for item in reversed(st.session_state.history):
            st.write(item)