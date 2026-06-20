import argparse, ast, random, joblib, warnings
import numpy as np
import pandas as pd
import gymnasium as gym

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

warnings.filterwarnings("ignore")

MAX_CARS = 10
DT_HOURS = 0.25   # 15 minutes
DEFAULT_RATE = 6.0


def parse_datetime(x):
    if pd.isna(x):
        return pd.NaT
    return pd.to_datetime(x, format="%a, %d %b %Y %H:%M:%S GMT", utc=True)


def extract_user_inputs(raw):
    try:
        data = ast.literal_eval(raw)
        if not data:
            return pd.Series({
                "kWhRequested": np.nan,
                "minutesAvailable": np.nan,
                "requestedDeparture": pd.NaT,
            })

        last = data[-1]

        return pd.Series({
            "kWhRequested": float(last.get("kWhRequested", np.nan)),
            "minutesAvailable": float(last.get("minutesAvailable", np.nan)),
            "requestedDeparture": parse_datetime(last.get("requestedDeparture")),
        })

    except Exception:
        return pd.Series({
            "kWhRequested": np.nan,
            "minutesAvailable": np.nan,
            "requestedDeparture": pd.NaT,
        })


def load_dataset(path):
    df = pd.read_csv(path)

    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col] = df[col].apply(parse_datetime)

    user_df = df["userInputs"].apply(extract_user_inputs)
    df = pd.concat([df, user_df], axis=1)

    df["date"] = df["connectionTime"].dt.date

    df["charging_hr"] = (
        df["doneChargingTime"] - df["connectionTime"]
    ).dt.total_seconds() / 3600

    df["session_hr"] = (
        df["disconnectTime"] - df["connectionTime"]
    ).dt.total_seconds() / 3600

    df["energy_rate"] = (
        df["kWhDelivered"] / df["charging_hr"].clip(lower=0.1)
    ).fillna(DEFAULT_RATE).clip(1, 50)

    df["energy_needed"] = (
        df["kWhRequested"]
        .fillna(df["kWhDelivered"] * 1.2)
        .fillna(10)
    )

    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    df = df[df["session_hr"] > 0]

    return df


def build_episodes(df, group_col="siteID"):
    episodes = []

    for _, group in df.groupby([group_col, "date"]):
        group = group.sort_values("connectionTime")

        if len(group) < 3:
            continue

        cars = []
        for _, row in group.iterrows():
            cars.append({
                "arrival": row["connectionTime"],
                "departure": row["disconnectTime"],
                "energy_needed": float(row["energy_needed"]),
                "remaining": float(row["energy_needed"]),
                "rate": float(row["energy_rate"]),
                "wait": 0.0,
            })

        episodes.append(cars)

    return episodes


class EVChargingEnv(gym.Env):
    def __init__(self, episodes, max_cars=MAX_CARS, dt=DT_HOURS):
        super().__init__()

        self.episodes = episodes
        self.max_cars = max_cars
        self.dt = dt

        # For each car:
        # remaining energy, time left, waiting time, charging flag, charge rate
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(self.max_cars * 5,),
            dtype=np.float32
        )

        # Action = choose which car index to charge
        self.action_space = spaces.Discrete(self.max_cars)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode = random.choice(self.episodes)
        self.episode = [car.copy() for car in self.episode]

        self.start_time = min(car["arrival"] for car in self.episode)
        self.end_time = max(car["departure"] for car in self.episode)

        self.current_time = self.start_time
        self.active_cars = []
        self.arrived = set()

        return self._get_state(), {}

    def _add_arrivals(self):
        for i, car in enumerate(self.episode):
            if i not in self.arrived and car["arrival"] <= self.current_time:
                self.active_cars.append(car)
                self.arrived.add(i)

    def _remove_departed(self):
        penalty = 0

        remaining_active = []

        for car in self.active_cars:
            if car["departure"] <= self.current_time:
                if car["remaining"] > 0:
                    penalty -= car["remaining"] * 5
            else:
                remaining_active.append(car)

        self.active_cars = remaining_active
        return penalty

    def _get_state(self):
        state = []

        sorted_cars = sorted(
            self.active_cars,
            key=lambda c: c["departure"]
        )[:self.max_cars]

        for car in sorted_cars:
            time_left = max(
                (car["departure"] - self.current_time).total_seconds() / 3600,
                0
            )

            state.extend([
                car["remaining"],
                time_left,
                car["wait"],
                0.0,
                car["rate"],
            ])

        while len(state) < self.max_cars * 5:
            state.extend([0, 0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

    def step(self, action):
        reward = 0

        self._add_arrivals()

        if len(self.active_cars) == 0:
            self.current_time += pd.Timedelta(hours=self.dt)
            done = self.current_time >= self.end_time
            return self._get_state(), reward, done, False, {}

        sorted_cars = sorted(
            self.active_cars,
            key=lambda c: c["departure"]
        )[:self.max_cars]

        if action >= len(sorted_cars):
            reward -= 2
        else:
            selected_car = sorted_cars[action]

            delivered = min(
                selected_car["rate"] * self.dt,
                selected_car["remaining"]
            )

            selected_car["remaining"] -= delivered
            reward += delivered * 2

            for car in self.active_cars:
                if car is not selected_car:
                    car["wait"] += self.dt
                    reward -= 0.05

        reward += self._remove_departed()

        self.current_time += pd.Timedelta(hours=self.dt)

        done = self.current_time >= self.end_time

        return self._get_state(), reward, done, False, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--group_col", default="siteID")
    args = parser.parse_args()

    df = load_dataset(args.data)

    episodes = build_episodes(df, group_col=args.group_col)

    print("Total episodes:", len(episodes))

    env = EVChargingEnv(episodes)
    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=args.timesteps)

    model.save("ev_rl_priority_model")

    joblib.dump({
        "episodes": episodes,
        "max_cars": MAX_CARS,
        "dt_hours": DT_HOURS,
    }, "ev_rl_metadata.pkl")

    print("Model saved as ev_rl_priority_model.zip")


if __name__ == "__main__":
    main()