"""
EV Charging Station — Multi-Car Priority Model
================================================
This script:
  1. Loads & cleans the ACN-Data CSV
  2. Extracts nested userInputs
  3. Engineers all features
  4. Builds pairwise conflict dataset (incumbent vs newcomer)
  5. Trains an XGBoost classifier
  6. Saves the model as  ev_priority_model.pkl

Run:
    python ev_model_training.py --data acndata_sessions_new.csv

Then use ev_simulation_app.py for the live Streamlit demo.
"""

import argparse, ast, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────
URGENCY_THRESHOLD      = 1.5   # newcomer urgency must exceed incumbent by this ratio
SAFETY_BUFFER          = 1.2   # incumbent must have 20 % slack time to be preempted
DEFAULT_RATE_KW        = 6.0
DEFAULT_ENERGY_KWH     = 10.0
DEFAULT_DEPART_HR      = 8.0
MODEL_OUT              = "ev_priority_model.pkl"
FEATURE_COLS = [
    "inc_energy_needed", "inc_energy_delivered", "inc_remaining_energy",
    "inc_time_to_depart", "inc_req_depart", "inc_energy_rate",
    "new_energy_needed", "new_time_to_depart", "new_req_depart", "new_energy_rate",
    "inc_urgency", "new_urgency", "urgency_ratio", "inc_completion_time",
]


# ── Step 1: Load & parse ──────────────────────────────────────
def load_and_parse(csv_path: str) -> pd.DataFrame:
    print(f"\n[1] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Rows: {len(df):,}  Cols: {len(df.columns)}")

    def parse_dt(s):
        if pd.isna(s):
            return pd.NaT
        return pd.to_datetime(s, format="%a, %d %b %Y %H:%M:%S GMT", utc=True)

    print("[2] Parsing datetime columns …")
    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col] = df[col].apply(parse_dt)
    return df


# ── Step 2: Flatten userInputs ────────────────────────────────
def extract_user_inputs(df: pd.DataFrame) -> pd.DataFrame:
    print("[3] Extracting userInputs …")

    def parse_dt(s):
        try:
            return pd.to_datetime(s, format="%a, %d %b %Y %H:%M:%S GMT", utc=True)
        except Exception:
            return pd.NaT

    def extract(raw):
        try:
            e = ast.literal_eval(raw)[-1]
            return {
                "kWhRequested"      : float(e.get("kWhRequested", np.nan)),
                "milesRequested"    : float(e.get("milesRequested", np.nan)),
                "minutesAvailable"  : float(e.get("minutesAvailable", np.nan)),
                "WhPerMile"         : float(e.get("WhPerMile", np.nan)),
                "requestedDeparture": parse_dt(e.get("requestedDeparture")),
            }
        except Exception:
            return {k: (np.nan if k != "requestedDeparture" else pd.NaT)
                    for k in ["kWhRequested","milesRequested","minutesAvailable",
                               "WhPerMile","requestedDeparture"]}

    ui = df["userInputs"].apply(extract).apply(pd.Series)
    df = pd.concat([df, ui], axis=1)
    print(f"    Sessions with no userInputs: {df['kWhRequested'].isna().sum():,}")
    return df


# ── Step 3: Engineer features ─────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[4] Engineering features …")
    df["session_hr"]    = (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    df["charging_hr"]   = (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    df["energy_rate_kW"]= (df["kWhDelivered"] / df["charging_hr"].clip(lower=0.1)).fillna(DEFAULT_RATE_KW).clip(0.5, 150)
    df["energy_needed"] = df["kWhRequested"].fillna(df["kWhDelivered"] * 1.1).fillna(DEFAULT_ENERGY_KWH)
    df["req_depart_hr"] = ((df["requestedDeparture"] - df["connectionTime"]).dt.total_seconds() / 3600
                           ).fillna(df["session_hr"]).fillna(DEFAULT_DEPART_HR)
    df["remaining_kWh"] = (df["energy_needed"] - df["kWhDelivered"]).clip(lower=0)
    return df


# ── Step 4: Build pairwise conflict dataset ───────────────────
def build_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    For each session (incumbent) we sample 3 random newcomers.
    Each (incumbent, newcomer) pair becomes one training row.
    Label = 1 → preempt incumbent in favour of newcomer.
    """
    print("[5] Simulating conflict pairs …")
    np.random.seed(seed)
    n = len(df)
    records = []
    idx_pool = np.arange(n)

    for i in range(n):
        inc = df.iloc[i]
        newcomer_idxs = np.random.choice(idx_pool, size=3, replace=False)
        for j in newcomer_idxs:
            nc = df.iloc[j]

            inc_rem  = float(inc["remaining_kWh"])
            inc_time = max(float(inc["session_hr"] or DEFAULT_DEPART_HR), 0.1)
            inc_rate = max(float(inc["energy_rate_kW"]), 0.5)

            nc_need  = float(nc["energy_needed"])
            nc_time  = max(float(nc["session_hr"] or DEFAULT_DEPART_HR), 0.1)

            inc_urg  = inc_rem  / inc_time
            nc_urg   = nc_need  / nc_time
            ratio    = nc_urg   / (inc_urg + 1e-3)
            inc_ct   = inc_rem  / inc_rate         # hours to finish

            label = int(ratio > URGENCY_THRESHOLD and inc_time > inc_ct * SAFETY_BUFFER)

            records.append({
                "inc_energy_needed"    : float(inc["energy_needed"]),
                "inc_energy_delivered" : float(inc["kWhDelivered"]),
                "inc_remaining_energy" : inc_rem,
                "inc_time_to_depart"   : inc_time,
                "inc_req_depart"       : float(inc["req_depart_hr"]),
                "inc_energy_rate"      : inc_rate,
                "new_energy_needed"    : nc_need,
                "new_time_to_depart"   : nc_time,
                "new_req_depart"       : float(nc["req_depart_hr"]),
                "new_energy_rate"      : max(float(nc["energy_rate_kW"]), 0.5),
                "inc_urgency"          : inc_urg,
                "new_urgency"          : nc_urg,
                "urgency_ratio"        : ratio,
                "inc_completion_time"  : inc_ct,
                "label"                : label,
            })

    feat = pd.DataFrame(records)
    dist = feat["label"].value_counts().to_dict()
    print(f"    Pairs: {len(feat):,}  |  Continue={dist.get(0,0):,}  Preempt={dist.get(1,0):,}")
    return feat


# ── Step 5: Train ─────────────────────────────────────────────
def train(feat: pd.DataFrame):
    print("[6] Training XGBoost …")
    X = feat[FEATURE_COLS]
    y = feat["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scale_pos = (y == 0).sum() / (y == 1).sum()

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    print("\n── Evaluation ──────────────────────────────────")
    print(classification_report(y_te, y_pred, target_names=["Continue", "Preempt"]))
    print(f"ROC-AUC : {roc_auc_score(y_te, y_prob):.4f}")

    print("\n── Feature Importances ─────────────────────────")
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for k, v in fi.items():
        bar = "█" * int(v * 50)
        print(f"  {k:<25s} {bar} {v:.4f}")

    joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_OUT)
    print(f"\n[✓] Model saved → {MODEL_OUT}")
    return model


# ── Inference helper (used by Streamlit app) ──────────────────
def compute_urgency_score(energy_needed: float, time_to_depart: float) -> float:
    """Simple urgency = energy needed per hour until departure."""
    return energy_needed / max(time_to_depart, 0.1)


def rank_queue(
    incumbent: dict,
    queue: list[dict],
    model_path: str = MODEL_OUT,
) -> list[dict]:
    """
    Given one incumbent and N queued cars, returns the queue sorted by
    who should charge next (highest priority first).

    Each car dict must have:
        car_id, energy_needed_kWh, time_to_depart_hr,
        energy_delivered_kWh (0 for newcomers), energy_rate_kW

    Returns list of dicts with added fields:
        urgency_score, preempt_prob, priority_rank
    """
    payload = joblib.load(model_path)
    model, feat_cols = payload["model"], payload["features"]

    inc_rem  = max(incumbent["energy_needed_kWh"] - incumbent["energy_delivered_kWh"], 0)
    inc_time = max(incumbent["time_to_depart_hr"], 0.1)
    inc_rate = max(incumbent["energy_rate_kW"], 0.5)
    inc_urg  = inc_rem / inc_time
    inc_ct   = inc_rem / inc_rate

    enriched = []
    rows = []
    for car in queue:
        nc_need = car["energy_needed_kWh"]
        nc_time = max(car["time_to_depart_hr"], 0.1)
        nc_rate = max(car["energy_rate_kW"], 0.5)
        nc_urg  = nc_need / nc_time
        ratio   = nc_urg / (inc_urg + 1e-3)

        row = {
            "inc_energy_needed"    : incumbent["energy_needed_kWh"],
            "inc_energy_delivered" : incumbent["energy_delivered_kWh"],
            "inc_remaining_energy" : inc_rem,
            "inc_time_to_depart"   : inc_time,
            "inc_req_depart"       : inc_time,
            "inc_energy_rate"      : inc_rate,
            "new_energy_needed"    : nc_need,
            "new_time_to_depart"   : nc_time,
            "new_req_depart"       : nc_time,
            "new_energy_rate"      : nc_rate,
            "inc_urgency"          : inc_urg,
            "new_urgency"          : nc_urg,
            "urgency_ratio"        : ratio,
            "inc_completion_time"  : inc_ct,
        }
        rows.append(row)
        enriched.append({**car, "urgency_score": round(nc_urg, 4)})

    X = pd.DataFrame(rows)[feat_cols]
    probs = model.predict_proba(X)[:, 1]

    for i, car in enumerate(enriched):
        car["preempt_prob"] = round(float(probs[i]), 4)

    # Rank: sort by preempt_prob descending (highest urgency first)
    enriched.sort(key=lambda c: c["preempt_prob"], reverse=True)
    for rank, car in enumerate(enriched, 1):
        car["priority_rank"] = rank

    return enriched


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="acndata_sessions_new.csv")
    args = parser.parse_args()

    df   = load_and_parse(args.data)
    df   = extract_user_inputs(df)
    df   = engineer_features(df)
    feat = build_dataset(df)
    train(feat)


if __name__ == "__main__":
    main()