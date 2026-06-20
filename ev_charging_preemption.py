"""
EV Charging Station Preemption Decision System
================================================
Dataset: ACN-Data (acndata_sessions_new.csv)

PIPELINE OVERVIEW
─────────────────
Step 1 : Load & inspect raw data
Step 2 : Parse & clean datetime columns
Step 3 : Extract nested userInputs (JSON-like string → flat columns)
Step 4 : Engineer features (urgency, energy rate, remaining charge, etc.)
Step 5 : Simulate conflict scenarios (incumbent vs newcomer)
Step 6 : Generate rule-based labels
Step 7 : Train XGBoost classifier
Step 8 : Evaluate (classification report, ROC-AUC, feature importance)
Step 9 : Real-time inference function  ← use this in production

Usage
─────
  # Train once:
      python ev_charging_preemption.py --mode train --data acndata_sessions_new.csv

  # Predict in real time:
      python ev_charging_preemption.py --mode predict
"""

import argparse, ast, warnings, joblib, os
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MODEL_PATH       = "ev_preempt_model.pkl"
URGENCY_THRESHOLD = 1.5   # newcomer must be this much more urgent to preempt
TIME_BUFFER       = 1.2   # incumbent must have 20% slack to be safely preempted
DEFAULT_ENERGY_RATE_KW = 6.0
DEFAULT_ENERGY_KWH     = 10.0
DEFAULT_DEPART_HR      = 8.0


# ─────────────────────────────────────────────────────────────
# STEP 1-2: LOAD & PARSE
# ─────────────────────────────────────────────────────────────
def load_and_parse(csv_path: str) -> pd.DataFrame:
    """Load raw CSV and parse all datetime columns to UTC-aware Timestamps."""
    print(f"[1] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    → {len(df):,} rows, {len(df.columns)} columns")

    def parse_dt(s):
        if pd.isna(s):
            return pd.NaT
        # Format: 'Tue, 01 Jan 2019 03:45:49 GMT'
        return pd.to_datetime(s, format="%a, %d %b %Y %H:%M:%S GMT", utc=True)

    print("[2] Parsing datetime columns …")
    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col] = df[col].apply(parse_dt)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: EXTRACT USERINPUTS
# ─────────────────────────────────────────────────────────────
def extract_user_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    userInputs is a string representation of a list of dicts.
    We take the LAST entry (most recent user update) and flatten it.
    """
    print("[3] Extracting nested userInputs …")

    def parse_dt_str(s):
        try:
            return pd.to_datetime(s, format="%a, %d %b %Y %H:%M:%S GMT", utc=True)
        except Exception:
            return pd.NaT

    def extract(raw):
        try:
            entries = ast.literal_eval(raw)
            e = entries[-1]          # most recent user edit
            return {
                "kWhRequested"      : float(e.get("kWhRequested", np.nan)),
                "milesRequested"    : float(e.get("milesRequested", np.nan)),
                "minutesAvailable"  : float(e.get("minutesAvailable", np.nan)),
                "WhPerMile"         : float(e.get("WhPerMile", np.nan)),
                "requestedDeparture": parse_dt_str(e.get("requestedDeparture")),
            }
        except Exception:
            return {
                "kWhRequested"      : np.nan,
                "milesRequested"    : np.nan,
                "minutesAvailable"  : np.nan,
                "WhPerMile"         : np.nan,
                "requestedDeparture": pd.NaT,
            }

    ui_df = df["userInputs"].apply(extract).apply(pd.Series)
    df = pd.concat([df, ui_df], axis=1)

    null_ui = df["kWhRequested"].isna().sum()
    print(f"    → {null_ui:,} sessions have no userInputs (will use fallback estimates)")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all per-session features needed for conflict resolution."""
    print("[4] Engineering session-level features …")

    # Durations
    df["session_duration_hr"]  = (
        (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    df["charging_duration_hr"] = (
        (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    ).clip(lower=0.1)

    # Energy rate (kW)  — how fast this car has been / will be charged
    df["energy_rate_kW"] = (
        df["kWhDelivered"] / df["charging_duration_hr"]
    ).fillna(DEFAULT_ENERGY_RATE_KW).clip(lower=0.5, upper=150)

    # Energy needed — user declared, else fallback: 10% above what was delivered
    df["energy_needed_kWh"] = df["kWhRequested"].fillna(
        df["kWhDelivered"] * 1.1
    ).fillna(DEFAULT_ENERGY_KWH)

    # Requested departure offset in hours from connection
    df["req_depart_hr"] = (
        (df["requestedDeparture"] - df["connectionTime"]).dt.total_seconds() / 3600
    ).fillna(df["session_duration_hr"]).fillna(DEFAULT_DEPART_HR)

    # Remaining energy at "now" (used in conflict features below)
    df["remaining_energy_kWh"] = (
        df["energy_needed_kWh"] - df["kWhDelivered"]
    ).clip(lower=0)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 5-6: SIMULATE CONFLICTS & LABEL
# ─────────────────────────────────────────────────────────────
def build_conflict_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Each row represents a conflict scenario:
      incumbent  – car currently charging
      newcomer   – car that just arrived

    Label = 1 (PREEMPT incumbent) if:
      1. Newcomer urgency is URGENCY_THRESHOLD × higher than incumbent's
      2. Incumbent has enough slack time to complete charging even if interrupted
    """
    print("[5] Simulating conflict scenarios …")
    np.random.seed(seed)
    n = len(df)
    newcomer = df.iloc[np.random.randint(0, n, size=n)].reset_index(drop=True)
    inc      = df.reset_index(drop=True)

    feat = pd.DataFrame()

    # Incumbent features
    feat["inc_energy_needed"]    = inc["energy_needed_kWh"]
    feat["inc_energy_delivered"] = inc["kWhDelivered"]
    feat["inc_remaining_energy"] = inc["remaining_energy_kWh"]
    feat["inc_time_to_depart"]   = inc["session_duration_hr"].fillna(DEFAULT_DEPART_HR)
    feat["inc_req_depart"]       = inc["req_depart_hr"]
    feat["inc_energy_rate"]      = inc["energy_rate_kW"]

    # Newcomer features
    feat["new_energy_needed"]   = newcomer["energy_needed_kWh"]
    feat["new_time_to_depart"]  = newcomer["session_duration_hr"].fillna(DEFAULT_DEPART_HR)
    feat["new_req_depart"]      = newcomer["req_depart_hr"]
    feat["new_energy_rate"]     = newcomer["energy_rate_kW"]

    # Urgency = energy needed per hour until departure
    feat["inc_urgency"]  = feat["inc_remaining_energy"] / feat["inc_time_to_depart"].clip(lower=0.1)
    feat["new_urgency"]  = feat["new_energy_needed"]    / feat["new_time_to_depart"].clip(lower=0.1)
    feat["urgency_ratio"]= feat["new_urgency"] / (feat["inc_urgency"] + 1e-3)

    # Time incumbent needs to finish charging (hours)
    feat["inc_completion_time"] = feat["inc_remaining_energy"] / feat["inc_energy_rate"].clip(lower=0.1)

    print("[6] Generating labels …")
    feat["label"] = (
        (feat["urgency_ratio"] > URGENCY_THRESHOLD) &
        (feat["inc_time_to_depart"] > feat["inc_completion_time"] * TIME_BUFFER)
    ).astype(int)

    dist = feat["label"].value_counts().to_dict()
    print(f"    → Continue (0): {dist.get(0,0):,} | Preempt (1): {dist.get(1,0):,}")
    return feat


# ─────────────────────────────────────────────────────────────
# STEP 7: TRAIN MODEL
# ─────────────────────────────────────────────────────────────
def train_model(feat: pd.DataFrame) -> object:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    print("[7] Training XGBoost classifier …")
    X = feat.drop("label", axis=1)
    y = feat["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class imbalance correction
    scale = (y == 0).sum() / (y == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n─── Evaluation on Hold-Out Test Set ───")
    print(classification_report(y_test, y_pred, target_names=["Continue", "Preempt"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

    print("\n─── Feature Importances ───")
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for k, v in fi.items():
        print(f"  {k:<25s} {v:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n[✓] Model saved → {MODEL_PATH}")
    return model


# ─────────────────────────────────────────────────────────────
# STEP 9: REAL-TIME INFERENCE
# ─────────────────────────────────────────────────────────────
def predict_preemption(
    # --- Incumbent (currently charging) ---
    inc_energy_needed    : float,  # kWh user declared they need
    inc_energy_delivered : float,  # kWh delivered so far
    inc_time_to_depart   : float,  # hours until their declared departure
    inc_req_depart       : float,  # hours until requested departure (from connectionTime)
    inc_energy_rate      : float,  # kW rate at which car is currently charging

    # --- Newcomer (just arrived) ---
    new_energy_needed    : float,  # kWh the newcomer declared
    new_time_to_depart   : float,  # hours until newcomer departure
    new_req_depart       : float,  # hours: newcomer's requestedDeparture offset
    new_energy_rate      : float,  # kW: expected charge rate for newcomer

    model_path: str = MODEL_PATH,
) -> dict:
    """
    Returns a decision dict:
      {
        "decision"    : "PREEMPT" | "CONTINUE",
        "confidence"  : float (0–1),
        "reason"      : str,
        "inc_urgency" : float,
        "new_urgency" : float,
      }
    """
    model = joblib.load(model_path)

    inc_remaining       = max(inc_energy_needed - inc_energy_delivered, 0)
    inc_urgency         = inc_remaining / max(inc_time_to_depart, 0.1)
    new_urgency         = new_energy_needed / max(new_time_to_depart, 0.1)
    urgency_ratio       = new_urgency / (inc_urgency + 1e-3)
    inc_completion_time = inc_remaining / max(inc_energy_rate, 0.5)

    row = pd.DataFrame([{
        "inc_energy_needed"    : inc_energy_needed,
        "inc_energy_delivered" : inc_energy_delivered,
        "inc_remaining_energy" : inc_remaining,
        "inc_time_to_depart"   : inc_time_to_depart,
        "inc_req_depart"       : inc_req_depart,
        "inc_energy_rate"      : inc_energy_rate,
        "new_energy_needed"    : new_energy_needed,
        "new_time_to_depart"   : new_time_to_depart,
        "new_req_depart"       : new_req_depart,
        "new_energy_rate"      : new_energy_rate,
        "inc_urgency"          : inc_urgency,
        "new_urgency"          : new_urgency,
        "urgency_ratio"        : urgency_ratio,
        "inc_completion_time"  : inc_completion_time,
    }])

    prob    = model.predict_proba(row)[0, 1]
    decision = "PREEMPT" if prob >= 0.5 else "CONTINUE"

    reasons = []
    if urgency_ratio > URGENCY_THRESHOLD:
        reasons.append(f"Newcomer urgency {urgency_ratio:.1f}× higher than incumbent's")
    if inc_completion_time < inc_time_to_depart / TIME_BUFFER:
        reasons.append(f"Incumbent can finish in {inc_completion_time:.1f}h; departs in {inc_time_to_depart:.1f}h (safe to pause)")
    if not reasons:
        reasons.append("Incumbent is more urgent or cannot safely be interrupted")

    return {
        "decision"    : decision,
        "confidence"  : round(prob if decision == "PREEMPT" else 1 - prob, 4),
        "reason"      : " | ".join(reasons),
        "inc_urgency" : round(inc_urgency, 4),
        "new_urgency" : round(new_urgency, 4),
        "urgency_ratio": round(urgency_ratio, 4),
    }


# ─────────────────────────────────────────────────────────────
# INTERACTIVE DEMO
# ─────────────────────────────────────────────────────────────
def interactive_demo():
    print("\n━━━  EV Charging Preemption — Real-Time Demo  ━━━\n")
    print("Enter incumbent (currently charging) info:")
    inc_needed    = float(input("  Energy needed by incumbent (kWh)    : "))
    inc_delivered = float(input("  Energy delivered so far (kWh)        : "))
    inc_depart    = float(input("  Hours until incumbent departs        : "))
    inc_rate      = float(input("  Charging rate of incumbent (kW)      : "))

    print("\nEnter newcomer (just arrived) info:")
    new_needed    = float(input("  Energy needed by newcomer (kWh)      : "))
    new_depart    = float(input("  Hours until newcomer must leave      : "))
    new_rate      = float(input("  Expected charging rate for newcomer (kW): "))

    result = predict_preemption(
        inc_energy_needed    = inc_needed,
        inc_energy_delivered = inc_delivered,
        inc_time_to_depart   = inc_depart,
        inc_req_depart       = inc_depart,
        inc_energy_rate      = inc_rate,
        new_energy_needed    = new_needed,
        new_time_to_depart   = new_depart,
        new_req_depart       = new_depart,
        new_energy_rate      = new_rate,
    )

    print("\n━━━  DECISION  ━━━")
    print(f"  → {result['decision']}  (confidence: {result['confidence']:.1%})")
    print(f"  Reason      : {result['reason']}")
    print(f"  Inc urgency : {result['inc_urgency']:.3f} kW equivalent")
    print(f"  New urgency : {result['new_urgency']:.3f} kW equivalent")
    print(f"  Ratio       : {result['urgency_ratio']:.2f}×\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EV Charging Preemption System")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data", default="acndata_sessions_new.csv",
                        help="Path to raw CSV (required for --mode train)")
    args = parser.parse_args()

    if args.mode == "train":
        df   = load_and_parse(args.data)
        df   = extract_user_inputs(df)
        df   = engineer_features(df)
        feat = build_conflict_dataset(df)
        train_model(feat)
        print("\n[✓] Training complete. Run with --mode predict to test live decisions.")

    elif args.mode == "predict":
        if not os.path.exists(MODEL_PATH):
            print(f"[!] Model not found at '{MODEL_PATH}'. Run --mode train first.")
            return
        interactive_demo()


if __name__ == "__main__":
    main()
