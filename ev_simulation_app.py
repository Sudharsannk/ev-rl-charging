"""
EV Charging Station — Live Priority Simulation
Run:  streamlit run ev_simulation_app.py
Requires: ev_priority_model.pkl  (from ev_model_training.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Charging Priority Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0A0F1E; color: #E2E8F0; }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    border: 1px solid #1E40AF33;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, #3B82F620 0%, transparent 60%);
  }
  .main-header h1 {
    font-size: 2rem; font-weight: 800; color: #F1F5F9;
    letter-spacing: -0.03em; margin: 0 0 6px 0;
  }
  .main-header p { color: #94A3B8; font-size: 0.95rem; margin: 0; }
  .pulse-dot {
    display: inline-block; width: 10px; height: 10px;
    background: #22C55E; border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px #22C55E;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.2)} }

  /* Station card */
  .station-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
  }
  .station-card.charging { border-color: #22C55E40; background: #052e1620; }
  .station-card.idle     { border-color: #6B728040; }

  /* Queue card */
  .queue-card {
    background: #111827;
    border-left: 4px solid #3B82F6;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.3s;
  }
  .queue-card.rank-1 { border-left-color: #F59E0B; background: #1a140520; }
  .queue-card.rank-2 { border-left-color: #3B82F6; }
  .queue-card.rank-3 { border-left-color: #8B5CF6; }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
  }
  .badge-green  { background: #052e1660; color: #4ADE80; border: 1px solid #22C55E40; }
  .badge-amber  { background: #451a0360; color: #FCD34D; border: 1px solid #F59E0B40; }
  .badge-blue   { background: #1e3a8a60; color: #93C5FD; border: 1px solid #3B82F640; }
  .badge-red    { background: #4c0519; color: #FCA5A5; border: 1px solid #EF444440; }

  /* Metric box */
  .metric-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-box .val { font-size: 1.5rem; font-weight: 700; color: #F1F5F9; font-family: 'JetBrains Mono'; }
  .metric-box .lbl { font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }

  /* Decision banner */
  .decision-preempt {
    background: linear-gradient(135deg, #4c0519, #1a0a05);
    border: 1px solid #EF444440;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 12px 0;
  }
  .decision-continue {
    background: linear-gradient(135deg, #052e16, #0a1a10);
    border: 1px solid #22C55E40;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 12px 0;
  }

  /* Progress bar custom */
  .progress-wrap { background: #1F2937; border-radius: 6px; overflow: hidden; height: 8px; }
  .progress-fill { height: 100%; border-radius: 6px; transition: width 0.5s; }

  /* Urgency bar */
  .urgency-bar { height: 6px; border-radius: 4px; margin-top: 4px; }

  /* Timeline event */
  .event-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid #1F2937;
    font-size: 0.85rem;
  }
  .event-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }

  /* Input form */
  .stNumberInput > div > div > input {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
  }
  .stSlider > div > div > div { background: #3B82F6 !important; }

  /* Scrollable log */
  .log-container {
    background: #050B14;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px;
    height: 280px;
    overflow-y: auto;
    font-family: 'JetBrains Mono';
    font-size: 0.78rem;
    color: #64748B;
  }
  .log-line-green { color: #4ADE80; }
  .log-line-amber { color: #FCD34D; }
  .log-line-blue  { color: #93C5FD; }
  .log-line-red   { color: #FCA5A5; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "incumbent": None,
        "queue": [],
        "history": [],
        "car_counter": 1,
        "model_loaded": False,
        "model": None,
        "features": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        payload = joblib.load("ev_priority_model.pkl")
        return payload["model"], payload["features"]
    except FileNotFoundError:
        return None, None

model, feat_cols = load_model()
if model is not None:
    st.session_state.model = model
    st.session_state.features = feat_cols
    st.session_state.model_loaded = True

# ── Core logic ─────────────────────────────────────────────────────────────────
FEAT_COLS = [
    "inc_energy_needed","inc_energy_delivered","inc_remaining_energy",
    "inc_time_to_depart","inc_req_depart","inc_energy_rate",
    "new_energy_needed","new_time_to_depart","new_req_depart","new_energy_rate",
    "inc_urgency","new_urgency","urgency_ratio","inc_completion_time",
]

def urgency(energy_needed, time_to_depart):
    return energy_needed / max(time_to_depart, 0.1)

def rank_queue_cars(incumbent, queue):
    if not queue:
        return []
    if not st.session_state.model_loaded:
        # fallback: rank by urgency
        scored = [{ **c, "urgency_score": urgency(c["energy_needed"], c["time_to_depart"]),
                    "preempt_prob": urgency(c["energy_needed"], c["time_to_depart"]) / 20,
                    "priority_rank": 0 } for c in queue]
        scored.sort(key=lambda x: x["urgency_score"], reverse=True)
        for i, c in enumerate(scored): c["priority_rank"] = i+1
        return scored

    m = st.session_state.model
    inc_rem  = max(incumbent["energy_needed"] - incumbent["energy_delivered"], 0)
    inc_time = max(incumbent["time_to_depart"], 0.1)
    inc_rate = max(incumbent["energy_rate"], 0.5)
    inc_urg  = inc_rem / inc_time
    inc_ct   = inc_rem / inc_rate

    rows, enriched = [], []
    for c in queue:
        nc_need = c["energy_needed"]
        nc_time = max(c["time_to_depart"], 0.1)
        nc_rate = max(c["energy_rate"], 0.5)
        nc_urg  = nc_need / nc_time
        ratio   = nc_urg / (inc_urg + 1e-3)
        rows.append({
            "inc_energy_needed"   : incumbent["energy_needed"],
            "inc_energy_delivered": incumbent["energy_delivered"],
            "inc_remaining_energy": inc_rem,
            "inc_time_to_depart"  : inc_time,
            "inc_req_depart"      : inc_time,
            "inc_energy_rate"     : inc_rate,
            "new_energy_needed"   : nc_need,
            "new_time_to_depart"  : nc_time,
            "new_req_depart"      : nc_time,
            "new_energy_rate"     : nc_rate,
            "inc_urgency"         : inc_urg,
            "new_urgency"         : nc_urg,
            "urgency_ratio"       : ratio,
            "inc_completion_time" : inc_ct,
        })
        enriched.append({**c, "urgency_score": round(nc_urg, 3)})

    X = pd.DataFrame(rows)[FEAT_COLS]
    probs = m.predict_proba(X)[:, 1]
    for i, c in enumerate(enriched):
        c["preempt_prob"] = round(float(probs[i]), 4)

    enriched.sort(key=lambda x: x["preempt_prob"], reverse=True)
    for i, c in enumerate(enriched): c["priority_rank"] = i + 1
    return enriched

def log(msg, level="blue"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({"ts": ts, "msg": msg, "level": level})

def car_label(car):
    return f"🚗 Car #{car['id']} — {car['name']}"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚡ EV Charging Priority Simulator</h1>
  <p><span class="pulse-dot"></span>
     Real-time queue management · AI-powered preemption decisions · Grid stability</p>
</div>
""", unsafe_allow_html=True)

# ── Model status bar ──────────────────────────────────────────────────────────
if st.session_state.model_loaded:
    st.success("✅ **Model loaded** — XGBoost priority model ready. All decisions are AI-driven.")
else:
    st.warning("⚠️ **Model not found** (`ev_priority_model.pkl`). Run `python ev_model_training.py` first. "
               "Showing urgency-based fallback rankings.")

st.markdown("---")

# ── Layout ────────────────────────────────────────────────────────────────────
left, mid, right = st.columns([1.1, 1.4, 1.1])

# ═══════════════════════════════════════════════════════
# LEFT PANEL — Add a Car
# ═══════════════════════════════════════════════════════
with left:
    st.markdown("### ➕ Add a Car")

    with st.form("add_car_form", clear_on_submit=True):
        car_name = st.text_input("Car name / plate", placeholder=f"e.g. Tesla Model 3")
        c1, c2 = st.columns(2)
        with c1:
            energy_needed = st.number_input("Energy needed (kWh)", min_value=1.0, max_value=150.0, value=20.0, step=0.5)
        with c2:
            time_to_depart = st.number_input("Hours until departure", min_value=0.1, max_value=24.0, value=3.0, step=0.25)

        c3, c4 = st.columns(2)
        with c3:
            energy_rate = st.number_input("Charge rate (kW)", min_value=1.0, max_value=150.0, value=7.4, step=0.5)
        with c4:
            energy_delivered = st.number_input("Already charged (kWh)", min_value=0.0, max_value=150.0, value=0.0, step=0.5,
                                                help="Non-zero only if this car is already charging")

        add_btn = st.form_submit_button("⚡ Add Car", use_container_width=True, type="primary")

    if add_btn:
        car = {
            "id"              : st.session_state.car_counter,
            "name"            : car_name or f"Car #{st.session_state.car_counter}",
            "energy_needed"   : energy_needed,
            "energy_delivered": energy_delivered,
            "time_to_depart"  : time_to_depart,
            "energy_rate"     : energy_rate,
            "added_at"        : datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.car_counter += 1

        if st.session_state.incumbent is None:
            st.session_state.incumbent = car
            log(f"Car #{car['id']} ({car['name']}) started charging", "green")
            st.rerun()
        else:
            st.session_state.queue.append(car)
            log(f"Car #{car['id']} ({car['name']}) joined queue (pos {len(st.session_state.queue)})", "blue")
            st.rerun()

    # Quick actions
    st.markdown("#### 🎛 Station Controls")

    if st.session_state.incumbent:
        inc = st.session_state.incumbent
        done_pct = min(inc["energy_delivered"] / max(inc["energy_needed"], 0.1) * 100, 100)
        inc_done = done_pct >= 95

        if st.button("✅ Incumbent Done Charging", use_container_width=True, disabled=not bool(st.session_state.incumbent)):
            old = st.session_state.incumbent
            log(f"Car #{old['id']} ({old['name']}) finished and left", "green")
            # promote top of ranked queue
            if st.session_state.queue:
                ranked = rank_queue_cars(old, st.session_state.queue)
                next_car = ranked[0]
                st.session_state.queue = [c for c in st.session_state.queue if c["id"] != next_car["id"]]
                next_car.pop("priority_rank", None)
                next_car.pop("preempt_prob", None)
                next_car.pop("urgency_score", None)
                st.session_state.incumbent = next_car
                log(f"Car #{next_car['id']} ({next_car['name']}) promoted to charger (was #1 in queue)", "amber")
            else:
                st.session_state.incumbent = None
                log("Station is now idle", "blue")
            st.rerun()

        if st.button("🚫 Remove Incumbent (Left Early)", use_container_width=True):
            old = st.session_state.incumbent
            log(f"Car #{old['id']} ({old['name']}) left early without completing", "red")
            if st.session_state.queue:
                ranked = rank_queue_cars(old, st.session_state.queue)
                next_car = ranked[0]
                st.session_state.queue = [c for c in st.session_state.queue if c["id"] != next_car["id"]]
                next_car.pop("priority_rank", None)
                next_car.pop("preempt_prob", None)
                next_car.pop("urgency_score", None)
                st.session_state.incumbent = next_car
                log(f"Car #{next_car['id']} ({next_car['name']}) auto-promoted to charger", "amber")
            else:
                st.session_state.incumbent = None
            st.rerun()

    if st.session_state.queue:
        st.markdown("##### Remove a queued car")
        q_names = {f"Car #{c['id']} — {c['name']}": c["id"] for c in st.session_state.queue}
        sel = st.selectbox("Select car to remove", list(q_names.keys()), label_visibility="collapsed")
        if st.button("🗑 Remove selected", use_container_width=True):
            cid = q_names[sel]
            removed = next(c for c in st.session_state.queue if c["id"] == cid)
            st.session_state.queue = [c for c in st.session_state.queue if c["id"] != cid]
            log(f"Car #{removed['id']} ({removed['name']}) removed from queue", "red")
            st.rerun()

    if st.button("🔄 Reset Everything", use_container_width=True):
        for k in ["incumbent","queue","history","car_counter"]:
            st.session_state[k] = None if k in ["incumbent"] else ([] if k in ["queue","history"] else 1)
        st.rerun()

# ═══════════════════════════════════════════════════════
# MID PANEL — Station Status + Ranked Queue
# ═══════════════════════════════════════════════════════
with mid:
    st.markdown("### 🔌 Charging Station")

    # ── Incumbent ──────────────────────────────────────
    if st.session_state.incumbent is None:
        st.markdown("""
        <div class="station-card idle">
          <div style="text-align:center; padding:30px 0; color:#4B5563;">
            <div style="font-size:2.5rem;">🔌</div>
            <div style="font-size:1.1rem;font-weight:600;margin-top:8px;">Station Idle</div>
            <div style="font-size:0.85rem;margin-top:4px;">Add a car to begin</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        inc = st.session_state.incumbent
        inc_rem  = max(inc["energy_needed"] - inc["energy_delivered"], 0)
        done_pct = min(inc["energy_delivered"] / max(inc["energy_needed"], 0.1) * 100, 100)
        inc_urg  = urgency(inc_rem, inc["time_to_depart"])
        eta_hr   = inc_rem / max(inc["energy_rate"], 0.5)

        urg_color = "#EF4444" if inc_urg > 8 else "#F59E0B" if inc_urg > 3 else "#22C55E"

        st.markdown(f"""
        <div class="station-card charging">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
            <div>
              <div style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">⚡ {inc['name']}</div>
              <div style="font-size:0.75rem;color:#64748B;">Car #{inc['id']} · Connected {inc['added_at']}</div>
            </div>
            <span class="badge badge-green">CHARGING</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:14px;">
            <div class="metric-box">
              <div class="val">{inc['energy_delivered']:.1f}</div>
              <div class="lbl">kWh Delivered</div>
            </div>
            <div class="metric-box">
              <div class="val">{inc_rem:.1f}</div>
              <div class="lbl">kWh Remaining</div>
            </div>
            <div class="metric-box">
              <div class="val">{inc['time_to_depart']:.1f}h</div>
              <div class="lbl">Time to Depart</div>
            </div>
          </div>
          <div style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94A3B8;margin-bottom:4px;">
              <span>Charge progress</span><span>{done_pct:.0f}%</span>
            </div>
            <div class="progress-wrap">
              <div class="progress-fill" style="width:{done_pct}%;background:linear-gradient(90deg,#22C55E,#4ADE80);"></div>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748B;margin-top:10px;">
            <span>Rate: <b style="color:#93C5FD;">{inc['energy_rate']:.1f} kW</b></span>
            <span>ETA to full: <b style="color:#FCD34D;">{eta_hr:.1f}h</b></span>
            <span>Urgency: <b style="color:{urg_color};">{inc_urg:.2f}</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Queue ───────────────────────────────────────────
    st.markdown("### 📋 Priority Queue")

    if not st.session_state.queue:
        st.markdown("""
        <div style="text-align:center;padding:24px;color:#4B5563;border:1px dashed #1F2937;border-radius:12px;">
          No cars waiting · Add a car to see the AI priority ranking
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.incumbent:
            ranked = rank_queue_cars(st.session_state.incumbent, st.session_state.queue)
        else:
            ranked = []
            for i, c in enumerate(st.session_state.queue):
                ranked.append({**c,
                    "urgency_score": urgency(c["energy_needed"], c["time_to_depart"]),
                    "preempt_prob": 0.5, "priority_rank": i+1})

        rank_colors = ["#F59E0B", "#3B82F6", "#8B5CF6", "#10B981", "#EC4899"]
        rank_labels = ["🥇 NEXT", "🥈 2ND", "🥉 3RD"] + [f"#{i+1}" for i in range(3, 20)]

        for c in ranked:
            r = c["priority_rank"] - 1
            color = rank_colors[min(r, len(rank_colors)-1)]
            rl    = rank_labels[min(r, len(rank_labels)-1)]
            urg   = c.get("urgency_score", 0)
            prob  = c.get("preempt_prob", 0)
            urg_pct = min(urg / 15 * 100, 100)
            urg_c = "#EF4444" if urg > 8 else "#F59E0B" if urg > 3 else "#22C55E"

            st.markdown(f"""
            <div class="queue-card rank-{min(c['priority_rank'],3)}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div>
                  <span style="color:{color};font-weight:700;font-size:0.85rem;">{rl}</span>
                  <span style="margin-left:10px;color:#F1F5F9;font-weight:600;">{c['name']}</span>
                  <span style="color:#64748B;font-size:0.75rem;margin-left:6px;">#{c['id']}</span>
                </div>
                <span class="badge badge-blue">WAITING</span>
              </div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:0.78rem;margin-bottom:10px;">
                <div style="color:#94A3B8;">Need: <b style="color:#F1F5F9;">{c['energy_needed']:.1f} kWh</b></div>
                <div style="color:#94A3B8;">Departs: <b style="color:#F1F5F9;">{c['time_to_depart']:.1f}h</b></div>
                <div style="color:#94A3B8;">Rate: <b style="color:#93C5FD;">{c['energy_rate']:.1f} kW</b></div>
              </div>
              <div style="font-size:0.74rem;color:#64748B;margin-bottom:4px;">
                Urgency score &nbsp;
                <span style="color:{urg_c};font-weight:600;">{urg:.3f}</span>
                &nbsp;·&nbsp;
                Preempt probability &nbsp;
                <span style="color:#FCD34D;font-weight:600;">{prob:.1%}</span>
              </div>
              <div class="progress-wrap">
                <div class="progress-fill urgency-bar" style="width:{urg_pct:.1f}%;background:{urg_c};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Preemption decision banner
        if st.session_state.incumbent and ranked:
            top = ranked[0]
            if top.get("preempt_prob", 0) >= 0.5:
                st.markdown(f"""
                <div class="decision-preempt">
                  <div style="font-weight:700;color:#FCA5A5;font-size:1rem;">🔴 RECOMMENDATION: PREEMPT</div>
                  <div style="color:#FDA4AF;font-size:0.85rem;margin-top:6px;">
                    Stop <b>{st.session_state.incumbent['name']}</b> and start charging
                    <b>{top['name']}</b> (urgency ratio {top['preempt_prob']:.1%} confidence).
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="decision-continue">
                  <div style="font-weight:700;color:#4ADE80;font-size:1rem;">🟢 RECOMMENDATION: CONTINUE</div>
                  <div style="color:#86EFAC;font-size:0.85rem;margin-top:6px;">
                    Keep charging <b>{st.session_state.incumbent['name']}</b>.
                    Queue cars can wait without grid impact.
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# RIGHT PANEL — Metrics + Log
# ═══════════════════════════════════════════════════════
with right:
    st.markdown("### 📊 Grid Metrics")

    total_in_system  = (1 if st.session_state.incumbent else 0) + len(st.session_state.queue)
    total_energy_dem = sum(c["energy_needed"] for c in st.session_state.queue)
    if st.session_state.incumbent:
        total_energy_dem += max(st.session_state.incumbent["energy_needed"] - st.session_state.incumbent["energy_delivered"], 0)
    active_kw = st.session_state.incumbent["energy_rate"] if st.session_state.incumbent else 0

    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:10px;">
          <div class="val">{total_in_system}</div>
          <div class="lbl">Cars in System</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:10px;">
          <div class="val">{len(st.session_state.queue)}</div>
          <div class="lbl">In Queue</div>
        </div>
        """, unsafe_allow_html=True)

    cols2 = st.columns(2)
    with cols2[0]:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:10px;">
          <div class="val">{active_kw:.1f}</div>
          <div class="lbl">Active kW</div>
        </div>
        """, unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:10px;">
          <div class="val">{total_energy_dem:.0f}</div>
          <div class="lbl">Total kWh Demand</div>
        </div>
        """, unsafe_allow_html=True)

    # Urgency comparison
    if st.session_state.incumbent and st.session_state.queue:
        st.markdown("#### ⚡ Urgency Comparison")
        inc = st.session_state.incumbent
        inc_rem = max(inc["energy_needed"] - inc["energy_delivered"], 0)
        all_cars = [{"name": f"[NOW] {inc['name']}", "urg": urgency(inc_rem, inc["time_to_depart"]), "is_inc": True}]
        for c in st.session_state.queue:
            all_cars.append({"name": c["name"], "urg": urgency(c["energy_needed"], c["time_to_depart"]), "is_inc": False})

        max_urg = max(x["urg"] for x in all_cars) or 1
        for car in all_cars:
            pct = car["urg"] / max_urg * 100
            col = "#22C55E" if car["is_inc"] else ("#EF4444" if car["urg"] > 8 else "#F59E0B" if car["urg"] > 3 else "#3B82F6")
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94A3B8;margin-bottom:3px;">
                <span>{'⚡' if car['is_inc'] else '🕐'} {car['name'][:20]}</span>
                <span style="color:{col};font-weight:600;">{car['urg']:.3f}</span>
              </div>
              <div class="progress-wrap">
                <div class="progress-fill" style="width:{pct:.1f}%;background:{col};height:8px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Event log
    st.markdown("#### 📜 Event Log")
    log_class = {"green": "log-line-green", "amber": "log-line-amber",
                 "blue": "log-line-blue", "red": "log-line-red"}

    log_lines = ""
    for ev in reversed(st.session_state.history[-40:]):
        cls = log_class.get(ev["level"], "")
        log_lines += f'<div class="event-row"><span class="event-dot" style="background:{"#22C55E" if ev["level"]==chr(103)+"reen" else "#F59E0B" if ev["level"]=="amber" else "#3B82F6" if ev["level"]=="blue" else "#EF4444"};"></span><span style="color:#475569;min-width:52px;">{ev["ts"]}</span><span class="{cls}">{ev["msg"]}</span></div>'

    if not log_lines:
        log_lines = '<div style="color:#374151;padding:20px 0;text-align:center;">No events yet</div>'

    st.markdown(f'<div class="log-container">{log_lines}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#374151;font-size:0.78rem;padding:12px 0;">
  Decision factors: <b>energy urgency</b> (kWh/hr) · <b>time to departure</b> · <b>charge rate</b> · <b>remaining charge</b>
  · XGBoost priority classifier trained on ACN-Data
</div>
""", unsafe_allow_html=True)