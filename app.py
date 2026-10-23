"""Streamlit demo: streamlit run app.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from rul import data, features, model, policy

ROOT = Path(__file__).resolve().parent
# C-MAPSS sensor meanings (from the dataset documentation)
NAMES = {"s2": "LPC outlet temp", "s3": "HPC outlet temp", "s4": "LPT outlet temp", "s7": "HPC outlet pressure",
         "s8": "fan speed", "s9": "core speed", "s11": "HPC static pressure", "s12": "fuel flow / Ps30",
         "s13": "corrected fan speed", "s14": "corrected core speed", "s15": "bypass ratio", "s17": "bleed enthalpy",
         "s20": "HPT coolant bleed", "s21": "LPT coolant bleed", "cycle": "cycles run so far"}


def pretty(f):
    s, _, kind = f.partition("_")
    return f"{NAMES.get(s, s)} ({s}{', ' + kind if kind else ''})" if s in NAMES and s != "cycle" else NAMES.get(f, f)
st.set_page_config(page_title="Engine RUL Forecaster", layout="wide")


@st.cache_resource
def load_all():
    m = model.load_models(ROOT / "artifacts")
    train, test, sensors = data.load()
    ft = features.build(test, sensors)
    X = ft[m["cols"]]
    test = test.assign(pred=m["point"].predict(X), p10=m["q10"].predict(X), p90=m["q90"].predict(X))
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    oof = pd.read_csv(ROOT / "artifacts" / "oof_train.csv.gz")
    return m, test, X, metrics, oof, shap.TreeExplainer(m["point"])


m, test, X, metrics, oof, explainer = load_all()

st.title("Jet-engine Remaining Useful Life (RUL) forecaster")
st.caption("NASA C-MAPSS FD001 (public data). XGBoost point model + P10/P90 quantile models, engine-wise CV, SHAP explanations.")

with st.sidebar:
    st.header("Pick an engine")
    unit = st.selectbox("Test engine", sorted(test.unit.unique()), index=30)
    eng = test[test.unit == unit].reset_index()
    cyc = st.slider("Cycle (replay its life)", int(eng.cycle.min()), int(eng.cycle.max()), int(eng.cycle.max()))
    st.header("Maintenance rule")
    lead = st.number_input("Cycles of notice needed to plan maintenance", 5, 50, policy.LEAD)
    alert_at = st.number_input("Alert when forecast RUL ≤", 5, 80, 26)

tab1, tab2, tab3 = st.tabs(["Engine health", "Model performance", "Maintenance policy & cost"])

with tab1:
    row = eng[eng.cycle == cyc].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast RUL", f"{row.pred:.0f} cycles")
    c2.metric("80% interval", f"{row.p10:.0f} – {row.p90:.0f}")
    c3.metric("True RUL (capped at 125)", f"{row.rul:.0f}")
    status = "🔴 Maintain" if row.pred <= alert_at else ("🟠 Watch" if row.p10 <= alert_at else "🟢 Healthy")
    c4.metric("Status", status)

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.fill_between(eng.cycle, eng.p10, eng.p90, alpha=0.25, label="P10–P90")
        ax.plot(eng.cycle, eng.pred, label="forecast")
        ax.plot(eng.cycle, eng.rul, "k--", label="true (capped)")
        ax.axhline(alert_at, color="red", lw=0.8, ls=":", label="alert threshold")
        ax.axvline(cyc, color="grey", lw=0.8)
        ax.set_xlabel("cycle"); ax.set_ylabel("RUL (cycles)"); ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)
    with right:
        st.subheader("Why this forecast? (SHAP)")
        i = X.index[(test.unit == unit) & (test.cycle == cyc)][0]
        sv = explainer.shap_values(X.loc[[i]])[0]
        top = pd.Series(sv, index=[pretty(c) for c in m["cols"]]).sort_values(key=np.abs, ascending=False).head(8)[::-1]
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        ax2.barh(top.index, top.values, color=["#c0392b" if v < 0 else "#2e86c1" for v in top.values])
        ax2.set_xlabel("effect on RUL (cycles); red = pulls RUL down")
        st.pyplot(fig2)
    s = st.selectbox("Sensor trend", m["sensors"], index=m["sensors"].index("s11"), format_func=pretty)
    fig4, ax4 = plt.subplots(figsize=(10, 2.4))
    ax4.plot(eng.cycle, eng[s], lw=0.8, alpha=0.5, label="raw")
    ax4.plot(eng.cycle, eng[s].rolling(15, min_periods=1).mean(), lw=1.8, label="15-cycle mean")
    ax4.axvline(cyc, color="grey", lw=0.8); ax4.set_xlabel("cycle"); ax4.set_ylabel(s); ax4.legend(fontsize=8)
    st.pyplot(fig4)

with tab2:
    t = metrics["xgb_features_test"]
    b = metrics["baselines"]
    st.subheader("Official test set (100 engines, prediction at last observed cycle)")
    st.dataframe(pd.DataFrame([
        {"model": "Predict the mean", "RMSE": b["predict_mean"]["rmse"], "NASA score (lower=better)": b["predict_mean"]["nasa"]},
        {"model": "Ridge on raw sensors", "RMSE": b["ridge_raw_sensors"]["rmse"], "NASA score (lower=better)": b["ridge_raw_sensors"]["nasa"]},
        {"model": "XGBoost on raw sensors", "RMSE": b["xgb_raw_sensors"]["rmse"], "NASA score (lower=better)": b["xgb_raw_sensors"]["nasa"]},
        {"model": "XGBoost + time-series features (tuned)", "RMSE": t["rmse"], "NASA score (lower=better)": t["nasa"]},
    ]), hide_index=True)
    st.write(f"MAE {t['mae']:.1f} cycles · P10–P90 interval covers {t['interval_p10_p90_coverage']:.0%} of engines "
             f"(target 80%) · {t['late_predictions_over_20_cycles']} engines predicted >20 cycles too late")
    last = test.loc[test.groupby("unit").cycle.idxmax()]
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.scatter(last.rul, last.pred, s=12)
    ax3.plot([0, 130], [0, 130], "k--", lw=0.8)
    ax3.set_xlabel("true RUL"); ax3.set_ylabel("predicted RUL")
    st.pyplot(fig3)
    st.caption("Engine-wise 5-fold CV grid: " + ", ".join(f"depth {g['max_depth']}/mcw {g['min_child_weight']}: {g['cv_rmse']}" for g in metrics["tuning"]["grid_engine_wise_cv"]))

with tab3:
    st.subheader("Does the forecast save money vs. fixed-interval maintenance?")
    st.caption("Replays 100 run-to-failure training engines using out-of-fold predictions. Thresholds are tuned on 50 engines "
               "and scored on the other 50. All costs are assumptions: change them.")
    a, b2, c = st.columns(3)
    c_fail = a.number_input("Cost of an unplanned failure", 10.0, 1000.0, policy.C_FAIL)
    c_plan = b2.number_input("Cost of a planned maintenance", 1.0, 100.0, policy.C_PLAN)
    c_waste = c.number_input("Cost per cycle of life thrown away", 0.0, 5.0, policy.C_WASTE)

    @st.cache_data
    def run_policy(lead, c_fail, c_plan, c_waste):
        return policy.compare(oof, oof.pred, oof.p10, lead=lead, c_fail=c_fail, c_plan=c_plan, c_waste=c_waste)

    res = run_policy(int(lead), c_fail, c_plan, c_waste)
    st.dataframe(pd.DataFrame([{"policy": k, **v} for k, v in res.items() if k != "note"]), hide_index=True)
    fx, pt = res["fixed_interval"]["cost_per_engine"], res["point_forecast"]["cost_per_engine"]
    st.success(f"Forecast-based maintenance: {pt} vs fixed interval {fx} cost units per engine "
               f"({(fx - pt) / fx:.0%} lower under these assumptions). {res['note'].capitalize()}.")
