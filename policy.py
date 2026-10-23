"""Turn predictions into a maintenance decision and price it.

Every engine in the training set runs to failure, so with out-of-fold predictions we
can replay its life cycle by cycle and ask: when would each policy have called for
maintenance, and was that early enough?

Rules (all costs are ASSUMPTIONS, in arbitrary units; change them in the app):
  - planning a maintenance slot needs LEAD cycles of notice
  - alert fires the first cycle the policy's signal <= threshold
  - if true RUL at the alert >= LEAD -> planned maintenance: C_PLAN + C_WASTE * (true RUL - LEAD)
  - otherwise (alert too late, or never) -> unplanned failure: C_FAIL
"""
import numpy as np
import pandas as pd

LEAD, C_FAIL, C_PLAN, C_WASTE = 15, 100.0, 10.0, 0.2


def simulate(df, signal, threshold, lead=LEAD, c_fail=C_FAIL, c_plan=C_PLAN, c_waste=C_WASTE):
    d = df[["unit", "cycle", "rul_true"]].copy()
    d["sig"] = signal
    rows = []
    for u, g in d.groupby("unit", sort=False):
        hit = g[g["sig"] <= threshold]
        if len(hit) and hit.iloc[0]["rul_true"] >= lead:
            waste = hit.iloc[0]["rul_true"] - lead
            rows.append((u, False, waste, c_plan + c_waste * waste))
        else:
            rows.append((u, True, 0.0, c_fail))
    r = pd.DataFrame(rows, columns=["unit", "failed", "wasted_cycles", "cost"])
    return {"failures": int(r.failed.sum()), "engines": len(r),
            "avg_wasted_cycles": round(float(r.loc[~r.failed, "wasted_cycles"].mean() if (~r.failed).any() else 0), 1),
            "cost_per_engine": round(float(r.cost.mean()), 2)}


def best_threshold(df, signal, grid, **kw):
    res = [(simulate(df, signal, t, **kw), t) for t in grid]
    res.sort(key=lambda x: x[0]["cost_per_engine"])
    return {**res[0][0], "threshold": res[0][1]}


def compare(df, point, p10, tune_units=None, **kw):
    """Fixed-interval baseline vs point forecast vs P10 (pessimistic) forecast.

    Each policy's threshold is chosen on `tune_units` engines and then scored on the
    remaining engines, so the reported cost is not tuned on the engines it is scored on.
    """
    df = df.reset_index(drop=True)
    signals = {"point_forecast": np.asarray(point), "p10_forecast": np.asarray(p10)}
    units = df["unit"].unique()
    tune_units = set(units[: len(units) // 2] if tune_units is None else tune_units)
    tmask = df["unit"].isin(tune_units).to_numpy()
    tune_df, eval_df = df[tmask], df[~tmask]
    out = {}
    fixed = []
    for T in range(100, 300, 2):
        r = simulate(tune_df, T - tune_df["cycle"].to_numpy(), 0, **kw)
        fixed.append((r["cost_per_engine"], T))
    T = min(fixed)[1]
    out["fixed_interval"] = {**simulate(eval_df, T - eval_df["cycle"].to_numpy(), 0, **kw), "rule": f"maintain every engine at cycle {T}"}
    for name, sig in signals.items():
        t = best_threshold(tune_df, sig[tmask], np.arange(0, 80, 1), **kw)["threshold"]
        out[name] = {**simulate(eval_df, sig[~tmask], t, **kw), "rule": f"alert when forecast <= {t} cycles"}
    out["note"] = f"thresholds tuned on {len(tune_units)} engines, scored on the other {len(units) - len(tune_units)}"
    return out
