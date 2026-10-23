"""End-to-end training + evaluation. Run: python train.py  (writes the model files and metrics.json next to this script)"""
import gzip
import json
import time
from pathlib import Path

import numpy as np
import shap

import data
import features
import model
import policy

ROOT = Path(__file__).resolve().parent


def run(quick=False):
    t0 = time.time()
    train, test, sensors = data.load()
    ftr = features.build(train, sensors)
    fte = features.build(test, sensors)
    cols = features.feature_columns(ftr)
    X, y, groups = ftr[cols], train["rul"], train["unit"]

    # official test protocol: one prediction per engine, at its last observed cycle
    last = fte.groupby("unit")["cycle"].idxmax()
    Xt, yt = fte.loc[last, cols], test.loc[last, "rul"]

    res = {"data": {"train_engines": int(train.unit.nunique()), "train_rows": len(train),
                    "test_engines": int(test.unit.nunique()), "sensors_used": sensors,
                    "n_features": len(cols), "rul_cap": data.RUL_CAP}}

    # baselines
    res["baselines"] = {
        "predict_mean": {"rmse": model.rmse(yt, np.full(len(yt), y.mean())), "nasa": model.nasa_score(yt, np.full(len(yt), y.mean()))},
    }
    r = model.ridge().fit(train[sensors], y)
    res["baselines"]["ridge_raw_sensors"] = {"rmse": model.rmse(yt, r.predict(test.loc[last, sensors])),
                                             "nasa": model.nasa_score(yt, r.predict(test.loc[last, sensors]))}
    xr = model.point_model().fit(train[sensors], y)
    res["baselines"]["xgb_raw_sensors"] = {"rmse": model.rmse(yt, xr.predict(test.loc[last, sensors])),
                                           "nasa": model.nasa_score(yt, xr.predict(test.loc[last, sensors]))}

    # tuned XGBoost on engineered features
    best, grid = ({}, []) if quick else model.tune(X, y, groups)
    res["tuning"] = {"best": best, "grid_engine_wise_cv": grid}
    point = model.point_model(**best).fit(X, y)
    q10 = model.quantile_model(0.1, **best).fit(X, y)
    q90 = model.quantile_model(0.9, **best).fit(X, y)
    p, lo, hi = point.predict(Xt), q10.predict(Xt), q90.predict(Xt)
    res["xgb_features_test"] = {
        "rmse": model.rmse(yt, p), "mae": float(np.mean(np.abs(yt - p))), "nasa": model.nasa_score(yt, p),
        "interval_p10_p90_coverage": float(np.mean((yt >= lo) & (yt <= hi))),
        "avg_interval_width": float(np.mean(hi - lo)),
        "late_predictions_over_20_cycles": int(np.sum(p - yt > 20)),
    }

    # maintenance policy on out-of-fold predictions over full run-to-failure trajectories
    op, olo, ohi = model.oof_predictions(X, y, groups, best)
    res["oof_rmse_all_cycles"] = model.rmse(y, op)
    res["maintenance_policy"] = policy.compare(train, op, olo)
    res["maintenance_policy_assumptions"] = {"lead_cycles": policy.LEAD, "cost_unplanned_failure": policy.C_FAIL,
                                             "cost_planned": policy.C_PLAN, "cost_per_wasted_cycle": policy.C_WASTE}

    # explanations
    expl = shap.TreeExplainer(point)
    sv = expl.shap_values(Xt)
    imp = sorted(zip(cols, np.abs(sv).mean(0)), key=lambda t: -t[1])[:12]
    res["shap_top_features"] = [{"feature": f, "mean_abs_shap": round(float(v), 2)} for f, v in imp]

    # native XGBoost JSON format: portable across xgboost versions (unlike pickle)
    for name, mdl in (("point", point), ("q10", q10), ("q90", q90)):
        raw = bytes(mdl.get_booster().save_raw("json"))
        (ROOT / f"{name}.json.gz").write_bytes(gzip.compress(raw, 9))
    (ROOT / "meta.json").write_text(json.dumps({"cols": cols, "sensors": sensors}))
    oof = train[["unit", "cycle", "rul_true"]].assign(pred=op, p10=olo, p90=ohi)
    oof.round(2).to_csv(ROOT / "oof_train.csv.gz", index=False, compression="gzip")
    res["runtime_sec"] = round(time.time() - t0, 1)

    def rnd(o):
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return round(o, 3) if isinstance(o, float) else o
    res = rnd(res)
    (ROOT / "metrics.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return res


if __name__ == "__main__":
    run()
    print((ROOT / "metrics.json").read_text())
