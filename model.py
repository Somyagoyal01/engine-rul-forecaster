"""Models: point RUL (XGBoost) + P10/P90 quantile models for a prediction interval."""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, n_jobs=4, random_state=0)
GRID = [dict(max_depth=d, min_child_weight=m) for d in (3, 4, 6) for m in (5, 20)]


def point_model(**kw):
    return xgb.XGBRegressor(objective="reg:squarederror", **{**PARAMS, **kw})


def quantile_model(alpha, **kw):
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, **{**PARAMS, **kw})


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def nasa_score(y, p):
    """PHM08 scoring: late predictions (over-estimating RUL) are punished harder than early ones."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def tune(X, y, groups, n_splits=5):
    """Small grid search with engine-wise CV (an engine never appears in train and validation at once)."""
    res = []
    for g in GRID:
        scores = []
        for tr, va in GroupKFold(n_splits).split(X, y, groups):
            m = point_model(**g).fit(X.iloc[tr], y.iloc[tr])
            scores.append(rmse(y.iloc[va], m.predict(X.iloc[va])))
        res.append((float(np.mean(scores)), g))
    res.sort(key=lambda r: r[0])
    return res[0][1], [{"cv_rmse": round(s, 2), **g} for s, g in res]


def oof_predictions(X, y, groups, params, n_splits=5):
    """Out-of-fold point + P10 + P90 for every training cycle (used for the maintenance-policy simulation)."""
    p = np.zeros(len(y))
    lo, hi = p.copy(), p.copy()
    for tr, va in GroupKFold(n_splits).split(X, y, groups):
        p[va] = point_model(**params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        lo[va] = quantile_model(0.1, **params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        hi[va] = quantile_model(0.9, **params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return p, lo, hi


def load_models(art_dir):
    """Load the saved point/P10/P90 models + feature metadata."""
    import json
    from pathlib import Path
    art_dir = Path(art_dir)
    meta = json.loads((art_dir / "meta.json").read_text())
    out = dict(meta)
    import gzip
    for name in ("point", "q10", "q90"):
        mdl = xgb.XGBRegressor()
        gz = art_dir / f"{name}.json.gz"
        mdl.load_model(bytearray(gzip.decompress(gz.read_bytes())) if gz.exists() else art_dir / f"{name}.json")
        out[name] = mdl
    return out
