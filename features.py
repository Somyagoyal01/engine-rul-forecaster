"""Per-engine time-series features. Every feature uses only the current and past cycles (no look-ahead)."""
import numpy as np
import pandas as pd

WINDOWS = (5, 15, 30)


def _slope(x):
    n = len(x)
    if n < 3:
        return 0.0
    t = np.arange(n) - (n - 1) / 2
    return float((t * (x - x.mean())).sum() / (t * t).sum())


def build(df, sensors):
    out = df[["unit", "cycle"]].copy()
    g = df.groupby("unit")
    for s in sensors:
        out[s] = df[s]
        for w in WINDOWS:
            out[f"{s}_mean{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
        out[f"{s}_std15"] = g[s].transform(lambda x: x.rolling(15, min_periods=2).std()).fillna(0)
        out[f"{s}_slope15"] = g[s].transform(lambda x: x.rolling(15, min_periods=3).apply(_slope, raw=True)).fillna(0)
        # drift from the engine's own early-life baseline (first 20 cycles): removes unit-to-unit offsets
        base = g[s].transform(lambda x: x.iloc[:20].mean())
        out[f"{s}_drift"] = out[f"{s}_mean15"] - base
    return out


def feature_columns(feat, raw_only=False, sensors=None):
    if raw_only:
        return list(sensors)
    return [c for c in feat.columns if c not in ("unit",)]
