import numpy as np
import pandas as pd

import data
import features
import model
import policy


def test_labels_and_cap():
    train, test, sensors = data.load()
    assert train.groupby("unit").rul_true.min().eq(0).all()      # every training engine runs to failure
    assert train.rul.max() == data.RUL_CAP and test.rul.max() <= data.RUL_CAP
    assert "s1" not in sensors and "s11" in sensors                # constant sensors dropped


def test_features_have_no_lookahead():
    train, _, sensors = data.load()
    eng = train[train.unit == 1]
    full = features.build(eng, sensors)
    cut = features.build(eng.iloc[:60], sensors)
    cols = [c for c in cut.columns if not c.endswith("_drift")]  # drift uses first 20 cycles only
    pd.testing.assert_frame_equal(full.iloc[:60][cols].reset_index(drop=True), cut[cols].reset_index(drop=True))


def test_nasa_score_punishes_late_more():
    y = np.array([50.0])
    assert model.nasa_score(y, y + 10) > model.nasa_score(y, y - 10)


def test_policy_simulation():
    df = pd.DataFrame({"unit": [1] * 30, "cycle": range(1, 31), "rul_true": list(range(29, -1, -1))})
    early = policy.simulate(df, df.rul_true.to_numpy(), threshold=20, lead=15)
    late = policy.simulate(df, df.rul_true.to_numpy(), threshold=10, lead=15)
    assert early["failures"] == 0 and late["failures"] == 1


def test_model_beats_baseline_quickly():
    train, test, sensors = data.load()
    sub = train[train.unit <= 40]
    f = features.build(sub, sensors)
    cols = features.feature_columns(f)
    m = model.point_model(n_estimators=150).fit(f[cols], sub.rul)
    ft = features.build(test, sensors)
    last = ft.groupby("unit").cycle.idxmax()
    rmse = model.rmse(test.loc[last, "rul"], m.predict(ft.loc[last, cols]))
    assert rmse < model.rmse(test.loc[last, "rul"], np.full(len(last), sub.rul.mean()))
    assert rmse < 20


def test_saved_models_load_and_predict():
    from pathlib import Path
    m = model.load_models(Path(__file__).resolve().parent)
    _, test, sensors = data.load()
    ft = features.build(test[test.unit == 1], sensors)
    p = m["point"].predict(ft[m["cols"]])
    assert len(p) == (test.unit == 1).sum() and np.all(np.isfinite(p))
