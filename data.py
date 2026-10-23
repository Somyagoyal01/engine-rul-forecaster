"""Load NASA C-MAPSS FD001 and build RUL labels."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125  # piecewise-linear target: early life is treated as "healthy", standard for C-MAPSS


SOURCE = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"


def ensure(name):
    """Download a C-MAPSS file from a public mirror on first use (keeps the repo small)."""
    path = DATA / f"{name}.txt"
    if not path.exists():
        import urllib.request
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SOURCE + f"{name}.txt", path)
    return path


def _read(name):
    df = pd.read_csv(ensure(name), sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def useful_sensors(train):
    """Drop sensors/settings that are (near) constant in FD001: they carry no signal."""
    cand = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    return [c for c in cand if train[c].std() > 1e-3 and train[c].nunique() > 2 and not c.startswith("op")]


def load(fd="FD001"):
    train = _read(f"train_{fd}")
    test = _read(f"test_{fd}")
    rul_end = pd.read_csv(ensure(f"RUL_{fd}"), header=None).iloc[:, 0].to_numpy()

    last = train.groupby("unit")["cycle"].transform("max")
    train["rul_true"] = last - train["cycle"]

    tlast = test.groupby("unit")["cycle"].transform("max")
    test["rul_true"] = test["unit"].map(dict(zip(range(1, len(rul_end) + 1), rul_end))) + (tlast - test["cycle"])

    for df in (train, test):
        df["rul"] = df["rul_true"].clip(upper=RUL_CAP)
    return train, test, useful_sensors(train)
