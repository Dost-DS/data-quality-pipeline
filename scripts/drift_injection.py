"""Deterministic drift injection for the controlled experiments.

Each scenario takes a chronologically ordered slice, leaves the first
`warmup_rows` records untouched, and applies a specific change from that
point onward. Shift magnitudes are expressed relative to the baseline
statistics computed in scripts/compute_baselines.py, so that each injection
is a stated multiple of the feature's own variability rather than an
arbitrary constant.
"""
import json

import numpy as np
import pandas as pd

SEED = 42
BASELINE_PATH = "great_expectations/reference_baseline.json"

SCENARIOS = ["C0", "S1", "S2", "S3", "S4", "S5"]

SCENARIO_LABELS = {
    "C0": "clean control",
    "S1": "numerical mean shift",
    "S2": "categorical value swap",
    "S3": "unseen category injection",
    "S4": "numerical variance shift",
    "S5": "mixed numerical and categorical",
}


def _baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def mean_shift(df, idx, column="fare_amount", n_std=0.75):
    """Add a fixed offset of n_std baseline standard deviations."""
    b = _baseline()["numerical"][column]
    shift = n_std * b["std"]
    out = df.copy()
    mask = out.index >= idx
    out.loc[mask, column] = out.loc[mask, column] + shift
    return out, {"column": column, "shift_absolute": float(shift),
                 "shift_in_baseline_sd": n_std,
                 "baseline_mean": b["mean"], "baseline_std": b["std"],
                 "affected_rows": int(mask.sum())}


def variance_shift(df, idx, column="trip_distance", scale=2.0):
    """Inflate spread multiplicatively about the local median.

    trip_distance is strictly positive and right-skewed. An additive
    rescaling about the mean drives the lower half of the distribution
    negative, and clipping at zero then creates a large point mass that
    would dominate any drift signal. Scaling the log-deviation about the
    median widens the distribution symmetrically in relative terms while
    keeping every value positive and leaving the median unchanged.
    """
    out = df.copy()
    mask = out.index >= idx
    vals = out.loc[mask, column].astype(float).values

    positive = vals[vals > 0]
    if len(positive) == 0:
        return out, {"column": column, "scale_factor": scale,
                     "affected_rows": 0, "note": "no positive values"}

    med = float(np.median(positive))
    scaled = np.where(vals > 0,
                      med * np.exp(scale * np.log(np.maximum(vals, 1e-6) / med)),
                      vals)
    out.loc[mask, column] = scaled
    return out, {"column": column, "scale_factor": scale,
                 "local_median_held": med,
                 "method": "log-scale about median",
                 "affected_rows": int(mask.sum())}


def value_swap(df, idx, column="payment_type", swap=None):
    """Exchange the frequencies of two common category codes."""
    swap = swap or {1: 2, 2: 1}
    out = df.copy()
    mask = out.index >= idx
    out.loc[mask, column] = out.loc[mask, column].map(lambda v: swap.get(v, v))
    return out, {"column": column, "swap_map": {str(k): v for k, v in swap.items()},
                 "affected_rows": int(mask.sum())}


def unseen_category(df, idx, column="RatecodeID", value=99.0, fraction=0.25):
    """Replace a fraction of records with a code absent from the baseline."""
    out = df.copy()
    rng = np.random.default_rng(SEED)
    eligible = out.index[out.index >= idx]
    n = int(len(eligible) * fraction)
    chosen = rng.choice(eligible, size=n, replace=False)
    out.loc[chosen, column] = value
    return out, {"column": column, "unseen_value": value, "fraction": fraction,
                 "affected_rows": int(n)}


def mixed(df, idx):
    """Numerical and categorical change at the same activation point."""
    out, m1 = mean_shift(df, idx, column="fare_amount", n_std=0.5)
    out, m2 = value_swap(out, idx, column="payment_type")
    return out, {"components": [
        {"type": "mean_shift", **m1},
        {"type": "value_swap", **m2},
    ]}


_FUNCS = {
    "S1": mean_shift,
    "S2": value_swap,
    "S3": unseen_category,
    "S4": variance_shift,
    "S5": mixed,
}

_TARGETS = {
    "C0": [],
    "S1": ["fare_amount"],
    "S2": ["payment_type"],
    "S3": ["RatecodeID"],
    "S4": ["trip_distance"],
    "S5": ["fare_amount", "payment_type"],
}


def apply_scenario(df, scenario, warmup_rows):
    """Return (mutated_df, ground_truth) for the named scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    df = df.reset_index(drop=True)
    if warmup_rows >= len(df):
        raise ValueError(f"warmup_rows ({warmup_rows}) >= rows ({len(df)})")

    activation_ts = df.loc[warmup_rows, "tpep_pickup_datetime"]
    meta = {
        "scenario": scenario,
        "label": SCENARIO_LABELS[scenario],
        "activation_idx": warmup_rows,
        "activation_timestamp": str(activation_ts),
        "injected_targets": _TARGETS[scenario],
        "total_rows": int(len(df)),
        "warmup_rows": int(warmup_rows),
        "seed": SEED,
    }

    if scenario == "C0":
        meta["affected_rows"] = 0
        return df, meta

    out, detail = _FUNCS[scenario](df, warmup_rows)
    meta.update(detail)
    return out, meta
