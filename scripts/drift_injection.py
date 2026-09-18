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

SCHEMA_BASELINE = "great_expectations/schema_baseline.json"
SCENARIOS = ["C0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]

SCENARIO_LABELS = {
    "C0": "clean control",
    "S1": "numerical mean shift",
    "S2": "categorical value swap",
    "S3": "rare-category frequency shift",
    "S4": "numerical variance shift",
    "S5": "mixed numerical and categorical",
    "S6": "unseen category injection",
    "S7": "schema change, no distributional injection",
}


def _baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def mean_shift(df, idx, column="fare_amount", n_std=0.75, propagate=True):
    """Add a fixed offset of n_std baseline standard deviations.

    total_amount is the arithmetic sum of the metered fare and its
    surcharges, so shifting an addend without shifting the sum would
    produce records that violate an identity holding in every real
    observation. The shift is therefore carried into total_amount as
    well, which is what makes the resulting change in total_amount a
    genuine propagation effect rather than an artefact.
    """
    b = _baseline()["numerical"][column]
    shift = n_std * b["std"]
    out = df.copy()
    mask = out.index >= idx

    out.loc[mask, column] = out.loc[mask, column] + shift

    propagated = []
    if propagate and column != "total_amount" and "total_amount" in out.columns:
        out.loc[mask, "total_amount"] = out.loc[mask, "total_amount"] + shift
        propagated.append("total_amount")

    return out, {"column": column, "shift_absolute": float(shift),
                 "shift_in_baseline_sd": n_std,
                 "baseline_mean": b["mean"], "baseline_std": b["std"],
                 "propagated_to": propagated,
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


def rare_category_amplification(df, idx, column="RatecodeID", value=99.0, fraction=0.25):
    """Amplify a rare existing category to a target share of records.

    RatecodeID 99 (unknown rate code) is present in the baseline at 0.88%.
    This raises it to 25%, a frequency shift rather than a novel value.
    """
    out = df.copy()
    rng = np.random.default_rng(SEED)
    eligible = out.index[out.index >= idx]
    n = int(len(eligible) * fraction)
    chosen = rng.choice(eligible, size=n, replace=False)
    out.loc[chosen, column] = value
    return out, {"column": column, "amplified_value": value, "fraction": fraction,
                 "affected_rows": int(n)}


def mixed(df, idx):
    """Numerical and categorical change at the same activation point."""
    out, m1 = mean_shift(df, idx, column="fare_amount", n_std=0.5)
    out, m2 = value_swap(out, idx, column="payment_type")
    return out, {"components": [
        {"type": "mean_shift", **m1},
        {"type": "value_swap", **m2},
    ]}


def unseen_category(df, idx, column="RatecodeID", value=88.0, fraction=0.25):
    """Replace a fraction of records with a category absent from the baseline.

    The value is checked against the persisted reference baseline before the
    injection is applied, and the function refuses to run if the category is
    already present. S3 was written on the unverified assumption that
    RatecodeID 99 was novel; it is not, and the error was only found after
    the experiments had completed. This scenario fails loudly instead.
    """
    baseline = _baseline()["categorical"][column]["value_counts"]
    present = {str(k) for k in baseline}
    if str(value) in present or str(int(value)) in present:
        raise ValueError(
            f"{column}={value} is already present in the reference baseline "
            f"(categories: {sorted(present)}); choose a value that is absent")

    out = df.copy()
    rng = np.random.default_rng(SEED)
    eligible = out.index[out.index >= idx]
    n = int(len(eligible) * fraction)
    chosen = rng.choice(eligible, size=n, replace=False)
    out.loc[chosen, column] = value
    return out, {"column": column, "unseen_value": value, "fraction": fraction,
                 "affected_rows": int(n),
                 "baseline_categories": sorted(present)}


_FUNCS = {
    "S1": mean_shift,
    "S2": value_swap,
    "S3": rare_category_amplification,
    "S4": variance_shift,
    "S5": mixed,
    "S6": unseen_category,
}

def _schema_baseline_columns():
    """Column names recorded in schema_baseline.json.

    This is the file kafka_consumer_schema.py compares each window against, so
    it is the authority on what counts as a new column. Accepts either a flat
    mapping of column to type or a document with a "columns" member, because
    the guard must not depend on which of the two the artifact happens to use.
    """
    import json
    with open(SCHEMA_BASELINE, encoding="utf-8") as fh:
        doc = json.load(fh)
    cols = doc.get("columns", doc) if isinstance(doc, dict) else doc
    if isinstance(cols, dict):
        return set(cols)
    return set(cols)


def schema_addition(df, idx, column="trip_source_system", value="legacy_ingest_v2"):
    """Introduce a column absent from the persisted schema baseline.

    No monitored feature is altered, so the drift metrics see exactly what they
    would see in the clean control. The schema consumer compares each window's
    observed columns against schema_baseline.json, which is fixed, so once the
    column appears every later window records has_schema_change.

    The column is assigned only from idx onward. Rows before idx are left as
    NaN and the producer drops the key while it is null, so the activation
    point is the first row that carries the new field rather than the first row
    of the file.
    """
    known = _schema_baseline_columns()
    if column in known:
        raise ValueError(
            f"{column!r} is already present in the schema baseline; "
            "S7 requires a column the baseline does not contain")

    if column not in df.columns:
        # object dtype: the column holds strings, and creating it as float64
        # would make the later assignment an incompatible-dtype write.
        df[column] = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    df.loc[df.index[idx:], column] = value
    return df


_TARGETS = {
    "C0": [],
    "S1": ["fare_amount"],
    "S2": ["payment_type"],
    "S3": ["RatecodeID"],
    "S4": ["trip_distance"],
    "S5": ["fare_amount", "payment_type"],
    "S6": ["RatecodeID"],
    # S7 alters no monitored feature; the change is structural.
    "S7": [],
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
    if scenario == "S7":
        return schema_addition(df, idx)

        return df, meta

    out, detail = _FUNCS[scenario](df, warmup_rows)
    meta.update(detail)
    return out, meta
