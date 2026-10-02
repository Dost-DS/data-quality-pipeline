"""Recompute three results the thesis reports without a surviving script.

1. Persistence rule: an alert counts only if the same feature and metric also
   alerted in the two preceding windows (three consecutive windows).
2. Per-feature calibration: an alert is raised when the metric value exceeds
   that feature-metric pair's 99th percentile in the clean control, with the
   same minimum window size of 500 rows. Magnitude only; p-values not used.
3. RatecodeID pre-activation means for PSI and chi-square in C0, S3 and S6,
   to trace the "35%" in Section 7.7.

Read-only. Writes results/diag_persistence.csv and results/diag_p99_calibration.csv.
Run from the repository root:  python scripts/diag_unsourced.py
"""
import os

import numpy as np
import pandas as pd

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")
ACTIVATION = pd.Timestamp("2024-07-03 23:35:25", tz="UTC")
MIN_ROWS = 500

TARGETS = {   # (feature, metric) pairs that carry an injected or propagated change
    "S1": [("fare_amount", "ks"), ("total_amount", "ks"), ("fare_amount", "psi"),
           ("total_amount", "psi")],
    "S2": [("payment_type", "chi2"), ("payment_type", "psi")],
    "S3": [("RatecodeID", "chi2"), ("RatecodeID", "psi")],
    "S4": [("trip_distance", "ks"), ("trip_distance", "psi")],
    "S5": [("fare_amount", "ks"), ("total_amount", "ks"), ("fare_amount", "psi"),
           ("total_amount", "psi"), ("payment_type", "chi2"), ("payment_type", "psi")],
    "S6": [("RatecodeID", "chi2"), ("RatecodeID", "psi")],
}


def load():
    import psycopg2
    conn = psycopg2.connect(**PG)
    df = pd.read_sql("""SELECT id, experiment_id, window_id, window_start, window_end,
                               window_rows, feature_name, metric_name, metric_value,
                               p_value, is_drift
                        FROM dq_monitoring.drift_scores
                        WHERE experiment_id NOT LIKE 'PILOT%'""", conn)
    conn.close()
    return df


def prepare(df):
    df = df.copy()
    df["run"] = df["experiment_id"].str[:2]
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    first_id = df.groupby(["experiment_id", "window_id"])["id"].transform("min")
    df["w"] = first_id.groupby(df["experiment_id"]).rank(method="dense").astype(int)
    df["post"] = df["window_end"] >= ACTIVATION
    return df.sort_values(["experiment_id", "feature_name", "metric_name", "w"])


def persistence(df, k=3):
    g = df.groupby(["experiment_id", "feature_name", "metric_name"])["is_drift"]
    run_ok = df["is_drift"].copy()
    for lag in range(1, k):
        run_ok &= g.shift(lag).fillna(False).astype(bool)
    df = df.assign(persisted=run_ok)

    c0 = df[df["run"] == "C0"]
    print(f"\n=== 1. Persistence ({k} consecutive windows) ===")
    print(f"  C0 alerts: {int(c0['is_drift'].sum()):,} -> {int(c0['persisted'].sum()):,} "
          f"of {len(c0):,} scores ({100*c0['is_drift'].mean():.2f}% -> "
          f"{100*c0['persisted'].mean():.2f}%)")
    rows = []
    for run, pairs in TARGETS.items():
        d = df[df["run"] == run]
        for f, m in pairs:
            x = d[(d["feature_name"] == f) & (d["metric_name"] == m) & d["post"]]
            if not x["is_drift"].any():
                continue
            first = x.loc[x["persisted"], "window_end"].min()
            first_raw = x.loc[x["is_drift"], "window_end"].min()
            r = dict(run=run, feature=f, metric=m, post_windows=len(x),
                     post_alerts=int(x["is_drift"].sum()),
                     post_alerts_persisted=int(x["persisted"].sum()),
                     latency_raw=str(first_raw - ACTIVATION),
                     latency_persisted=str(first - ACTIVATION) if pd.notna(first) else "none")
            rows.append(r)
            print("  " + ", ".join(f"{k2}={v}" for k2, v in r.items()))
    pd.DataFrame(rows).to_csv("results/diag_persistence.csv", index=False)


def calibration(df, pct=99.0):
    c0 = df[df["run"] == "C0"]
    cuts = (c0.dropna(subset=["metric_value"])
              .groupby(["feature_name", "metric_name"])["metric_value"]
              .apply(lambda v: float(np.percentile(v, pct))))
    key = list(zip(df["feature_name"], df["metric_name"]))
    cut = pd.Series([cuts.get(k2, np.inf) for k2 in key], index=df.index)
    df = df.assign(cal=(df["window_rows"] >= MIN_ROWS) & (df["metric_value"] > cut))

    c0 = df[df["run"] == "C0"]
    print(f"\n=== 2. Per-feature calibration at p{pct:g} of the control ===")
    print(f"  C0 alerts: {int(c0['is_drift'].sum()):,} -> {int(c0['cal'].sum()):,} "
          f"of {len(c0):,} scores ({100*c0['cal'].mean():.2f}%)")
    rows = []
    for run, pairs in TARGETS.items():
        d = df[df["run"] == run]
        for f, m in pairs:
            x = d[(d["feature_name"] == f) & (d["metric_name"] == m) & d["post"]]
            r = dict(run=run, feature=f, metric=m, post_windows=len(x),
                     post_alerts_fixed=int(x["is_drift"].sum()),
                     post_alerts_calibrated=int(x["cal"].sum()),
                     calibrated_pct=round(100 * x["cal"].mean(), 1),
                     cut=round(cuts.get((f, m), np.nan), 6))
            rows.append(r)
            print("  " + ", ".join(f"{k2}={v}" for k2, v in r.items()))
    pd.DataFrame(rows).to_csv("results/diag_p99_calibration.csv", index=False)


def ratecode_pre(df):
    print("\n=== 3. RatecodeID pre-activation means (window start before activation) ===")
    d = df[(df["feature_name"] == "RatecodeID") & df["run"].isin(["C0", "S3", "S6"])
           & (df["window_start"] < ACTIVATION)]
    t = d.groupby(["metric_name", "run"])["metric_value"].agg(["count", "mean"])
    for m in ["psi", "chi2"]:
        base = t.loc[(m, "C0"), "mean"]
        for run in ["C0", "S3", "S6"]:
            if (m, run) not in t.index:
                continue
            n, mu = t.loc[(m, run)]
            print(f"  {m:<5}{run}: windows={int(n):,} mean={mu:.6g} "
                  f"change vs C0={100*(mu/base-1):+.1f}%")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare(load())
    print(f"loaded {len(data):,} drift scores from "
          f"{data['experiment_id'].nunique()} runs")
    persistence(data)
    calibration(data)
    ratecode_pre(data)
