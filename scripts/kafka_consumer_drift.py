"""Compute PSI, Kolmogorov-Smirnov, and Chi-square drift metrics per window.

Each window of records is compared against the persisted baseline. PSI is
computed over the baseline's fixed bin edges; KS compares raw values against
a fixed baseline sample; Chi-square compares category frequencies. An alert
is raised only when the statistical result, a minimum sample size, and a
magnitude condition are all satisfied.
"""
import argparse
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from kafka import KafkaConsumer

import mlflow

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = os.environ.get("KAFKA_TOPIC", "taxi-trips")
GROUP_ID = "drift-consumer"
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID")

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "drift_detection"
USE_MLFLOW = os.environ.get("USE_MLFLOW", "1") == "1"

BASELINE_PATH = "great_expectations/reference_baseline.json"

# Alert thresholds. Justified and calibrated in the window-size pilot.
PSI_THRESHOLD = 0.2          # midpoint of the 0.10/0.25 convention
                             # (Siddiqi, 2006, p. 137); no fixed value has a
                             # stated error rate (Yurdakul & Naranjo, 2020).
                             # Examined critically in Ch. 5.
P_VALUE_THRESHOLD = 0.05
MIN_WINDOW_ROWS = 500        # below this, no alert is raised at all
KS_MIN_EFFECT = 0.1          # D statistic must also exceed this
CHI2_MIN_CRAMERS_V = 0.1     # sqrt(chi2 / (n * (k-1))). Named for Cramer's V
                             # but NOT bounded by 1 here: this is a
                             # goodness-of-fit test against fixed baseline
                             # proportions, not a contingency table, so
                             # Cramer's inequality does not apply. Measured
                             # values exceed 1 in 2,142 feature-windows.
                             # See Ch. 4; Cohen's w is the GoF analogue.
EPS = 1e-6


def load_baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def psi(bin_edges, baseline_counts, values):
    cur_counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bin_edges)
    b = np.asarray(baseline_counts, dtype=float)
    if b.sum() == 0 or cur_counts.sum() == 0:
        return None
    bp = np.clip(b / b.sum(), EPS, None)
    cp = np.clip(cur_counts / cur_counts.sum(), EPS, None)
    return float(np.sum((cp - bp) * np.log(cp / bp)))


def ks(baseline_sample, values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or len(baseline_sample) < 2:
        return None, None
    stat, p = stats.ks_2samp(baseline_sample, v)
    return float(stat), float(p)


def chi2(baseline_counts, values):
    """Chi-square with an explicit guard against zero expected counts.

    An unseen category has an expected count of zero, which makes the
    statistic diverge. The floor keeps the value finite; the returned
    flag records that the assumption was violated so that the result
    can be reported as unreliable rather than taken at face value.
    """
    cur = pd.Series(values, dtype=str).value_counts()
    cats = sorted(set(baseline_counts) | set(cur.index))
    b = np.array([baseline_counts.get(c, 0) for c in cats], dtype=float)
    o = np.array([cur.get(c, 0) for c in cats], dtype=float)
    if b.sum() == 0 or o.sum() == 0:
        return None, None, False, 0

    unseen = int(((b == 0) & (o > 0)).sum())
    expected = b / b.sum() * o.sum()
    floor = max(expected[expected > 0].min() * 1e-3, EPS) if (expected > 0).any() else EPS
    expected = np.clip(expected, floor, None)
    expected = expected / expected.sum() * o.sum()

    try:
        stat, p = stats.chisquare(f_obs=o, f_exp=expected)
        n, k = o.sum(), len(cats)
        cramers_v = float(np.sqrt(stat / (n * max(k - 1, 1)))) if n > 0 else 0.0
        return float(stat), float(p), unseen > 0, cramers_v
    except Exception:
        return None, None, unseen > 0, 0.0


def insert(conn, rows):
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO dq_monitoring.drift_scores
               (experiment_id, window_id, window_start, window_end, window_rows,
                feature_name, feature_type, metric_name, metric_value,
                p_value, is_drift)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)
    conn.commit()


def process(df, baseline, conn, verbose):
    wid = str(uuid.uuid4())
    ws, we, n = df["tpep_pickup_datetime"].min(), df["tpep_pickup_datetime"].max(), len(df)
    enough = n >= MIN_WINDOW_ROWS
    rows, metrics, flagged = [], {}, []

    for feat, ref in baseline["numerical"].items():
        if feat not in df.columns:
            continue
        vals = df[feat].dropna().values
        if len(vals) == 0:
            continue

        v = psi(ref["bin_edges"], ref["hist_counts"], vals)
        if v is not None:
            drift = enough and v > PSI_THRESHOLD
            rows.append((EXPERIMENT_ID, wid, ws, we, n, feat, "numerical",
                         "psi", v, None, drift))
            metrics[f"psi_{feat}"] = v
            if drift:
                flagged.append(f"{feat}/psi")

        d, p = ks(ref["raw_values_sample"], vals)
        if d is not None:
            drift = enough and p < P_VALUE_THRESHOLD and d > KS_MIN_EFFECT
            rows.append((EXPERIMENT_ID, wid, ws, we, n, feat, "numerical",
                         "ks", d, p, drift))
            metrics[f"ks_{feat}"] = d
            if drift:
                flagged.append(f"{feat}/ks")

    for feat, ref in baseline["categorical"].items():
        if feat not in df.columns:
            continue
        vals = df[feat].dropna().astype(str).values
        if len(vals) == 0:
            continue
        bc = ref["value_counts"]

        cats = sorted(set(bc) | set(pd.Series(vals).unique()))
        b = np.array([bc.get(c, 0) for c in cats], dtype=float)
        o = np.array([(vals == c).sum() for c in cats], dtype=float)
        if b.sum() > 0 and o.sum() > 0:
            bp = np.clip(b / b.sum(), EPS, None)
            cp = np.clip(o / o.sum(), EPS, None)
            v = float(np.sum((cp - bp) * np.log(cp / bp)))
            drift = enough and v > PSI_THRESHOLD
            rows.append((EXPERIMENT_ID, wid, ws, we, n, feat, "categorical",
                         "psi", v, None, drift))
            metrics[f"psi_{feat}"] = v
            if drift:
                flagged.append(f"{feat}/psi")

        stat, p, had_unseen, cv = chi2(bc, vals)
        if stat is not None:
            drift = enough and p < P_VALUE_THRESHOLD and cv > CHI2_MIN_CRAMERS_V
            rows.append((EXPERIMENT_ID, wid, ws, we, n, feat, "categorical",
                         "chi2", stat, p, drift))
            metrics[f"chi2_{feat}"] = stat
            metrics[f"cramersv_{feat}"] = cv
            if drift:
                flagged.append(f"{feat}/chi2" + (" [unseen]" if had_unseen else ""))

    insert(conn, rows)
    if USE_MLFLOW:
        for k, val in metrics.items():
            mlflow.log_metric(k, val)

    if verbose:
        print(f"  window {wid[:8]} rows={n:,} {ws} .. {we}")
        print(f"    alerts: {', '.join(flagged) if flagged else 'none'}")
    return len(rows), len(flagged)


def main(window_size, max_windows, timeout, verbose):
    baseline = load_baseline()
    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=BOOTSTRAP, group_id=GROUP_ID,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode()),
        consumer_timeout_ms=timeout * 1000)
    conn = psycopg2.connect(**PG)

    if USE_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_name = EXPERIMENT_ID or f"drift_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({"window_size": window_size,
                           "psi_threshold": PSI_THRESHOLD,
                           "p_value_threshold": P_VALUE_THRESHOLD,
                           "min_window_rows": MIN_WINDOW_ROWS})

    buf, n_win, n_scores, n_alerts = [], 0, 0, 0
    print(f"drift consumer: window_size={window_size}")

    def flush(buf):
        df = pd.DataFrame(buf)
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"],
                                                    errors="coerce")
        return process(df, baseline, conn, verbose)

    for msg in consumer:
        buf.append(msg.value)
        if len(buf) >= window_size:
            s, a = flush(buf); n_scores += s; n_alerts += a
            buf = []; n_win += 1
            if max_windows and n_win >= max_windows:
                break
    if buf:
        s, a = flush(buf); n_scores += s; n_alerts += a; n_win += 1

    conn.close()
    if USE_MLFLOW:
        mlflow.end_run()
    print(f"\n{n_win} windows, {n_scores:,} scores, {n_alerts:,} alerts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--timeout-seconds", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    main(a.window_size, a.max_windows, a.timeout_seconds, not a.quiet)
