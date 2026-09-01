"""Determine a defensible monitoring window size.

Replays the same clean slice at several window sizes. Because nothing is
injected, every alert raised is a false alert, so the alert rate at each
size measures the background noise the configuration would produce in
production. Metric stability is reported as the coefficient of variation
across windows, which is comparable between metrics on different scales.
"""
import argparse
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import load_slice, reset_topic, produce, PG

WINDOW_SIZES = [1000, 2500, 5000, 10000, 25000]


def register(experiment_id, window_size, rows):
    conn = psycopg2.connect(**PG)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dq_monitoring.experiments
               (experiment_id, scenario, activation_idx, activation_timestamp, ground_truth)
               VALUES (%s,%s,%s,%s,%s)""",
            (experiment_id, "PILOT", None, None,
             psycopg2.extras.Json({"purpose": "window size pilot",
                                   "window_size": window_size,
                                   "total_rows": rows})))
    conn.commit(); conn.close()


def main(months, total_rows, timeout):
    df = load_slice(months, total_rows)
    print(f"pilot slice: {len(df):,} rows, "
          f"{df['tpep_pickup_datetime'].min()} .. {df['tpep_pickup_datetime'].max()}\n")

    for ws in WINDOW_SIZES:
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        eid = f"PILOT_w{ws}_{stamp}_{uuid.uuid4().hex[:4]}"
        print(f"--- window_size={ws:,} ({eid}) ---")
        register(eid, ws, len(df))
        reset_topic()
        produce(df, report_every=0)

        env = os.environ.copy()
        env["EXPERIMENT_ID"] = eid
        t0 = time.time()
        res = subprocess.run(
            [sys.executable, "scripts/kafka_consumer_drift.py",
             "--window-size", str(ws), "--timeout-seconds", str(timeout), "--quiet"],
            env=env, capture_output=True, text=True)
        last = [l for l in res.stdout.strip().split("\n") if l.strip()][-1:]
        print(f"    {last[0] if last else '(no output)'}  [{time.time()-t0:.0f}s]")
        if res.returncode != 0:
            print(f"    STDERR: {res.stderr[-400:]}")
        time.sleep(2)

    print("\npilot complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="*", default=[2])
    ap.add_argument("--total-rows", type=int, default=200000)
    ap.add_argument("--timeout-seconds", type=int, default=45)
    a = ap.parse_args()
    main(a.months, a.total_rows, a.timeout_seconds)
