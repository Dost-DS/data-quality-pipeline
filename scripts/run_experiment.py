"""Run one controlled experiment end to end.

Applies a drift scenario to a chronological slice, records the ground truth,
replays the mutated slice through Kafka, then runs the validation, drift, and
schema consumers with the experiment identifier attached to every row they write.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import UnknownTopicOrPartitionError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_injection import apply_scenario, SCENARIOS, SCENARIO_LABELS

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = os.environ.get("KAFKA_TOPIC", "taxi-trips")
PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")
DATA_DIR = "data/raw"


def load_slice(months, total_rows):
    import glob
    paths = sorted(glob.glob(f"{DATA_DIR}/yellow_tripdata_2024-*.parquet"))
    if months:
        keep = {f"{m:02d}" for m in months}
        paths = [p for p in paths if p.split("-")[-1].replace(".parquet", "") in keep]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    if total_rows:
        df = df.head(total_rows)
    return df


def reset_topic():
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP)
    try:
        admin.delete_topics([TOPIC])
    except UnknownTopicOrPartitionError:
        pass
    time.sleep(3)
    admin.create_topics([NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)])
    admin.close()


def to_record(row):
    r = row.to_dict()
    for k, v in r.items():
        if isinstance(v, pd.Timestamp):
            r[k] = v.isoformat()
        elif pd.isna(v):
            r[k] = None
    return r


def produce(df, report_every):
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
        linger_ms=50, batch_size=65536, compression_type="gzip")
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        producer.send(TOPIC, value=to_record(row))
        if report_every and i % report_every == 0:
            print(f"    {i:,} produced ({i/(time.time()-t0):,.0f} rec/s)", flush=True)
    producer.flush(); producer.close()
    el = time.time() - t0
    print(f"    produced {len(df):,} rows in {el:.1f}s ({len(df)/el:,.0f} rec/s)")
    return el


def register(experiment_id, meta):
    conn = psycopg2.connect(**PG)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dq_monitoring.experiments
               (experiment_id, scenario, activation_idx, activation_timestamp, ground_truth)
               VALUES (%s,%s,%s,%s,%s)""",
            (experiment_id, meta["scenario"], meta["activation_idx"],
             meta["activation_timestamp"], psycopg2.extras.Json(meta)))
    conn.commit(); conn.close()


def run_consumer(script, window_size, timeout, experiment_id):
    env = os.environ.copy()
    env["EXPERIMENT_ID"] = experiment_id
    flag = "--batch-size" if "validation" in script else "--window-size"
    cmd = [sys.executable, f"scripts/{script}", flag, str(window_size),
           "--timeout-seconds", str(timeout), "--quiet"]
    if "validation" in script:
        cmd.remove("--quiet")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    el = time.time() - t0
    tail = [l for l in res.stdout.strip().split("\n") if l.strip()][-1:] or ["(no output)"]
    print(f"    {script}: {tail[0]}  [{el:.1f}s]")
    if res.returncode != 0:
        print(f"    STDERR: {res.stderr[-500:]}")
    return el


def main(scenario, months, total_rows, warmup_rows, window_size, timeout, report_every):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    experiment_id = f"{scenario}_{stamp}_{uuid.uuid4().hex[:6]}"

    print(f"=== {experiment_id} : {SCENARIO_LABELS[scenario]} ===")
    t_start = time.time()

    print("  loading slice ...")
    df = load_slice(months, total_rows)
    print(f"    {len(df):,} rows, {df['tpep_pickup_datetime'].min()} .. "
          f"{df['tpep_pickup_datetime'].max()}")

    mutated, meta = apply_scenario(df, scenario, warmup_rows)
    meta["months"] = months
    print(f"    activation at row {meta['activation_idx']:,} "
          f"({meta['activation_timestamp']}), targets={meta['injected_targets']}")

    register(experiment_id, meta)
    reset_topic()
    t_produce = produce(mutated, report_every)

    print("  consumers:")
    t_val = run_consumer("kafka_consumer_validation.py", window_size, timeout, experiment_id)
    t_drift = run_consumer("kafka_consumer_drift.py", window_size, timeout, experiment_id)
    t_schema = run_consumer("kafka_consumer_schema.py", window_size, timeout, experiment_id)

    total = time.time() - t_start
    print(f"\n  timings: produce={t_produce:.0f}s validation={t_val:.0f}s "
          f"drift={t_drift:.0f}s schema={t_schema:.0f}s total={total:.0f}s")
    print(f"=== {experiment_id} complete ===")
    return experiment_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--months", type=int, nargs="*", default=None)
    ap.add_argument("--total-rows", type=int, default=None)
    ap.add_argument("--warmup-rows", type=int, required=True)
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--timeout-seconds", type=int, default=60)
    ap.add_argument("--report-every", type=int, default=100000)
    a = ap.parse_args()
    main(a.scenario, a.months, a.total_rows, a.warmup_rows,
         a.window_size, a.timeout_seconds, a.report_every)
