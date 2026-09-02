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


def month_paths(months):
    import glob
    paths = sorted(glob.glob(f"{DATA_DIR}/yellow_tripdata_2024-*.parquet"))
    if months:
        keep = {f"{m:02d}" for m in months}
        paths = [p for p in paths
                 if p.split("-")[-1].replace(".parquet", "") in keep]
    if not paths:
        raise FileNotFoundError(f"no parquet files matched in {DATA_DIR}")
    return paths


def count_rows(paths):
    """Row count per file, read cheaply from Parquet metadata."""
    import pyarrow.parquet as pq
    return [pq.ParquetFile(p).metadata.num_rows for p in paths]


def load_month(path):
    """Load one month, sorted by pickup time.

    The full corpus is never held in memory at once: each monthly file is
    loaded, sorted, produced, and released before the next is read. Since
    the files are already partitioned by month, sorting within a month
    yields globally chronological order across the sequence.
    """
    df = pd.read_parquet(path)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    return df.sort_values("tpep_pickup_datetime").reset_index(drop=True)


def load_slice(months, total_rows):
    """Load a bounded slice into memory. Used for pilots and smoke tests only."""
    paths = month_paths(months)
    frames, remaining = [], total_rows
    for p in paths:
        df = load_month(p)
        if remaining is not None:
            df = df.head(remaining)
            remaining -= len(df)
        frames.append(df)
        if remaining is not None and remaining <= 0:
            break
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("tpep_pickup_datetime").reset_index(drop=True)


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

    print(f"=== {experiment_id} : {SCENARIO_LABELS[scenario]} ===", flush=True)
    t_start = time.time()

    paths = month_paths(months)
    counts = count_rows(paths)
    corpus_rows = sum(counts)
    if total_rows:
        corpus_rows = min(corpus_rows, total_rows)
    print(f"  corpus: {len(paths)} file(s), {corpus_rows:,} rows", flush=True)

    if warmup_rows >= corpus_rows:
        raise SystemExit(f"warmup_rows ({warmup_rows:,}) >= corpus rows ({corpus_rows:,})")

    # Locate the month containing the activation row.
    cum, act_file, act_offset = 0, None, None
    for i, n in enumerate(counts):
        if cum + n > warmup_rows:
            act_file, act_offset = i, warmup_rows - cum
            break
        cum += n
    print(f"  activation in file {act_file + 1}/{len(paths)} at offset {act_offset:,}",
          flush=True)

    reset_topic()
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
        linger_ms=50, batch_size=65536, compression_type="gzip")

    meta = None
    produced = 0
    t_produce = time.time()

    for i, path in enumerate(paths):
        df = load_month(path)
        if total_rows:
            room = total_rows - produced
            if room <= 0:
                break
            df = df.head(room)

        if i == act_file:
            # Inject within this month, at the offset computed above.
            df, meta = apply_scenario(df, scenario, act_offset)
            meta["activation_idx"] = warmup_rows
            meta["total_rows"] = corpus_rows
            meta["warmup_rows"] = warmup_rows
            meta["months"] = months
            meta["activation_file"] = os.path.basename(path)
            register(experiment_id, meta)
            print(f"    injected in {os.path.basename(path)}: "
                  f"activation={meta['activation_timestamp']} "
                  f"targets={meta['injected_targets']}", flush=True)
        elif i > act_file and scenario != "C0":
            # Every month after activation carries the injection from row 0.
            df, _ = apply_scenario(df, scenario, 0)

        for _, row in df.iterrows():
            producer.send(TOPIC, value=to_record(row))
            produced += 1
            if report_every and produced % report_every == 0:
                el = time.time() - t_produce
                pct = 100.0 * produced / corpus_rows
                print(f"    {produced:,}/{corpus_rows:,} ({pct:.1f}%) "
                      f"{produced/el:,.0f} rec/s", flush=True)

        del df
        print(f"    finished {os.path.basename(path)} "
              f"({produced:,} rows so far)", flush=True)

    producer.flush(); producer.close()
    t_prod = time.time() - t_produce
    print(f"    produced {produced:,} rows in {t_prod/60:.1f} min "
          f"({produced/t_prod:,.0f} rec/s)", flush=True)

    if meta is None:
        # C0 with activation past the end, or an unreached activation file.
        meta = {"scenario": scenario, "label": SCENARIO_LABELS[scenario],
                "activation_idx": warmup_rows, "activation_timestamp": None,
                "injected_targets": [], "total_rows": corpus_rows,
                "warmup_rows": warmup_rows, "months": months}
        register(experiment_id, meta)

    print("  consumers:", flush=True)
    t_val = run_consumer("kafka_consumer_validation.py", window_size, timeout, experiment_id)
    t_drift = run_consumer("kafka_consumer_drift.py", window_size, timeout, experiment_id)
    t_schema = run_consumer("kafka_consumer_schema.py", window_size, timeout, experiment_id)

    total = time.time() - t_start
    print(f"\n  timings: produce={t_prod/60:.0f}m validation={t_val/60:.0f}m "
          f"drift={t_drift/60:.0f}m schema={t_schema/60:.0f}m total={total/60:.0f}m",
          flush=True)
    print(f"=== {experiment_id} complete ===", flush=True)
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
