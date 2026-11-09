"""Validate batches of taxi records against the Great Expectations suite. 

Consumes from Kafka in fixed-size batches, runs the expectation suite over
each batch, writes a summary row per batch, one row per failed expectation,
and one quarantine row per record that failed at least one expectation.
"""
import argparse
import json
import os
import uuid

import pandas as pd
import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer

import great_expectations as gx

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = os.environ.get("KAFKA_TOPIC", "taxi-trips")
GROUP_ID = "validation-consumer"
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID")

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

SUITE = "taxi_trip_suite"
DATASOURCE = "taxi_datasource"
ASSET = "taxi_asset"
BATCH_DEF = "taxi_batch_def"

TIMESTAMP_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]


def get_gx():
    ctx = gx.get_context(mode="file", project_root_dir="great_expectations")
    try:
        ds = ctx.data_sources.get(DATASOURCE)
    except Exception:
        ds = ctx.data_sources.add_pandas(name=DATASOURCE)
    try:
        asset = ds.get_asset(ASSET)
    except Exception:
        asset = ds.add_dataframe_asset(name=ASSET)
    try:
        bd = asset.get_batch_definition(BATCH_DEF)
    except Exception:
        bd = asset.add_batch_definition_whole_dataframe(BATCH_DEF)
    return bd, ctx.suites.get(SUITE)


def safe_dict(row):
    d = row.to_dict()
    for k, v in d.items():
        if isinstance(v, pd.Timestamp):
            d[k] = v.isoformat()
        elif pd.isna(v):
            d[k] = None
    return d


def process(df, batch_def, suite, conn, quarantine_limit):
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite, result_format="COMPLETE")

    failing = {}
    failures = []
    for r in result.results:
        if r.success:
            continue
        cfg = r.expectation_config
        idx = r.result.get("unexpected_index_list") or []
        failures.append({
            "type": cfg.type,
            "column": cfg.kwargs.get("column"),
            "count": r.result.get("unexpected_count", len(idx)),
        })
        for i in idx:
            failing.setdefault(i, []).append(
                {"expectation_type": cfg.type, "column": cfg.kwargs.get("column")})

    total = len(df)
    failed = len(failing)
    batch_id = str(uuid.uuid4())
    ws = df["tpep_pickup_datetime"].min()
    we = df["tpep_pickup_datetime"].max()

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dq_monitoring.validation_runs
               (experiment_id, batch_id, window_start, window_end,
                total_rows, passed_rows, failed_rows, success)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (EXPERIMENT_ID, batch_id, ws, we, total, total - failed, failed,
             result.success))
        run_id = cur.fetchone()[0]

        for f in failures:
            cur.execute(
                """INSERT INTO dq_monitoring.expectation_failures
                   (validation_run_id, expectation_type, column_name,
                    failure_count, details)
                   VALUES (%s,%s,%s,%s,%s)""",
                (run_id, f["type"], f["column"], f["count"],
                 psycopg2.extras.Json({"unexpected_count": f["count"]})))

        for i, (idx, why) in enumerate(failing.items()):
            if quarantine_limit and i >= quarantine_limit:
                break
            try:
                row = df.loc[idx]
            except KeyError:
                continue
            cur.execute(
                """INSERT INTO dq_monitoring.quarantined_rows
                   (validation_run_id, row_data, failed_expectations)
                   VALUES (%s,%s,%s)""",
                (run_id, psycopg2.extras.Json(safe_dict(row)),
                 psycopg2.extras.Json(why)))
    conn.commit()

    print(f"  batch {batch_id[:8]} rows={total:,} failed={failed:,} "
          f"success={result.success}")
    return total, failed


def main(batch_size, max_batches, timeout, quarantine_limit):
    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=BOOTSTRAP, group_id=GROUP_ID,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode()),
        consumer_timeout_ms=timeout * 1000)

    batch_def, suite = get_gx()
    conn = psycopg2.connect(**PG)

    buf, n_batches, tot, fail = [], 0, 0, 0
    print(f"validation consumer: batch_size={batch_size}")

    def flush(buf):
        df = pd.DataFrame(buf)
        for c in TIMESTAMP_COLS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return process(df, batch_def, suite, conn, quarantine_limit)

    for msg in consumer:
        buf.append(msg.value)
        if len(buf) >= batch_size:
            t, f = flush(buf)
            tot += t; fail += f; buf = []; n_batches += 1
            if max_batches and n_batches >= max_batches:
                break

    if buf:
        t, f = flush(buf)
        tot += t; fail += f; n_batches += 1

    conn.close()
    rate = fail / tot if tot else 0
    print(f"\n{n_batches} batches, {tot:,} rows, {fail:,} failed ({rate:.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--timeout-seconds", type=int, default=30)
    ap.add_argument("--quarantine-limit", type=int, default=100,
                    help="max quarantine rows stored per batch; 0 for no limit")
    a = ap.parse_args()
    main(a.batch_size, a.max_batches, a.timeout_seconds, a.quarantine_limit)
