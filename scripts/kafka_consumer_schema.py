"""Detect schema changes per window against the persisted schema baseline.

Compares the set of columns and their coarse types in each window against
the baseline captured from the source files. Reports new columns, missing
columns, and type changes as distinct events.
"""
import argparse
import json
import os
import uuid

import pandas as pd
import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = os.environ.get("KAFKA_TOPIC", "taxi-trips")
GROUP_ID = "schema-consumer"
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID")

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

SCHEMA_BASELINE = "great_expectations/schema_baseline.json"


def coarse_type(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        # ISO-8601 timestamps arrive as strings after JSON serialisation
        if len(v) >= 19 and v[4] == "-" and v[7] == "-" and "T" in v:
            return "datetime"
        return "string"
    return "string"


def observed_schema(records):
    seen = {}
    for rec in records:
        for col, val in rec.items():
            t = coarse_type(val)
            if t is None:
                continue
            seen.setdefault(col, set()).add(t)

    resolved = {}
    for col, types in seen.items():
        # A column carrying both ints and floats is a float column.
        if "float" in types and "integer" in types:
            resolved[col] = "float"
        elif len(types) == 1:
            resolved[col] = next(iter(types))
        else:
            resolved[col] = sorted(types)[0]
    return resolved


def process(records, baseline, conn, verbose):
    wid = str(uuid.uuid4())
    ts = [r.get("tpep_pickup_datetime") for r in records if r.get("tpep_pickup_datetime")]
    ws, we = (min(ts), max(ts)) if ts else (None, None)

    obs = observed_schema(records)
    base_cols, obs_cols = set(baseline), set(obs)

    new = sorted(obs_cols - base_cols)
    missing = sorted(base_cols - obs_cols)
    changed = {c: {"expected": baseline[c], "observed": obs[c]}
               for c in base_cols & obs_cols if baseline[c] != obs[c]}
    changed_any = bool(new or missing or changed)

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dq_monitoring.schema_snapshots
               (experiment_id, window_id, window_start, window_end,
                new_columns, missing_columns, type_changes, has_schema_change)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (EXPERIMENT_ID, wid, ws, we,
             psycopg2.extras.Json(new), psycopg2.extras.Json(missing),
             psycopg2.extras.Json(changed), changed_any))
    conn.commit()

    if verbose:
        print(f"  window {wid[:8]} rows={len(records):,} change={changed_any}")
        if changed_any:
            print(f"    new={new} missing={missing} types={changed}")
    return changed_any


def main(window_size, max_windows, timeout, verbose):
    with open(SCHEMA_BASELINE) as f:
        baseline = json.load(f)

    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=BOOTSTRAP, group_id=GROUP_ID,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode()),
        consumer_timeout_ms=timeout * 1000)
    conn = psycopg2.connect(**PG)

    buf, n_win, n_changed = [], 0, 0
    print(f"schema consumer: window_size={window_size}")

    for msg in consumer:
        buf.append(msg.value)
        if len(buf) >= window_size:
            n_changed += process(buf, baseline, conn, verbose)
            buf = []; n_win += 1
            if max_windows and n_win >= max_windows:
                break
    if buf:
        n_changed += process(buf, baseline, conn, verbose); n_win += 1

    conn.close()
    print(f"\n{n_win} windows, {n_changed} with schema change")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--timeout-seconds", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    main(a.window_size, a.max_windows, a.timeout_seconds, not a.quiet)
