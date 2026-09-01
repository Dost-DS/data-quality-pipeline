"""Replay NYC TLC taxi records chronologically to a Kafka topic.

Records are sorted by pickup timestamp and published with inter-arrival
times compressed by a configurable factor, so that event-time order is
preserved while a day of trips replays in seconds.
"""
import argparse
import glob
import json
import os
import time

import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = os.environ.get("KAFKA_TOPIC", "taxi-trips")
DATA_DIR = "data/raw"


def load_sorted(months, limit_rows):
    paths = sorted(glob.glob(f"{DATA_DIR}/yellow_tripdata_2024-*.parquet"))
    if months:
        keep = {f"{m:02d}" for m in months}
        paths = [p for p in paths if p.split("-")[-1].replace(".parquet", "") in keep]
    if not paths:
        raise FileNotFoundError(f"no parquet files matched in {DATA_DIR}")

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    if limit_rows:
        df = df.head(limit_rows)
    return df


def to_record(row):
    rec = row.to_dict()
    for k, v in rec.items():
        if isinstance(v, pd.Timestamp):
            rec[k] = v.isoformat()
        elif pd.isna(v):
            rec[k] = None
    return rec


def main(months, limit_rows, speed_factor, max_seconds, report_every):
    print(f"loading and sorting ...", flush=True)
    df = load_sorted(months, limit_rows)
    print(f"{len(df):,} rows spanning "
          f"{df['tpep_pickup_datetime'].min()} to {df['tpep_pickup_datetime'].max()}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=50,
        batch_size=65536,
        compression_type="gzip",
    )

    t0 = time.time()
    prev_event = None
    sent = 0

    for _, row in df.iterrows():
        event_time = row["tpep_pickup_datetime"]
        if prev_event is not None and speed_factor > 0:
            gap = (event_time - prev_event).total_seconds() / speed_factor
            if gap > 0:
                time.sleep(min(gap, 2.0))
        producer.send(TOPIC, value=to_record(row))
        prev_event = event_time
        sent += 1

        if sent % report_every == 0:
            el = time.time() - t0
            print(f"  {sent:,} sent | {el:.1f}s | {sent/el:,.0f} rec/s", flush=True)

        if max_seconds and (time.time() - t0) > max_seconds:
            print(f"  stopping at max_seconds={max_seconds}")
            break

    producer.flush()
    producer.close()
    elapsed = time.time() - t0
    print(f"\n{sent:,} records in {elapsed:.1f}s = {sent/elapsed:,.0f} records/second")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="*", default=None,
                    help="months to replay, e.g. --months 1 2 3 (default: all)")
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--speed-factor", type=float, default=0,
                    help="event-time compression; 0 replays as fast as possible")
    ap.add_argument("--max-seconds", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=100000)
    a = ap.parse_args()
    main(a.months, a.limit_rows, a.speed_factor, a.max_seconds, a.report_every)
