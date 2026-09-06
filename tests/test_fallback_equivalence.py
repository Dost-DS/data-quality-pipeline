"""Substitution evidence for the remaining four adapters (step 20).

Each test compares a fallback against the behaviour the primary
implementation is relied upon for. The Kafka comparison is marked
integration because it needs a running broker and a populated topic;
the rest run offline.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fallbacks import (CsvMetricStore, JsonRunTracker, ParquetEventSource,
                       SequentialScheduler)

WINDOW = 10_000
SLICE_ROWS = 50_000


# ---------- EventSource ----------

def test_parquet_source_window_boundaries():
    """Windows must be consecutive, ordered, and sized as requested.

    These are the guarantees the single-partition Kafka topic provides
    and on which latency measurement depends.
    """
    files = sorted(glob.glob("data/raw/*.parquet"))
    if not files:
        pytest.skip("data/raw parquet files not present")
    src = ParquetEventSource(files[:1], limit_rows=SLICE_ROWS)
    windows = list(src.windows(WINDOW))
    src.close()

    assert sum(len(w) for w in windows) == SLICE_ROWS
    assert all(len(w) == WINDOW for w in windows[:-1])
    assert len(windows[-1]) <= WINDOW

    ref = (pd.read_parquet(files[0])
             .sort_values("tpep_pickup_datetime")
             .reset_index(drop=True)
             .head(SLICE_ROWS))
    got = pd.concat(windows).reset_index(drop=True)
    pd.testing.assert_series_equal(
        got["tpep_pickup_datetime"], ref["tpep_pickup_datetime"],
        check_names=False)


def test_parquet_source_partial_final_window():
    files = sorted(glob.glob("data/raw/*.parquet"))
    if not files:
        pytest.skip("data/raw parquet files not present")
    src = ParquetEventSource(files[:1], limit_rows=25_000)
    sizes = [len(w) for w in src.windows(WINDOW)]
    assert sizes == [10_000, 10_000, 5_000]


@pytest.mark.integration
def test_kafka_and_parquet_agree_on_boundaries():
    """Same slice through Kafka and through Parquet yields the same windows."""
    kafka = pytest.importorskip("kafka")
    files = sorted(glob.glob("data/raw/*.parquet"))
    if not files:
        pytest.skip("data/raw parquet files not present")
    try:
        c = kafka.KafkaConsumer(
            os.environ.get("KAFKA_TOPIC", "taxi-trips"),
            bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS",
                                             "localhost:29092"),
            auto_offset_reset="earliest", consumer_timeout_ms=20_000,
            value_deserializer=lambda v: json.loads(v.decode()))
    except Exception as exc:
        pytest.skip(f"kafka unavailable: {type(exc).__name__}")

    rows = []
    for msg in c:
        rows.append(msg.value)
        if len(rows) >= SLICE_ROWS:
            break
    c.close()
    if len(rows) < SLICE_ROWS:
        pytest.skip(f"topic held only {len(rows)} records")

    from_kafka = pd.DataFrame(rows)["tpep_pickup_datetime"].astype(str).tolist()
    src = ParquetEventSource(files[:1], limit_rows=SLICE_ROWS)
    from_parquet = pd.concat(list(src.windows(WINDOW)))[
        "tpep_pickup_datetime"].astype(str).tolist()
    src.close()
    assert from_kafka == from_parquet


# ---------- MetricStore ----------

def test_csv_store_roundtrip_preserves_values(tmp_path):
    store = CsvMetricStore(str(tmp_path / "scores.csv"))
    rows = [("E1", "w1", "2024-06-01 10:00:00", "2024-06-01 10:05:00", 10_000,
             "fare_amount", "numerical", "ks", 0.6856, 1.2e-30, True),
            ("E1", "w1", "2024-06-01 10:00:00", "2024-06-01 10:05:00", 10_000,
             "trip_distance", "numerical", "psi", 0.000068, None, False),
            ("E2", "w9", "2024-07-01 00:00:00", "2024-07-01 00:05:00", 9_720,
             "RatecodeID", "categorical", "chi2", 78261.5922, 0.0, True)]
    assert store.write_drift_scores(rows) == 3

    e1 = store.read_drift_scores("E1")
    assert len(e1) == 2
    assert float(e1[0]["metric_value"]) == pytest.approx(0.6856)
    assert float(e1[1]["metric_value"]) == pytest.approx(0.000068)
    assert e1[0]["is_drift"] == "True"
    assert store.read_drift_scores("E2")[0]["feature_name"] == "RatecodeID"
    assert store.read_drift_scores("missing") == []
    store.close()


def test_csv_store_column_order_matches_insert_statement(tmp_path):
    """Column order is the contract shared with the PostgreSQL INSERT."""
    from fallbacks import DRIFT_COLUMNS
    store = CsvMetricStore(str(tmp_path / "s.csv"))
    with open(store.path) as f:
        assert f.readline().strip().split(",") == DRIFT_COLUMNS
    src = open("scripts/kafka_consumer_drift.py").read()
    stmt = src.split("INSERT INTO dq_monitoring.drift_scores", 1)[1]
    cols = stmt.split("(", 1)[1].split(")", 1)[0]
    named = [c.strip() for c in cols.replace("\n", " ").split(",")]
    assert named == DRIFT_COLUMNS


# ---------- RunTracker ----------

def test_json_tracker_records_params_and_stepped_metrics(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    t = JsonRunTracker(path)
    t.start_run("S1_test")
    t.log_params({"window_size": 10_000, "psi_threshold": 0.2})
    for step, v in enumerate([0.31, 0.42, 0.69]):
        t.log_metric("ks_fare_amount", v, step=step)
    t.end_run()

    rec = json.loads(open(path).readline())
    assert rec["run_name"] == "S1_test"
    assert rec["params"]["window_size"] == 10_000
    assert [m["step"] for m in rec["metrics"]] == [0, 1, 2]
    assert rec["metrics"][2]["value"] == pytest.approx(0.69)


def test_json_tracker_step_allows_window_reindexing(tmp_path):
    """The property the MLflow calls in the experiments lack."""
    path = str(tmp_path / "r.jsonl")
    t = JsonRunTracker(path)
    t.start_run("r")
    for step in range(5):
        t.log_metric("psi_trip_distance", step * 0.001, step=step)
    t.end_run()
    metrics = json.loads(open(path).readline())["metrics"]
    assert {m["step"] for m in metrics} == set(range(5))


# ---------- Scheduler ----------

def test_sequential_scheduler_respects_dependencies():
    ran = []
    s = SequentialScheduler()
    s.register("persist", lambda: ran.append("persist"),
               depends_on=["rates", "mix"])
    s.register("rates", lambda: ran.append("rates"))
    s.register("mix", lambda: ran.append("mix"))
    order = s.run()
    assert set(order) == {"rates", "mix", "persist"}
    assert order.index("persist") > order.index("rates")
    assert order.index("persist") > order.index("mix")
    assert ran == order


def test_sequential_scheduler_detects_a_cycle():
    s = SequentialScheduler()
    s.register("a", lambda: None, depends_on=["b"])
    s.register("b", lambda: None, depends_on=["a"])
    with pytest.raises(RuntimeError, match="unsatisfiable"):
        s.run()
