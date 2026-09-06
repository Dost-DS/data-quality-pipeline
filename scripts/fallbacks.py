"""Fallback implementations for four adapter interfaces (step 20).

Each satisfies the corresponding Protocol in interfaces.py without the
service the pipeline normally uses. Together with PlainValidator these
give every interface a second implementation, which is what the
substitutability claim in Chapter 4 rests on.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterator, Sequence

import pandas as pd

DRIFT_COLUMNS = ["experiment_id", "window_id", "window_start", "window_end",
                 "window_rows", "feature_name", "feature_type", "metric_name",
                 "metric_value", "p_value", "is_drift"]


class ParquetEventSource:
    """EventSource reading Parquet directly, bypassing Kafka.

    Records are sorted globally by pickup timestamp across all supplied
    files, matching kafka_producer.load_sorted. The raw files are ordered
    by dropoff month, so sorting is what makes the two sources equivalent:
    the single Kafka partition preserves producer order, and the producer
    imposes global event-time order.
    """

    SORT_KEY = "tpep_pickup_datetime"

    def __init__(self, paths: Sequence[str], limit_rows: int | None = None):
        self.paths, self.limit_rows = list(paths), limit_rows

    def _frame(self) -> pd.DataFrame:
        if not self.paths:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(p) for p in self.paths],
                       ignore_index=True)
        df = df.sort_values(self.SORT_KEY).reset_index(drop=True)
        if self.limit_rows:
            df = df.head(self.limit_rows)
        return df

    def windows(self, size: int) -> Iterator[pd.DataFrame]:
        df = self._frame()
        for start in range(0, len(df), size):
            yield df.iloc[start:start + size].reset_index(drop=True)

    def close(self) -> None:
        return None


class CsvMetricStore:
    """MetricStore writing CSV instead of PostgreSQL."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(DRIFT_COLUMNS)

    def write_drift_scores(self, rows: Sequence[tuple]) -> int:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        return len(rows)

    def read_drift_scores(self, experiment_id: str) -> list[dict]:
        with open(self.path, newline="") as f:
            return [r for r in csv.DictReader(f)
                    if r["experiment_id"] == experiment_id]

    def close(self) -> None:
        return None


class JsonRunTracker:
    """RunTracker writing a JSON log instead of MLflow.

    Unlike the MLflow calls in kafka_consumer_drift.py, this records an
    explicit step per metric, so a series can be re-indexed to the window
    that produced it.
    """

    def __init__(self, path: str):
        self.path, self.run = path, None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def start_run(self, run_name: str) -> None:
        self.run = {"run_name": run_name, "params": {}, "metrics": []}

    def log_params(self, params: dict[str, Any]) -> None:
        self.run["params"].update(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.run["metrics"].append({"key": key, "value": float(value),
                                    "step": step})

    def end_run(self) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(self.run) + "\n")
        self.run = None


class SequentialScheduler:
    """Scheduler running tasks in dependency order without Airflow."""

    def __init__(self):
        self.tasks: dict[str, tuple[Any, list[str]]] = {}

    def register(self, name: str, fn: Any, depends_on: Sequence[str] = ()) -> None:
        self.tasks[name] = (fn, list(depends_on))

    def run(self) -> list[str]:
        done, order = set(), []
        while len(done) < len(self.tasks):
            progressed = False
            for name, (fn, deps) in self.tasks.items():
                if name in done or not set(deps) <= done:
                    continue
                fn()
                done.add(name)
                order.append(name)
                progressed = True
            if not progressed:
                raise RuntimeError(
                    f"unsatisfiable dependencies among {sorted(set(self.tasks) - done)}")
        return order
