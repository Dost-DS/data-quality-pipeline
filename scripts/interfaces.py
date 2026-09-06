"""Adapter interfaces for the monitoring pipeline (pipeline step 20).

Five boundaries at which a component can be substituted without changing
the meaning of an experiment. Each is a structural protocol rather than a
base class: an implementation satisfies it by having the right methods,
so the existing Kafka, Great Expectations, PostgreSQL, MLflow and Airflow
components conform without being modified.

A substitution claim is only evidenced when the primary and fallback
implementations produce the same result on the same input. The tests in
tests/test_adapter_equivalence.py are what support that claim; these
definitions alone do not.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import pandas as pd


@runtime_checkable
class EventSource(Protocol):
    """Supplies the record stream in event order.

    Ordering is part of the contract, not an implementation detail: the
    experiments measure detection latency against an activation point, so
    a source that reorders records makes latency unmeasurable. This is why
    the Kafka topic uses a single partition.
    """

    def windows(self, size: int) -> Iterator[pd.DataFrame]:
        """Yield consecutive windows of `size` records in arrival order.

        The final window may contain fewer than `size` records.
        """
        ...

    def close(self) -> None: ...


@runtime_checkable
class Validator(Protocol):
    """Applies rule-based expectations to a window."""

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return a result of the form:

            {"success": bool,
             "results": [{"expectation": str,
                          "column": str,
                          "success": bool,
                          "unexpected_count": int}, ...]}

        Ordering of `results` is not part of the contract; the set of
        (expectation, column) pairs and their outcomes is.
        """
        ...


@runtime_checkable
class MetricStore(Protocol):
    """Persists drift scores and validation outcomes."""

    def write_drift_scores(self, rows: Sequence[tuple]) -> int:
        """Persist rows in the drift_scores column order. Returns the count."""
        ...

    def read_drift_scores(self, experiment_id: str) -> list[dict]: ...

    def close(self) -> None: ...


@runtime_checkable
class RunTracker(Protocol):
    """Records parameters and metrics for one experiment run.

    `log_metric` takes an explicit step so that a metric series can be
    re-indexed to the window that produced it. The MLflow implementation
    used in the experiments omitted the step, which is a limitation
    recorded in Chapter 4 rather than a property of this interface.
    """

    def start_run(self, run_name: str) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    def end_run(self) -> None: ...


@runtime_checkable
class Scheduler(Protocol):
    """Executes named tasks in dependency order."""

    def register(self, name: str, fn: Any, depends_on: Sequence[str] = ()) -> None: ...

    def run(self) -> list[str]:
        """Execute all registered tasks. Returns the names in execution order.

        The order must satisfy every declared dependency. Tasks with no
        dependency relation may run in any relative order, so equivalence
        is judged on dependency satisfaction rather than on an exact
        sequence.
        """
        ...
