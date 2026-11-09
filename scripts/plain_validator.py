"""Rule-based Validator fallback without Great Expectations.

Reads the same suite definition Great Expectations reads, so both
implementations derive from one source of truth and differ only in
execution. Supports the four expectation types the taxi suite uses.

Semantics follow Great Expectations:
  - null values are excluded from value-level checks
  - unexpected_count counts failing non-null values, ignoring `mostly`
  - success is (nonnull - unexpected) / nonnull >= mostly, default 1.0
  - a column with no non-null values succeeds vacuously
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

SUITE_PATH = "great_expectations/gx/expectations/taxi_trip_suite.json"


def load_suite(path: str = SUITE_PATH) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d.get("expectations") or d.get("expectation_configurations") or []


def _coerce(series: pd.Series, ref: Any) -> pd.Series:
    """Align dtype with the bound being compared against."""
    if isinstance(ref, str) and len(ref) >= 10 and ref[4] == "-":
        return pd.to_datetime(series, errors="coerce")
    return series


def _outcome(nonnull: int, unexpected: int, mostly: float | None) -> bool:
    if nonnull == 0:
        return True
    m = 1.0 if mostly is None else float(mostly)
    return (nonnull - unexpected) / nonnull >= m


class PlainValidator:
    """Validator implementation using pandas only."""

    def __init__(self, suite: list[dict] | None = None):
        self.suite = suite if suite is not None else load_suite()

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        results = []
        for e in self.suite:
            etype = e.get("expectation_type") or e.get("type")
            kw = e.get("kwargs", {})
            col = kw.get("column")
            results.append(self._one(df, etype, col, kw))
        return {"success": all(r["success"] for r in results), "results": results}

    def _one(self, df, etype, col, kw) -> dict[str, Any]:
        out = {"expectation": etype, "column": col,
               "success": False, "unexpected_count": 0}

        if etype == "expect_column_to_exist":
            out["success"] = col in df.columns
            return out

        if col not in df.columns:
            # Great Expectations cannot evaluate a value-level rule on a
            # missing column; recorded as a failure with no counted values.
            out["success"] = False
            return out

        s = df[col]

        if etype == "expect_column_values_to_not_be_null":
            unexpected = int(s.isna().sum())
            nonnull = int(len(s))          # nulls are the subject here
            out["unexpected_count"] = unexpected
            out["success"] = _outcome(nonnull, unexpected, kw.get("mostly"))
            return out

        if etype == "expect_column_values_to_be_between":
            lo, hi = kw.get("min_value"), kw.get("max_value")
            s = _coerce(s, lo if lo is not None else hi)
            is_dt = pd.api.types.is_datetime64_any_dtype(s)
            valid = s.notna()
            nonnull = int(valid.sum())
            bad = pd.Series(False, index=s.index)
            if lo is not None:
                bad |= valid & (s < (pd.Timestamp(lo) if is_dt else lo))
            if hi is not None:
                bad |= valid & (s > (pd.Timestamp(hi) if is_dt else hi))
            unexpected = int(bad.sum())
            out["unexpected_count"] = unexpected
            out["success"] = _outcome(nonnull, unexpected, kw.get("mostly"))
            return out

        if etype == "expect_column_values_to_be_in_set":
            vs = kw.get("value_set", [])
            valid = s.notna()
            nonnull = int(valid.sum())
            unexpected = int((valid & ~s.isin(vs)).sum())
            out["unexpected_count"] = unexpected
            out["success"] = _outcome(nonnull, unexpected, kw.get("mostly"))
            return out

        raise ValueError(f"unsupported expectation type: {etype!r}")


if __name__ == "__main__":
    suite = load_suite()
    print(f"loaded {len(suite)} expectations from {SUITE_PATH}")
    types = {}
    for e in suite:
        t = e.get("expectation_type") or e.get("type")
        types[t] = types.get(t, 0) + 1
    for t, n in sorted(types.items()):
        print(f"  {t:<44}{n:>3}")

    v = PlainValidator(suite)
    row = {"VendorID": 1, "tpep_pickup_datetime": "2024-06-01T10:00:00",
           "tpep_dropoff_datetime": "2024-06-01T10:20:00", "passenger_count": 1,
           "trip_distance": 3.0, "RatecodeID": 1, "store_and_fwd_flag": "N",
           "PULocationID": 100, "DOLocationID": 200, "payment_type": 1,
           "fare_amount": 15.0, "extra": 0.0, "mta_tax": 0.5, "tip_amount": 2.0,
           "tolls_amount": 0.0, "improvement_surcharge": 0.3,
           "total_amount": 17.8, "congestion_surcharge": 2.5, "Airport_fee": 0.0}
    clean = pd.DataFrame([row] * 20)
    r = v.validate(clean)
    print(f"\nclean window: success={r['success']}, "
          f"{sum(1 for x in r['results'] if not x['success'])} failing expectations")
    for x in r["results"]:
        if not x["success"]:
            print(f"  FAIL {x['expectation']} on {x['column']} "
                  f"(unexpected={x['unexpected_count']})")
